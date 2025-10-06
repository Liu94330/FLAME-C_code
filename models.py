# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


@dataclass
class ForwardOut:
    preds: torch.Tensor           # (B, 3) = [understanding, empathy, helpfulness]
    pooled: torch.Tensor          # (B, H') representation fed to head
    loss: Optional[torch.Tensor]  # scalar if targets provided


# ---------- 小组件 ----------
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class SEChannelGate(nn.Module):
    """ Squeeze-and-Excitation：按 channel 做门控 """
    def __init__(self, hidden: int, reduction: int = 4):
        super().__init__()
        mid = max(8, hidden // reduction)
        self.fc = nn.Sequential(
            nn.Linear(hidden, mid),
            nn.ReLU(),
            nn.Linear(mid, hidden),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B,T,H) ; mask: (B,T) 1=valid 0=pad
        if mask is None:
            m = x.mean(dim=1)  # (B,H)
        else:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
            m = (x * mask.unsqueeze(-1).float()).sum(dim=1) / denom
        gate = self.fc(m).unsqueeze(1)  # (B,1,H)
        return x * gate


class AttentionPool(nn.Module):
    """ 可掩码的时间注意力池化 """
    def __init__(self, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        s = self.score(self.drop(x)).squeeze(-1)  # (B,T)
        if mask is not None:
            s = s.masked_fill(mask == 0, float("-inf"))
        alpha = torch.softmax(s, dim=1).unsqueeze(-1)  # (B,T,1)
        pooled = (alpha * x).sum(dim=1)               # (B,H)
        return pooled, alpha


# ========== 架构 #1：MLP on [CLS] ==========
class BertMLPScorer(nn.Module):
    def __init__(self, model_name: str = 'hfl/chinese-roberta-wwm-ext', dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = MLPHead(H, H, dropout)
        self.mse = nn.MSELoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, targets: Optional[torch.Tensor]=None) -> ForwardOut:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = out.last_hidden_state[:, 0]               # [CLS]
        logits = self.head(self.drop(pooled))
        loss = self.mse(logits, targets) if targets is not None else None
        return ForwardOut(preds=logits, pooled=pooled, loss=loss)


# ========== 架构 #2：ConcatPool（CLS + Mean + Max） ==========
class BertConcatPoolScorer(nn.Module):
    def __init__(self, model_name: str = 'hfl/chinese-roberta-wwm-ext', dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(3 * H, H)
        self.head = MLPHead(H, H, dropout)
        self.mse = nn.MSELoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, targets: Optional[torch.Tensor]=None) -> ForwardOut:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        X = out.last_hidden_state  # (B,T,H)
        cls = X[:, 0]
        mean = (X * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True).clamp_min(1)
        maxv = X.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf")).amax(1)
        rep = torch.cat([cls, mean, maxv], dim=-1)
        rep = torch.relu(self.proj(self.drop(rep)))
        logits = self.head(self.drop(rep))
        loss = self.mse(logits, targets) if targets is not None else None
        return ForwardOut(preds=logits, pooled=rep, loss=loss)


# ========== 架构 #3：SE-Channel + AttentionPool ==========
class BertSEAttnScorer(nn.Module):
    def __init__(self, model_name: str = 'hfl/chinese-roberta-wwm-ext', dropout: float = 0.2, reduction: int = 4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size
        self.se = SEChannelGate(H, reduction=reduction)
        self.pool = AttentionPool(H, dropout=dropout)
        self.head = MLPHead(H, H, dropout)
        self.mse = nn.MSELoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, targets: Optional[torch.Tensor]=None) -> ForwardOut:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        X = self.se(out.last_hidden_state, attention_mask)
        rep, _ = self.pool(X, attention_mask)
        logits = self.head(rep)
        loss = self.mse(logits, targets) if targets is not None else None
        return ForwardOut(preds=logits, pooled=rep, loss=loss)


# ========== 架构 #4：BiLSTM Head + AttentionPool ==========
class BertBiLSTMScorer(nn.Module):
    def __init__(self, model_name: str = 'hfl/chinese-roberta-wwm-ext', dropout: float = 0.2, lstm_units: int = 256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size
        self.lstm = nn.LSTM(input_size=H, hidden_size=lstm_units, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.pool = AttentionPool(2 * lstm_units, dropout=dropout)
        self.head = MLPHead(2 * lstm_units, 2 * lstm_units, dropout)
        self.mse = nn.MSELoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, targets: Optional[torch.Tensor]=None) -> ForwardOut:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        X, _ = self.lstm(out.last_hidden_state)
        rep, _ = self.pool(X, attention_mask)
        logits = self.head(rep)
        loss = self.mse(logits, targets) if targets is not None else None
        return ForwardOut(preds=logits, pooled=rep, loss=loss)


# ========== 架构 #5：Transformer-Encoder Head + AttentionPool ==========
class BertTransEncScorer(nn.Module):
    def __init__(self, model_name: str = 'hfl/chinese-roberta-wwm-ext', dropout: float = 0.2,
                 nhead: int = 8, num_layers: int = 1, ff_mult: int = 4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size
        ff = min(4096, ff_mult * H)
        enc_layer = nn.TransformerEncoderLayer(d_model=H, nhead=nhead,
                                               dim_feedforward=ff, dropout=dropout,
                                               batch_first=True, activation="gelu")
        self.trf = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = AttentionPool(H, dropout=dropout)
        self.head = MLPHead(H, H, dropout)
        self.mse = nn.MSELoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, targets: Optional[torch.Tensor]=None) -> ForwardOut:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        X = out.last_hidden_state
        key_padding_mask = (attention_mask == 0)
        X = self.trf(X, src_key_padding_mask=key_padding_mask)
        rep, _ = self.pool(X, attention_mask)
        logits = self.head(rep)
        loss = self.mse(logits, targets) if targets is not None else None
        return ForwardOut(preds=logits, pooled=rep, loss=loss)


# ---------- Pairwise 排序损失（确保同设备 & 批内混合维度） ----------
def pairwise_rank_loss(scores_A: torch.Tensor, scores_B: torch.Tensor,
                       dim_idx, label: torch.Tensor) -> torch.Tensor:
    """
    scores_*: (B, 3)
    dim_idx: int 或 (B,) Tensor；按样本选择比较维度
    label: (B,) {0,1}；1 表示 A > B
    """
    if not isinstance(dim_idx, torch.Tensor):
        dim_idx = torch.tensor(dim_idx, device=scores_A.device)
    else:
        dim_idx = dim_idx.to(scores_A.device)
    label = label.to(scores_A.device).float()

    dim_idx = dim_idx.long().view(-1, 1)          # (B,1)
    sA = scores_A.gather(1, dim_idx).squeeze(1)   # (B,)
    sB = scores_B.gather(1, dim_idx).squeeze(1)   # (B,)
    return F.binary_cross_entropy_with_logits(sA - sB, label)


# ---------- 工厂 & 默认导出 ----------
ARCHS = {
    "bert_mlp": BertMLPScorer,
    "bert_concat": BertConcatPoolScorer,
    "bert_se_attn": BertSEAttnScorer,
    "bert_bilstm": BertBiLSTMScorer,
    "bert_trf": BertTransEncScorer,
}

def get_model(arch: str = "bert_bilstm", model_name: str = "hfl/chinese-roberta-wwm-ext",
              dropout: float = 0.2, **kwargs) -> nn.Module:
    cls = ARCHS[arch]
    return cls(model_name=model_name, dropout=dropout, **kwargs)

# 向后兼容：train.py 可 from models import BertScorer
BertScorer = BertMLPScorer
