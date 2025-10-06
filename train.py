# -*- coding: utf-8 -*-
import os, json, random, argparse
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# 来自扩展后的 models.py
from models import get_model, ARCHS, pairwise_rank_loss


# ---------------------- Datasets ----------------------
class PointwiseDS(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_len: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(
            r['user_text'], r['cand_text'],
            truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        x = {k: v.squeeze(0) for k, v in enc.items()}
        y = torch.tensor(
            [r['understanding'], r['empathy'], r['helpfulness']],
            dtype=torch.float
        )
        return x, y


class PairwiseDS(Dataset):
    DIM_MAP = {'understanding': 0, 'helpfulness': 2}

    def __init__(self, rows: List[Dict], tokenizer, max_len: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        encA = self.tok(
            r['user_text'], r['A_text'],
            truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        encB = self.tok(
            r['user_text'], r['B_text'],
            truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        A = {k: v.squeeze(0) for k, v in encA.items()}
        B = {k: v.squeeze(0) for k, v in encB.items()}
        dim_idx = self.DIM_MAP[r['better_on']]  # int
        label = torch.tensor(1, dtype=torch.long)       # A 优于 B
        return A, B, dim_idx, label


# ---------------------- Utilities ----------------------
def load_jsonl(path: str) -> List[Dict]:
    return [json.loads(x) for x in open(path, 'r', encoding='utf-8')]

def set_seed(sd=42):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)


# ---------------------- Train ----------------------
def train_loop(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Load preprocessed data
    pointwise = load_jsonl(os.path.join(args.data_dir, 'pointwise.jsonl'))
    pairwise  = load_jsonl(os.path.join(args.data_dir, 'pairwise.jsonl'))

    # 防止空数据导致 DataLoader 报错
    if (not pointwise) or (not pairwise):
        raise RuntimeError(
            f"Empty dataset: pointwise={len(pointwise)}, pairwise={len(pairwise)}. "
            f"Please re-run data_prep.py with the correct --zip_path/--data_root."
        )

    # simple random split by (dialog_id, turn_id) for val
    dialogs = sorted({(r['dialog_id'], r['turn_id']) for r in pointwise})
    random.shuffle(dialogs)
    n_val = max(200, int(0.05 * len(dialogs)))
    val_set = set(dialogs[:n_val])

    pw_train = [r for r in pointwise if (r['dialog_id'], r['turn_id']) not in val_set]
    pw_val   = [r for r in pointwise if (r['dialog_id'], r['turn_id']) in val_set]
    pr_train = [r for r in pairwise  if (r['dialog_id'], r['turn_id']) not in val_set]

    print(f"Pointwise: train={len(pw_train)} val={len(pw_val)} | Pairwise(train)={len(pr_train)}")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = get_model(args.arch, args.model_name, args.dropout).to(device)

    ds_pw_tr = PointwiseDS(pw_train, tok, args.max_len)
    ds_pw_va = PointwiseDS(pw_val, tok, args.max_len)
    ds_pr_tr = PairwiseDS(pr_train, tok, args.max_len)

    dl_pw_tr = DataLoader(ds_pw_tr, batch_size=args.batch_pw, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(not args.cpu))
    dl_pw_va = DataLoader(ds_pw_va, batch_size=args.batch_eval, shuffle=False,
                          num_workers=args.num_workers)
    dl_pr_tr = DataLoader(ds_pr_tr, batch_size=args.batch_pr, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(not args.cpu))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * (len(dl_pw_tr) + len(dl_pr_tr))
    sched = get_linear_schedule_with_warmup(optim, int(0.05 * total_steps), total_steps)

    best_val = 1e9
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(total=len(dl_pw_tr) + len(dl_pr_tr), desc=f"Epoch {epoch}")
        total_loss = 0.0

        # alternate over pointwise and pairwise
        it_pw = iter(dl_pw_tr); it_pr = iter(dl_pr_tr)
        n_loop = max(len(dl_pw_tr), len(dl_pr_tr))

        for _ in range(n_loop):
            # ---- pointwise step ----
            try:
                xb, yb = next(it_pw)
                xb = {k: v.to(device) for k, v in xb.items()}
                yb = yb.to(device)
                out = model(**xb, targets=yb)
                loss_pw = out.loss
                (args.w_pointwise * loss_pw).backward()
                total_loss += float(loss_pw.detach().cpu())
            except StopIteration:
                pass

            # ---- pairwise step（批内混合维度，交给 pairwise_rank_loss 处理）----
            try:
                A, B, dim_idx, label = next(it_pr)
                A = {k: v.to(device) for k, v in A.items()}
                B = {k: v.to(device) for k, v in B.items()}
                label = label.to(device)

                outA = model(**A)
                outB = model(**B)

                loss_pr = pairwise_rank_loss(outA.preds, outB.preds, dim_idx, label)
                (args.w_pairwise * loss_pr).backward()
                total_loss += float(loss_pr.detach().cpu())
            except StopIteration:
                pass

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            pbar.update(1)

        pbar.close()

        # ---- validation on pointwise mean ----
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in dl_pw_va:
                xb = {k: v.to(device) for k, v in xb.items()}
                out = model(**xb)
                y_pred.extend(out.preds.mean(-1).cpu().tolist())
                y_true.extend(yb.mean(-1).cpu().tolist())

        mse = mean_squared_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        print(f"[Val] MSE={mse:.4f}  R2={r2:.4f}  (epoch {epoch})")

        # save checkpoint
        ckpt = os.path.join(args.out_dir, f"checkpoint-epoch{epoch}.pt")
        torch.save({'model': model.state_dict(), 'args': vars(args)}, ckpt)
        if mse < best_val:
            best_val = mse
            torch.save({'model': model.state_dict(), 'args': vars(args)},
                       os.path.join(args.out_dir, "best.pt"))
        # write a small log
        with open(os.path.join(args.out_dir, 'log.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps({'epoch': epoch, 'val_mse': mse, 'val_r2': r2}) + '\n')

    print("Training done. Best val MSE:", best_val)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, required=True,
                    help='Folder containing pointwise.jsonl, pairwise.jsonl')
    ap.add_argument('--model_name', type=str, default='hfl/chinese-roberta-wwm-ext')
    ap.add_argument('--arch', type=str, default='bert_bilstm', choices=list(ARCHS.keys()),
                    help=f"Model architecture: {list(ARCHS.keys())}")
    ap.add_argument('--max_len', type=int, default=256)
    ap.add_argument('--batch_pw', type=int, default=16)
    ap.add_argument('--batch_pr', type=int, default=12)
    ap.add_argument('--batch_eval', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--w_pointwise', type=float, default=1.0)
    ap.add_argument('--w_pairwise', type=float, default=0.5)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--out_dir', type=str, default='./runs/eud_simple')
    return ap


if __name__ == '__main__':
    args = build_argparser().parse_args()
    train_loop(args)
