# -*- coding: utf-8 -*-
"""
可视化 AttentionPool 的 token 权重（隐藏中文刻度，另存 CSV）
用法示例：
  export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
  python viz_attn_tokens.py \
    --arch bert_bilstm \
    --ckpt runs/bert_bilstm/best.pt \
    --model_name hfl/chinese-roberta-wwm-ext \
    --text1 "用户：这个功能总是出错，我有点生气。" \
    --text2 "助手：抱歉给你带来困扰，我们一步步排查。" \
    --out runs/bert_bilstm/attn.png
"""
import os, argparse, csv
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from models import get_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', type=str, default='bert_bilstm',
                    choices=['bert_mlp','bert_concat','bert_se_attn','bert_bilstm','bert_trf'])
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--model_name', type=str, default='hfl/chinese-roberta-wwm-ext')
    ap.add_argument('--text1', type=str, default='用户：这个功能总是出错，我有点生气。')
    ap.add_argument('--text2', type=str, default='助手：抱歉给你带来困扰，我们一步步排查。')
    ap.add_argument('--max_len', type=int, default=128)
    ap.add_argument('--out', type=str, default='./attn.png')
    args = ap.parse_args()

    # 尝试离线加载（若本地已有缓存会生效）
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    enc = tok(args.text1, args.text2,
              truncation=True, max_length=args.max_len,
              padding='max_length', return_tensors='pt')

    # 构建模型并加载 ckpt（兼容 {'model':..., 'args':...} 以及形状过滤）
    model = get_model(arch=args.arch, model_name=args.model_name)
    sd_raw = torch.load(args.ckpt, map_location='cpu')
    sd = sd_raw.get('model', sd_raw)
    own = model.state_dict()
    sd_f = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
    missing = [k for k in own if k not in sd_f]
    model.load_state_dict(sd_f, strict=False)
    model.eval()
    print(f"Loaded {len(sd_f)}/{len(own)} params; skipped {len(missing)}.")

    if not hasattr(model, 'pool'):
        raise RuntimeError(f'架构 {args.arch} 无 AttentionPool，无法可视化 token 权重。'
                           f'建议使用 bert_bilstm / bert_se_attn / bert_trf。')

    cache = {}
    def hook(_, __, out):
        # AttentionPool.forward 返回 (pooled, alpha)
        cache['alpha'] = out[1].detach().cpu().squeeze(0).squeeze(-1).numpy()
    h = model.pool.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(input_ids=enc['input_ids'],
                  attention_mask=enc['attention_mask'],
                  token_type_ids=enc.get('token_type_ids'))

    alpha = cache['alpha']
    tokens = tok.convert_ids_to_tokens(enc['input_ids'][0])
    attn_mask = enc['attention_mask'][0].bool().numpy()
    tokens = [t for t, m in zip(tokens, attn_mask) if m]
    alpha = alpha[:len(tokens)]

    # 画图（隐藏中文刻度，避免字体告警）
    plt.figure(figsize=(max(6, len(tokens) * 0.2), 3))
    plt.bar(range(len(tokens)), alpha)
    plt.title('AttentionPool token weights')
    plt.xticks([])  # 关键：不显示中文刻度
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches='tight')

    # 另存 CSV（token, weight）
    csv_path = args.out.rsplit('.', 1)[0] + '.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['token', 'weight'])
        for t, a in zip(tokens, alpha.tolist()):
            w.writerow([t, a])

    print(f"Saved figure to: {args.out}")
    print(f"Saved CSV to   : {csv_path}")

if __name__ == '__main__':
    main()
