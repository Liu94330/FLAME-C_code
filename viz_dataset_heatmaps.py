# -*- coding: utf-8 -*-
"""
可视化英文数据集（pointwise.jsonl）：
1) U/E/H 相关性热力图 corr_heatmap.png
2) 两两维度 2D 密度热力图（直方图）: hist2d_u_e.png / hist2d_u_h.png / hist2d_e_h.png
3) 文本长度 vs 平均分 的 2D 密度图：hist2d_len_mean.png
4) 各维度分布直方图：hists_scores.png
用法：
  python viz_dataset_heatmaps.py --data_dir /path/to/data --outdir runs/bert_bilstm/diag
"""
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt

def load_jsonl(p):
    with open(p, 'r', encoding='utf-8') as f:
        return [json.loads(x) for x in f]

def text_len(en, tok_field='input_ids'):
    # 如果你有 tokenizer 的长度，直接改这里；默认为词级近似
    return len(en.split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, required=True, help='folder with pointwise.jsonl')
    ap.add_argument('--outdir', type=str, required=True)
    ap.add_argument('--pointwise_name', type=str, default='pointwise.jsonl')
    ap.add_argument('--bins', type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load_jsonl(os.path.join(args.data_dir, args.pointwise_name))
    if not rows:
        raise RuntimeError('pointwise.jsonl is empty or not found.')

    # 取出英文文本与三维标签
    U, E, H, L = [], [], [], []
    for r in rows:
        U.append(float(r['understanding']))
        E.append(float(r['empathy']))
        H.append(float(r['helpfulness']))
        # 近似长度：用户+候选连接后的英文词数（按空格分）
        L.append(text_len((r.get('user_text','') + ' ' + r.get('cand_text','')).strip()))

    U = np.array(U); E = np.array(E); H = np.array(H); L = np.array(L)
    M = np.vstack([U, E, H])  # (3, N)

    # 1) 相关性热力图
    corr = np.corrcoef(M)
    plt.figure(figsize=(4.2,3.8))
    im = plt.imshow(corr, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = ['Understanding','Empathy','Helpfulness']
    plt.xticks(range(3), ticks, rotation=30)
    plt.yticks(range(3), ticks)
    # 标注数值
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{corr[i,j]:.2f}", ha='center', va='center', color='white' if abs(corr[i,j])>0.5 else 'black')
    plt.title('Correlation Heatmap (U/E/H)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'corr_heatmap.png'), bbox_inches='tight')

    # 2) 两两维度 2D 密度（直方图）
    pairs = [
        (U, E, 'U vs E', 'hist2d_u_e.png'),
        (U, H, 'U vs H', 'hist2d_u_h.png'),
        (E, H, 'E vs H', 'hist2d_e_h.png'),
    ]
    for x, y, ttl, fn in pairs:
        plt.figure(figsize=(4.2,3.8))
        plt.hist2d(x, y, bins=args.bins, cmap='magma')
        plt.xlabel(ttl.split(' vs ')[0])
        plt.ylabel(ttl.split(' vs ')[1])
        plt.title(f'2D Density: {ttl}')
        plt.colorbar(label='count', fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, fn), bbox_inches='tight')

    # 3) 文本长度 vs 平均分（2D 密度）
    mean_score = (U + E + H) / 3.0
    plt.figure(figsize=(4.6,3.8))
    plt.hist2d(L, mean_score, bins=[min(args.bins, max(10, int(L.max()/5))), args.bins], cmap='plasma')
    plt.xlabel('Text length (approx. words)')
    plt.ylabel('Mean score (U/E/H)')
    plt.title('2D Density: Length vs Mean score')
    plt.colorbar(label='count', fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'hist2d_len_mean.png'), bbox_inches='tight')

    # 4) 各维度直方图
    plt.figure(figsize=(7.8,3.2))
    for i,(arr,name) in enumerate([(U,'Understanding'),(E,'Empathy'),(H,'Helpfulness')],1):
        plt.subplot(1,3,i)
        plt.hist(arr, bins=args.bins, edgecolor='black', linewidth=0.4)
        plt.title(name); plt.xlabel('score'); plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'hists_scores.png'), bbox_inches='tight')

    print('Saved to:', args.outdir)

if __name__ == '__main__':
    main()
