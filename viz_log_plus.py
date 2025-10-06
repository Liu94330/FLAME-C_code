# -*- coding: utf-8 -*-
"""
可视化训练日志（JSONL）— 增强版
用法示例：
python viz_log_plus.py \
  --log ./runs/bert_bilstm/log.jsonl \
  --outdir ./runs/bert_bilstm/figs \
  --smooth ema --alpha 0.3 --patience 5 --topk 3 --fmt png
"""
import os, json, argparse, math
import numpy as np
import matplotlib as mpl
mpl.use("Agg")  # 仅保存图片，无需显示
import matplotlib.pyplot as plt

def read_jsonl(path):
    xs, mse, r2 = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln: 
                continue
            try:
                j = json.loads(ln)
            except Exception:
                continue
            # 兼容大小写/不同字段名
            e   = j.get('epoch', j.get('Epoch', j.get('EPOCH')))
            vm  = j.get('val_mse', j.get('ValMSE', j.get('valMSE')))
            vr2 = j.get('val_r2',  j.get('ValR2',  j.get('valR2')))
            if e is None or vm is None or vr2 is None:
                continue
            xs.append(int(e)); mse.append(float(vm)); r2.append(float(vr2))
    # 按 epoch 排序
    idx = np.argsort(xs)
    xs  = list(np.array(xs)[idx])
    mse = list(np.array(mse)[idx])
    r2  = list(np.array(r2)[idx])
    return xs, mse, r2

def moving_average(arr, window=3):
    if window <= 1: 
        return np.array(arr, dtype=float)
    pad = window // 2
    a = np.pad(arr, (pad, pad), mode='edge')
    ker = np.ones(window) / window
    return np.convolve(a, ker, mode='valid')

def ema(arr, alpha=0.3):
    out = np.zeros(len(arr), dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

def choose_smoother(name, **kw):
    name = (name or 'none').lower()
    if name == 'ema':
        a = float(kw.get('alpha', 0.3))
        return lambda x: ema(np.asarray(x, dtype=float), alpha=a)
    if name in ('ma', 'moving', 'moving_average'):
        w = int(kw.get('window', 3))
        return lambda x: moving_average(np.asarray(x, dtype=float), window=w)
    return lambda x: np.asarray(x, dtype=float)

def best_idx_min(y):
    return int(np.nanargmin(y)) if len(y) else -1

def best_idx_max(y):
    return int(np.nanargmax(y)) if len(y) else -1

def find_early_stop_idx(y, mode='min', patience=5):
    """基于“最后一次提升后连续 patience 轮无提升”给出建议停点（返回最佳点的 idx）。"""
    if not y: 
        return -1
    best = y[0]; best_i = 0
    for i in range(1, len(y)):
        better = (y[i] < best) if mode=='min' else (y[i] > best)
        if better:
            best = y[i]; best_i = i
        if (i - best_i) >= patience:
            return best_i
    return -1  # 未触发

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def save_csv(xs, mse, r2, mse_s, r2_s, outdir):
    import csv
    csv_path = os.path.join(outdir, 'metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'val_mse', 'val_r2', 'mse_smooth', 'r2_smooth'])
        for i in range(len(xs)):
            w.writerow([xs[i], mse[i], r2[i], mse_s[i], r2_s[i]])
    return csv_path

def save_topk(xs, mse, r2, k, outdir):
    k = max(1, min(k, len(xs)))
    mse_idx = np.argsort(mse)[:k]
    r2_idx  = np.argsort(-np.asarray(r2))[:k]
    path = os.path.join(outdir, 'topk.csv')
    import csv
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'epoch_mse', 'val_mse', 'epoch_r2', 'val_r2'])
        for i in range(k):
            w.writerow([i+1, xs[mse_idx[i]], mse[mse_idx[i]], xs[r2_idx[i]], r2[r2_idx[i]]])
    return path

def annotate_best(ax, xs, ys, idx, text_fmt='{:.4f}', color='tab:red'):
    if idx < 0: return
    ax.scatter([xs[idx]], [ys[idx]], s=80, marker='*', zorder=5)
    ax.axvline(xs[idx], linestyle='--', linewidth=1)
    ax.text(xs[idx], ys[idx], f'  best@{xs[idx]}: '+text_fmt.format(ys[idx]),
            va='bottom', ha='left')

def annotate_es(ax, xs, es_idx, label='early-stop@', color='tab:gray'):
    if es_idx < 0: return
    ax.axvline(xs[es_idx], linestyle=':', linewidth=1)
    ax.text(xs[es_idx], ax.get_ylim()[1], f' {label}{xs[es_idx]}',
            va='top', ha='left')

def line_plot(xs, y, y_s, title, ylabel, out_path):
    plt.figure(figsize=(8, 4.6))
    plt.plot(xs, y, marker='o', linewidth=1.5, label='raw')
    if y_s is not None:
        plt.plot(xs, y_s, linewidth=2.0, label='smoothed')
    plt.title(title)
    plt.xlabel('Epoch'); plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def dual_subplots(xs, mse, mse_s, r2, r2_s, best_mse_i, best_r2_i, es_i, out_path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    # MSE
    ax[0].plot(xs, mse, marker='o', linewidth=1.5, label='MSE (raw)')
    if mse_s is not None: ax[0].plot(xs, mse_s, linewidth=2.0, label='MSE (smoothed)')
    annotate_best(ax[0], xs, mse, best_mse_i, text_fmt='{:.6f}')
    annotate_es(ax[0], xs, es_i, label='ES on MSE@')
    ax[0].set_title('Validation MSE'); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('MSE')
    ax[0].grid(True, linestyle='--', alpha=0.4); ax[0].legend()

    # R2
    ax[1].plot(xs, r2, marker='o', linewidth=1.5, label='R² (raw)')
    if r2_s is not None: ax[1].plot(xs, r2_s, linewidth=2.0, label='R² (smoothed)')
    annotate_best(ax[1], xs, r2, best_r2_i, text_fmt='{:.5f}')
    ax[1].set_title('Validation R²'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('R²')
    ax[1].grid(True, linestyle='--', alpha=0.4); ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def scatter_r2_mse(xs, mse, r2, out_path):
    # 随 epoch 渐变色，直观观察“向左下&向上”是否整体改善
    c = np.linspace(0, 1, len(xs))
    plt.figure(figsize=(5.2, 5.2))
    sc = plt.scatter(mse, r2, c=c, cmap='viridis', s=36)
    for i, e in enumerate(xs):
        plt.annotate(str(e), (mse[i], r2[i]), fontsize=8, alpha=0.6)
    plt.gca().invert_xaxis()  # 希望 MSE 越小越靠右？如果不习惯可注释掉
    plt.xlabel('Validation MSE'); plt.ylabel('Validation R²')
    plt.title('R² vs. MSE across epochs')
    plt.grid(True, linestyle='--', alpha=0.4)
    cb = plt.colorbar(sc); cb.set_label('Epoch progression')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def write_summary(xs, mse, r2, best_mse_i, best_r2_i, es_i, outdir):
    p = os.path.join(outdir, 'summary.txt')
    with open(p, 'w', encoding='utf-8') as f:
        f.write('== Training Log Summary ==\n')
        f.write(f'#epochs = {len(xs)}\n')
        if best_mse_i >= 0:
            f.write(f'Best MSE @epoch {xs[best_mse_i]}: {mse[best_mse_i]:.8f}\n')
        if best_r2_i >= 0:
            f.write(f'Best R²  @epoch {xs[best_r2_i]}: {r2[best_r2_i]:.6f}\n')
        if es_i >= 0:
            f.write(f'Early-stop suggestion on MSE at epoch {xs[es_i]} (patience rule)\n')
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', type=str, default='./runs/bert_bilstm/log.jsonl')
    ap.add_argument('--outdir', type=str, default='./runs/bert_bilstm/figs')
    ap.add_argument('--smooth', type=str, default='ema', choices=['none','ema','ma','moving','moving_average'])
    ap.add_argument('--alpha', type=float, default=0.3, help='EMA 系数')
    ap.add_argument('--window', type=int, default=3, help='MA 滑窗')
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--fmt', type=str, default='png', choices=['png','pdf','svg'])
    args = ap.parse_args()

    xs, mse, r2 = read_jsonl(args.log)
    if not xs:
        raise SystemExit(f'空日志或字段缺失：{args.log}')

    ensure_dir(args.outdir)

    smoother = choose_smoother(args.smooth, alpha=args.alpha, window=args.window)
    mse_s = smoother(mse) if args.smooth != 'none' else None
    r2_s  = smoother(r2)  if args.smooth != 'none' else None

    # 以平滑后序列作为“评估曲线”，但标注显示原始值
    mse_eval = mse_s if mse_s is not None else np.asarray(mse, dtype=float)
    r2_eval  = r2_s  if r2_s  is not None else np.asarray(r2,  dtype=float)

    best_mse_i = best_idx_min(mse_eval)
    best_r2_i  = best_idx_max(r2_eval)
    es_i       = find_early_stop_idx(list(mse_eval), mode='min', patience=args.patience)

    # 导出 CSV & TopK & 摘要
    csv_path   = save_csv(xs, mse, r2, mse_s if mse_s is not None else ['']*len(xs),
                          r2_s if r2_s is not None else ['']*len(xs), args.outdir)
    topk_path  = save_topk(xs, mse, r2, args.topk, args.outdir)
    sum_path   = write_summary(xs, mse, r2, best_mse_i, best_r2_i, es_i, args.outdir)

    # 单图：MSE / R2
    line_plot(xs, mse, mse_s if mse_s is not None else None,
              f'Validation MSE (smooth={args.smooth})', 'MSE',
              os.path.join(args.outdir, f'val_mse.{args.fmt}'))
    line_plot(xs, r2, r2_s if r2_s is not None else None,
              f'Validation R² (smooth={args.smooth})', 'R²',
              os.path.join(args.outdir, f'val_r2.{args.fmt}'))

    # 双子图：并排对比 + 标注最佳与早停建议
    dual_subplots(xs, mse, mse_s if mse_s is not None else None,
                  r2, r2_s if r2_s is not None else None,
                  best_mse_i, best_r2_i, es_i,
                  os.path.join(args.outdir, f'curves_side_by_side.{args.fmt}'))

    # 散点：R² vs MSE（随 epoch 渐变色）
    scatter_r2_mse(xs, mse, r2, os.path.join(args.outdir, f'r2_vs_mse.{args.fmt}'))

    print('[OK] 输出文件：')
    print('  -', csv_path)
    print('  -', topk_path)
    print('  -', sum_path)
    print('  -', os.path.join(args.outdir, f'val_mse.{args.fmt}'))
    print('  -', os.path.join(args.outdir, f'val_r2.{args.fmt}'))
    print('  -', os.path.join(args.outdir, f'curves_side_by_side.{args.fmt}'))
    print('  -', os.path.join(args.outdir, f'r2_vs_mse.{args.fmt}'))

if __name__ == '__main__':
    main()
