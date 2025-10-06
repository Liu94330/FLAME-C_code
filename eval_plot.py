# -*- coding: utf-8 -*-
import json, os, argparse
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', type=str, default='./runs/bert_bilstm/log.jsonl')
    ap.add_argument('--out', type=str, default='./runs/bert_bilstm/curve.png')
    args = ap.parse_args()

    xs, mse, r2 = [], [], []
    for line in open(args.log, 'r', encoding='utf-8'):
        j = json.loads(line)
        xs.append(j['epoch']); mse.append(j['val_mse']); r2.append(j['val_r2'])

    plt.figure()
    plt.plot(xs, mse, marker='o')
    plt.title('Validation MSE vs. Epoch')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.grid(True)
    plt.savefig(args.out.replace('.png', '_mse.png'), bbox_inches='tight')

    plt.figure()
    plt.plot(xs, r2, marker='o')
    plt.title('Validation R2 vs. Epoch')
    plt.xlabel('Epoch'); plt.ylabel('R2'); plt.grid(True)
    plt.savefig(args.out.replace('.png', '_r2.png'), bbox_inches='tight')

    print('Saved plots to', args.out.replace('.png', '_*.png'))

if __name__ == '__main__':
    main()
