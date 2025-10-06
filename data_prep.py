# -*- coding: utf-8 -*-
"""
Build pointwise & pairwise training samples from the Emotion Understanding Dataset (EUD) v1.0.

Usage:
  python data_prep.py --data_root /path/to/eud_v1_0_large --out_dir ./data
  # (or read directly from the zip)
  python data_prep.py --zip_path /mnt/data/emotion_understanding_dataset_v1_0_large.zip --out_dir ./data

It creates:
  - data/pointwise.jsonl  (one item per (turn, candidate))
  - data/pairwise.jsonl   (one item per pairwise comparison)
  - data/stats.json       (counts, label distributions)
"""
import os, json, argparse, zipfile, random, re
from tqdm import tqdm

def iter_dialog_jsonl(root_or_zip: str):
    if os.path.isdir(root_or_zip):
        # path to eud_v1_0_large
        dlg_dir = os.path.join(root_or_zip, 'train', 'dialogs')
        for name in sorted(os.listdir(dlg_dir)):
            if name.endswith('.jsonl'):
                with open(os.path.join(dlg_dir, name), 'r', encoding='utf-8') as f:
                    for line in f:
                        yield json.loads(line)
    else:
        # assume zip file
        with zipfile.ZipFile(root_or_zip, 'r') as z:
            jsonl_names = [n for n in z.namelist() if n.startswith('eud_v1_0_large/train/dialogs/') and n.endswith('.jsonl')]
            for n in sorted(jsonl_names):
                with z.open(n) as f:
                    for raw in f:
                        yield json.loads(raw.decode('utf-8'))

def build_pointwise_and_pairwise(root_or_zip: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    pw_file = open(os.path.join(out_dir, 'pointwise.jsonl'), 'w', encoding='utf-8')
    pr_file = open(os.path.join(out_dir, 'pairwise.jsonl'), 'w', encoding='utf-8')

    num_pw, num_pr = 0, 0
    style_text_cache = {}

    for ex in tqdm(iter_dialog_jsonl(root_or_zip), desc='Preparing'):
        user_text = ex['user']['text']
        turn_meta = {
            'dialog_id': ex['dialog_id'],
            'turn_id': ex['turn_id'],
            'lang': ex.get('lang', 'mix'),
            'difficulty': ex.get('difficulty', 'easy'),
            'scenario_key': ex.get('scenario_key', ''),
            'participant_id': ex.get('participant_id', '')
        }

        # Map style -> candidate text for this sample
        s2txt = {cand['style']: cand['text'] for cand in ex['ai_candidates']}
        # Save per-candidate ratings as pointwise
        for pc in ex['ratings']['per_candidate']:
            style = pc['style']
            item = {
                **turn_meta,
                'style': style,
                'user_text': user_text,
                'cand_text': s2txt.get(style, ''),
                'understanding': pc['understanding'],
                'empathy': pc['empathy'],
                'helpfulness': pc['helpfulness'],
                'target_mean': float(pc['understanding'] + pc['empathy'] + pc['helpfulness']) / 3.0
            }
            pw_file.write(json.dumps(item, ensure_ascii=False) + '\n')
            num_pw += 1

        # Pairwise comparisons: label 1 if A better than B on 'better_on'
        for pr in ex['ratings']['pairwise']:
            A, B = pr['A'], pr['B']
            better_on = pr['better_on']  # understanding | helpfulness
            item = {
                **turn_meta,
                'better_on': better_on,
                'user_text': user_text,
                'A_style': A, 'A_text': s2txt.get(A, ''),
                'B_style': B, 'B_text': s2txt.get(B, ''),
                'label': 1  # always treat as A > B
            }
            pr_file.write(json.dumps(item, ensure_ascii=False) + '\n')
            num_pr += 1

    pw_file.close(); pr_file.close()
    stats = {'pointwise': num_pw, 'pairwise': num_pr}
    json.dump(stats, open(os.path.join(out_dir, 'stats.json'), 'w'), indent=2, ensure_ascii=False)
    print('Done.', stats)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='', help='Path to eud_v1_0_large directory')
    ap.add_argument('--zip_path', type=str, default='', help='Path to emotion_understanding_dataset_v1_0_large.zip')
    ap.add_argument('--out_dir', type=str, required=True)
    args = ap.parse_args()

    src = args.data_root if args.data_root else args.zip_path
    assert src, 'Provide either --data_root or --zip_path'
    build_pointwise_and_pairwise(src, args.out_dir)

if __name__ == '__main__':
    main()
