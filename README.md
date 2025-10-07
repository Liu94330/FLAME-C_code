# FLAME-C_code (Pointwise + Pairwise)

A **runnable** PyTorch project for the Emotion Understanding Dataset (EUD) v1.0 (the zip you uploaded).
It builds pointwise regression samples and pairwise ranking samples, then trains a BERT-based scorer to predict
three dimensions per response: **understanding, empathy, helpfulness**. Validation monitors the mean score.

# Environment

| Component     | Spec                          |
|---------------|-------------------------------|
| GPU           | NVIDIA RTX 2080 Ti            |
| CPU           | 12 cores                      |
| RAM           | 43 GB                         |
| OS            | Ubuntu 20.04                  |
| Python        | 3.10                          |
| PyTorch       | 1.11.0                        |
| CUDA          | ≤ 12.4                        |

# Datasets

Emotion Understanding Dataset (EUD)
Source: Kaggle – Emotion Understanding Dataset (CC0)
Link: kaggle.com/datasets/kiddkaito/emotion-understanding-dataset

## 1. Setup

```bash
# create env (example)
conda create -n eud python=3.10 -y
conda activate eud

# go to project folder
cd FLAME-C_code

# install deps
pip install -r requirements.txt
```

## 2. Prepare data

You can read **directly from the zip** (fastest to start) or from the extracted folder.

### Option A) From ZIP (recommended for first run)
```bash
python data_prep.py --zip_path /mnt/data/emotion_understanding_dataset_v1_0_large.zip --out_dir ./data
```

### Option B) From extracted folder (if you unzip it yourself)
```bash
unzip emotion_understanding_dataset_v1_0_large.zip -d /path/to/
python data_prep.py --data_root /path/to/eud_v1_0_large --out_dir ./data
```

This creates:
```
data/pointwise.jsonl
data/pairwise.jsonl
data/stats.json
```

## 3. Train

```bash
python train.py \
  --data_dir ./data \
  --model_name hfl/chinese-roberta-wwm-ext \
  --arch bert_mlp \
  --epochs 6 --max_len 256 \
  --batch_pw 16 --batch_pr 12 --batch_eval 32 \
  --lr 2e-5 --weight_decay 0.01 \
  --w_pointwise 1.0 --w_pairwise 0.5 \
  --num_workers 0 \
  --out_dir ./runs/bert_mlp

python train.py \
  --data_dir ./data \
  --model_name hfl/chinese-roberta-wwm-ext \
  --arch bert_concat \
  --epochs 6 --max_len 256 \
  --batch_pw 16 --batch_pr 12 --batch_eval 32 \
  --lr 2e-5 --weight_decay 0.01 \
  --w_pointwise 1.0 --w_pairwise 0.5 \
  --num_workers 0 \
  --out_dir ./runs/bert_concat

python train.py \
  --data_dir ./data \
  --model_name hfl/chinese-roberta-wwm-ext \
  --arch bert_se_attn \
  --epochs 6 --max_len 256 \
  --batch_pw 16 --batch_pr 12 --batch_eval 32 \
  --lr 2e-5 --weight_decay 0.01 \
  --w_pointwise 1.0 --w_pairwise 0.5 \
  --num_workers 0 \
  --out_dir ./runs/bert_se_attn

python train.py \
  --data_dir ./data \
  --model_name hfl/chinese-roberta-wwm-ext \
  --arch bert_bilstm \
  --epochs 6 --max_len 256 \
  --batch_pw 16 --batch_pr 12 --batch_eval 32 \
  --lr 2e-5 --weight_decay 0.01 \
  --w_pointwise 1.0 --w_pairwise 0.5 \
  --num_workers 0 \
  --out_dir ./runs/bert_bilstm

python train.py \
  --data_dir ./data \
  --model_name hfl/chinese-roberta-wwm-ext \
  --arch bert_trf \
  --epochs 6 --max_len 256 \
  --batch_pw 16 --batch_pr 12 --batch_eval 32 \
  --lr 2e-5 --weight_decay 0.01 \
  --w_pointwise 1.0 --w_pairwise 0.5 \
  --num_workers 0 \
  --out_dir ./runs/bert_trf

```

Notes:
- Uses GPU automatically if available.
- Validation is a random 5% split **by (dialog, turn)**.
- Checkpoints are saved under `--out_dir`, with the best model at `best.pt`.

## 4. Evaluate & Plot

```bash
python viz_dataset_heatmaps.py \
  --data_dir /path/to/emotion_understanding_dataset_v1_0_large \
  --outdir runs/diag
```
This script generates multiple visualization figures for analyzing the Emotion Understanding Dataset v1.0 (Large).
All output images are saved under the specified --outdir directory.

Generated figures include:
corr_heatmap.png: Correlation heatmap among U / E / H dimensions, showing inter-dimensional relationships.
hist2d_u_e.png: 2D density of U vs E.
hist2d_u_h.png: 2D density of U vs H.
hist2d_e_h.png: 2D density of E vs H.
hist2d_len_mean.png: 2D density between text length and mean emotion score.
hists_scores.png: Histograms of each emotion dimension (U, E, H) showing score distributions.

```bash
python viz_log_plus.py \
  --log ./runs/bert_mlp/log.jsonl \
  --outdir ./runs/bert_mlp/figs \
  --smooth ema --alpha 0.3 --patience 5 --topk 3 --fmt png
```
This script visualizes and summarizes the training log stored in JSONL format.
It reads validation metrics (MSE and R²) across epochs, applies smoothing, detects early-stop points,
and saves multiple plots and summary files under the specified --outdir directory.

Generated outputs include:
val_mse.png: Validation MSE curve (raw and smoothed).
val_r2.png: Validation R² curve (raw and smoothed).
curves_side_by_side.png: Side-by-side comparison of MSE and R² curves with annotated best epochs and early-stop suggestion.
r2_vs_mse.png: Scatter plot of R² vs MSE across epochs (color-coded by epoch index).
metrics.csv: CSV file containing raw and smoothed metric values.
topk.csv: Top-k epochs ranked by best MSE and best R².
summary.txt: Text summary including best MSE, best R², total epochs, and suggested early-stop epoch.

These plots and summary files provide a quick and comprehensive overview of model training dynamics,
helping to identify convergence behavior, optimal checkpoints, and potential overfitting trends.


## 5. Inference (example)

To score a new user text + a few candidate replies, load `best.pt` and run the model:

```python
import torch, json
from transformers import AutoTokenizer
from models import BertScorer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
model = BertScorer('hfl/chinese-roberta-wwm-ext').to(device)
state = torch.load('./runs/eud_roberta/best.pt', map_location=device)
model.load_state_dict(state['model']); model.eval()

user = "最近工作压力太大，经常失眠。"
cands = [
  "我理解你的感受，先深呼吸，我们可以从睡前仪式开始改善。",
  "建议你服用一些药物。",
  "哈哈，别想太多啦，喝点酒就行。"
]
with torch.no_grad():
    for t in cands:
        enc = tok(user, t, truncation=True, max_length=256, return_tensors='pt')
        out = model(**{k:v.to(device) for k,v in enc.items()})
        u,e,h = out.preds.squeeze(0).tolist()
        print(t, "| U,E,H =", [round(u,2), round(e,2), round(h,2)])
```

## 6. What the model is learning?

- **Pointwise**: direct regression to human 1–7 ratings per candidate.
- **Pairwise**: when the data says “A better than B on understanding/helpfulness”, we apply a logistic ranking loss on the
  relevant dimension. This improves relative ordering of styles.

## 7. Files

- `data_prep.py`: build `pointwise.jsonl` and `pairwise.jsonl` from either the zip or extracted folder.
- `models.py`: BERT encoder with a 3-dim regression head + pairwise loss helper.
- `train.py`: training loop mixing pointwise and pairwise; validates on mean score.
- `eval_plot.py`: quick plotting from the training log.
- `dataset_schema.json`: the official schema copied from the dataset for reference.
- `requirements.txt`: pinned high-level deps.

---

If you hit CUDA/torch install issues, try installing a matching torch build from https://pytorch.org/ first,
then run `pip install -r requirements.txt` again (it will skip torch if already satisfied).
