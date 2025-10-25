# Automated Abstract Screening Pipeline

This repository provides an end-to-end, *reproducible* pipeline for high-recall abstract screening in biomedical literature.

It supports two major use cases:

1. **Supervised evaluation (with gold labels):**
   - You already have binary relevance labels (0/1) for each record.
   - You want to train/evaluate models, choose an operating threshold, estimate workload reduction, and report WSS@95 etc.
   - We provide three tiers:
     - `MiniLM_LR` (cheap baseline),
     - `LightHybrid` (MiniLM + semantic + TF-IDF priors),
     - `HeavyHybrid` (fine-tuned SciBERT + priors).

2. **Label-free / weakly supervised deployment:**
   - You have *no labels yet*.
   - You want a ranked list of which abstracts to screen first.
   - We provide:
     - a purely unsupervised fusion of domain keywords + semantic similarity,
     - a weakly supervised mode that bootstraps pseudo-labels from keyword rules.

---

## 0. TL;DR Quickstart

```bash
# 1) create a Python env
pip install -r requirements_supervised.txt

# 2) run the supervised MiniLM baseline
python supervised/src/cli_pipeline_minilm_only.py \
  --input_csv your_input.csv \
  --output_dir supervised/run_minilm \
  --recall_target 0.95

# 3) inspect outputs
ls supervised/run_minilm
cat supervised/run_minilm/summary_test_minilm.csv
```

The `summary_test_*.csv` file already contains recall, precision, AUC, AP, WSS@95, Recall@k, etc.
You can cite that directly in Results.

---

## 1. Input data format

All pipelines expect a CSV with at least:

- `title` (string)
- `abstract` (string)
- `label` (0/1 for supervised mode; not required for unsupervised mode)
- `pmid` or `doi` (optional but strongly recommended; used for manifesting IDs)

Example (minimal):

```csv
pmid,title,abstract,label
12345,"Phase-contrast MRI of CSF flow","We measured intracranial CSF transport ...",1
12346,"Cardiac perfusion MRI","Left ventricle perfusion quantification ...",0
```

### IMPORTANT
- PubMed retrieval, English-only filtering, and deduplication (PMID/DOI match, Title+Year fallback) happen *before* this CSV is created.  
- This repo intentionally does **not** include PubMed scraping logic. The goal is to keep the benchmark reproducible without redistributing copyrighted abstracts.

---

## 2. Supervised pipelines (`/supervised/src/`)

These require `label` in the CSV.

All supervised pipelines follow the same experimental protocol:

1. **Deterministic stratified split (70/10/20):**
   - The script does a single split into:
     - 70% train
     - 10% validation
     - 20% test  
   stratified by `label`, using a fixed random seed (42).
   - A `split_manifest.csv` file is written so you know which record ID went where.

2. **Training (train split only):**
   - Model parameters, TF-IDF vocabulary, centroid priors, etc. are fit *only* on the training split → no leakage.

3. **Threshold selection (validation split only):**
   - We choose an operating threshold `τ*` that **maximizes precision subject to \`Recall ≥ R\`**, with `R = --recall_target` (default 0.95).
   - Ties are broken in favor of a higher threshold.
   - This mirrors a realistic workflow: “hit ≥95% recall but within that, keep workload small.”

4. **Final evaluation (test split only):**
   - `τ*` is *frozen*.
   - We evaluate the frozen model on the untouched test split.
   - We report binary metrics (precision, recall/sensitivity, specificity, F1, accuracy) and ranking/workload metrics.

5. **Outputs saved per run:**
   - `split_manifest.csv`  
     Mapping of each record ID to `train` / `val` / `test`.
   - `test_scores_*.csv`  
     Per-item score, predicted label at `τ*`, and rank.
   - `summary_test_*.csv`  
     One-row summary: precision, recall, specificity, F1, accuracy, ROC AUC, Average Precision, WSS@95, Recall@k...
   - `manifest_*.json`  
     Seed, hyperparameters, threshold `τ*`, model choices, etc. → this is what you cite for reproducibility.

That is exactly what we describe in the manuscript sections on:
- leakage control,
- recall-oriented operating point (`Recall ≥ 0.95`),
- workload estimation (WSS@95, Recall@k),
- reproducibility (frozen threshold, persisted manifests).

### 2.1 MiniLM baseline (`cli_pipeline_minilm_only.py`)

- Embeds each `[title + abstract]` using `sentence-transformers/all-MiniLM-L6-v2` (384-dim). Embeddings are L2-normalized.
- Trains a class-weighted logistic regression (`class_weight="balanced"`) on the train split.
- Uses the predicted probability of class 1 (relevant) as the score.

Command:

```bash
python supervised/src/cli_pipeline_minilm_only.py \
  --input_csv ./your_input.csv \
  --output_dir ./supervised/run_minilm \
  --recall_target 0.95
```

Outputs will appear under `./supervised/run_minilm/`:
- `split_manifest.csv`
- `test_scores_minilm.csv`
- `summary_test_minilm.csv`
- `manifest_minilm.json`

The summary CSV is meant to drop directly into tables.

### 2.2 Light hybrid (`cli_pipeline_light.py`)

Adds *unsupervised priors* on top of the MiniLM baseline:

1. **LR probability component:** same logistic regression prob as baseline.
2. **Centroid similarity prior:** cosine similarity of each abstract's MiniLM embedding to the centroid of *positive training abstracts only*.
3. **Domain TF-IDF prior:** fit a TF-IDF model on the train split; compute cosine similarity of each abstract to a domain vocabulary (glymphatic, perivascular, CSF circulation, ASL, phase-contrast MRI, etc.).

We min–max normalize each component based on the training distribution and fuse them with non-negative weights that sum to 1.

You can pass custom fusion weights:

```bash
python supervised/src/cli_pipeline_light.py \
  --input_csv ./your_input.csv \
  --output_dir ./supervised/run_light \
  --recall_target 0.95 \
  --weights 0.5,0.3,0.2 \
  --domain_terms "glymphatic,perivascular,pvs,csf,diffusion,asl,flair,spectroscopy"
```

Exactly the same outputs are produced:
- `split_manifest.csv`
- `test_scores_lighthybrid.csv`
- `summary_test_lighthybrid.csv`
- `manifest_lighthybrid.json`

These runs let you say “unsupervised priors improve early yield / workload at fixed recall”.

### 2.3 Heavy hybrid (`cli_pipeline_heavy.py`)

This is the “upper bound” tier.

Steps:
1. **Fine-tune SciBERT** (`allenai/scibert_scivocab_uncased`) as a binary classifier with class-weighted cross-entropy, on the train split only.
2. Optionally apply Platt scaling on the validation split (`--calibrate_platt`) to get calibrated probabilities.
3. Extract contextual embeddings for each abstract (CLS / pooled output from the fine-tuned SciBERT).
4. Build the same *centroid similarity prior* and *TF-IDF prior* as in the light hybrid, again training those priors strictly on the train split.
5. Fuse the three normalized components:
   - `fused = α * transformer_component + (1-α)/2 * centroid_component + (1-α)/2 * tfidf_component`
   with `--alpha` (default 0.7) giving priority to the transformer signal.
6. Pick `τ*` on validation using the same Recall ≥ 0.95 rule.
7. Evaluate frozen `τ*` on test, export the same manifests and metrics.

Example:

```bash
python supervised/src/cli_pipeline_heavy.py \
  --input_csv ./your_input.csv \
  --output_dir ./supervised/run_heavy \
  --recall_target 0.95 \
  --alpha 0.7 \
  --max_len 256 \
  --epochs 4 \
  --batch_size 8 \
  --lr 2e-5 \
  --warmup_ratio 0.1 \
  --calibrate_platt \
  --domain_terms "glymphatic,perivascular,pvs,csf,diffusion,asl,flair,spectroscopy"
```

Outputs:
- `split_manifest.csv`
- `test_scores_heavyhybrid.csv`
- `summary_test_heavyhybrid.csv`
- `manifest_heavyhybrid.json`
- plus the HuggingFace checkpoint directory `hf_ckpt/` so the fine-tuned model is reproducible.

---

## 3. Metrics you get out of the box

Each supervised run’s `summary_test_*.csv` includes:

- **precision**
- **recall** (sensitivity)
- **specificity**
- **f1**
- **accuracy**
- **roc_auc** (AUC-ROC)
- **average_precision** (area under the PR curve / AP)
- **tp / fp / tn / fn**
- **threshold** (`τ*` chosen on validation)
- **k_R_for_recall_target**  
  How many top-ranked abstracts you’d need to screen to recover ≥R of known-relevant studies.
- **WSS_at_recall_target**  
  `1 - k_R/N`, “Work Saved over Sampling @R” (often reported as WSS@95 if R=0.95).
- **recall@50, recall@100, ...**  
  Early-yield recall.

These are the exact workload / recall-oriented metrics we talk about in the manuscript.  
You can literally lift these numbers into tables.

---

## 4. Unsupervised / label-free pipelines (`/unsupervised/src/`)

These do **not** require `label`. They are meant for “day 0 deployment”, where no gold labels exist yet.

### 4.1 `cli_pipeline_unsupervised.py`

- Builds two signals:
  1. TF-IDF similarity to a domain vocabulary (glymphatic / perivascular / CSF flow / MRI flow terms).
  2. Semantic similarity (MiniLM embeddings) between each abstract and the same domain vocabulary treated as a “query”.
- Min–max normalize both.
- Fuse them with user-chosen weights (default 0.5/0.5).
- Rank all abstracts by that fused score.
- Output:
  - `ranked_unsupervised.csv` (record_id, rank, fused_score, components, title, abstract)
  - `topk_binary.csv`: the top K (default 200) marked `screen_flag=1`.

Command:

```bash
python unsupervised/src/cli_pipeline_unsupervised.py \
  --input_csv ./your_input.csv \
  --output_dir ./unsupervised/run_unsup \
  --domain_terms "glymphatic,perivascular,pvs,csf,diffusion,asl,flair,spectroscopy" \
  --weights 0.5,0.5 \
  --top_k 200
```

Interpretation:
- “Here are the 200 abstracts you should manually screen first.”

### 4.2 `cli_pipeline_weaksup.py`

- Creates pseudo-labels from keyword rules:
  - positive if it matches `--pos_keywords`
  - negative if it matches `--neg_keywords`
- Trains a class-weighted logistic regression on MiniLM embeddings using those pseudo-labels.
- Scores and ranks all abstracts.
- Outputs:
  - `ranked_weaksup.csv`
  - `topk_binary.csv` with the top K suggestions to screen first.

Command:

```bash
python unsupervised/src/cli_pipeline_weaksup.py \
  --input_csv ./your_input.csv \
  --output_dir ./unsupervised/run_weaksup \
  --pos_keywords "glymphatic|perivascular|perivascular space|pvs|csf circulation|phase-contrast mri|4d flow|asl|flair" \
  --neg_keywords "cardiac|renal|liver" \
  --top_k 200
```

This mode approximates “weak supervision” / “rule-based pseudolabels” that you describe in the manuscript.

---

## 5. SAFE-style stopping + bootstrap CIs

In the manuscript we also describe:
- multi-signal SAFE-style stopping rules (e.g. ≥15% screened, all sentinel studies found, ≤1% new yield in last 200, ≥40 consecutive irrelevants, stabilization of top-50),
- bootstrap confidence intervals on metrics,
- paired tests (DeLong, McNemar).

Those pieces are **evaluation layers that operate on the ranked lists and per-record outputs** generated here:
- They’re not required to *train or run* the models.
- They *consume* the exported CSVs (e.g. `test_scores_*.csv`, `ranked_unsupervised.csv`) and simulate an active human-in-the-loop screening workflow.

This repository focuses on the core reproducible pipeline: deterministic splits, leakage control, operating-threshold selection under a recall constraint, workload metrics, and fully persisted manifests.

---

## 6. Installation

Two requirements files are provided:

### Supervised / heavy (needs PyTorch + transformers)
```bash
pip install -r requirements_supervised.txt
```

### Unsupervised-only (no SciBERT fine-tuning)
```bash
pip install -r requirements_unsupervised.txt
```

`requirements_supervised.txt` is a superset of `requirements_unsupervised.txt`, so if you install the supervised one you can run everything.

---

## 7. Reproducibility and what to report in a paper

Every supervised run writes:
- `split_manifest.csv` → which ID was in train / val / test.
- `manifest_*.json` → random seed, hyperparameters, chosen threshold `τ*`, fusion weights / alpha, calibration flag.
- `summary_test_*.csv` → all headline metrics including WSS@95 and Recall@k.

These three artifacts are what a reviewer needs to:
1. verify no data leakage,
2. understand how the operating point was chosen (recall-first),
3. reproduce the ranking/workload claims.

When you describe the method in a manuscript, you can truthfully say:
- “All priors (TF-IDF vocabularies, centroid priors) are derived strictly from the training split.”
- “Operating thresholds and fusion weights are selected only on validation.”
- “Final performance is reported exclusively on the held-out test split, frozen before evaluation.”
- “We persist split manifests, thresholds, and per-record scores to allow byte-for-byte reproduction.”

That matches the exported files here.

---
