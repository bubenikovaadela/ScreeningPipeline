#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from metrics import (
    select_threshold,
    compute_binary_metrics,
    compute_workload_metrics,
)

RANDOM_STATE = 42
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"


def normalize_text(title, abstract):
    t = (str(title) if pd.notna(title) else "").strip()
    a = (str(abstract) if pd.notna(abstract) else "").strip()
    combo = (t + " " + a).lower()
    combo = " ".join(combo.split())
    return combo


def build_ids(df):
    for col in ["pmid", "PMID", "doi", "DOI"]:
        if col in df.columns:
            return df[col].astype(str).tolist()
    return [f"row_{i}" for i in range(len(df))]


class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = self.encodings["token_type_ids"][idx]
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


class WeightedCETrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # [batch, 2]

        if self.class_weights is None:
            loss_fct = torch.nn.CrossEntropyLoss()
        else:
            cw = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=cw)

        loss = loss_fct(logits, labels)
        if return_outputs:
            return loss, outputs
        return loss


def softmax_prob(logits):
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exps / exps.sum(axis=1, keepdims=True)
    return probs[:, 1]


def platt_scale_fit(val_logits, y_val):
    margins = val_logits[:, 1] - val_logits[:, 0]
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(margins.reshape(-1, 1), y_val.astype(int))
    return clf


def platt_scale_apply(clf, logits):
    margins = logits[:, 1] - logits[:, 0]
    return clf.predict_proba(margins.reshape(-1, 1))[:, 1]


def minmax_fit_apply(train_vec, other_vecs):
    lo = np.min(train_vec)
    hi = np.max(train_vec)
    if hi > lo:
        def scale(v):
            return (v - lo) / (hi - lo)
    else:
        def scale(v):
            return np.zeros_like(v)
    return [scale(v) for v in [train_vec] + list(other_vecs)]


def get_logits(trainer, dataset, device):
    trainer.model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = {k: v.unsqueeze(0).to(device)
                     for k, v in dataset[i].items() if k != "labels"}
            outputs = trainer.model(**batch)
            all_logits.append(outputs.logits.cpu().numpy()[0])
    return np.vstack(all_logits)


def get_cls_embeddings(trainer, dataset, device):
    """
    Get normalized CLS-like embeddings from the fine-tuned SciBERT.
    We'll try model.bert (BERT-style base) and fall back to model.base_model.
    """
    model = trainer.model
    base_model = getattr(model, "bert", getattr(model, "base_model", model))
    base_model.eval()

    embs = []
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = {k: v.unsqueeze(0).to(device)
                     for k, v in dataset[i].items() if k != "labels"}
            out = base_model(**batch)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                cls_vec = out.pooler_output[0].cpu().numpy()
            else:
                last_hidden = out.last_hidden_state[0].cpu().numpy()
                cls_vec = last_hidden[0]
            embs.append(cls_vec)

    embs = np.vstack(embs)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def tfidf_similarity(train_texts, eval_texts, domain_terms):
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_train = tfidf.fit_transform(train_texts)
    tfidf_eval = tfidf.transform(eval_texts)

    vocab = tfidf.vocabulary_
    q_vec = np.zeros((len(vocab),), dtype=float)
    for term in domain_terms:
        if term in vocab:
            q_vec[vocab[term]] = 1.0
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm

    def cosine_rowwise(mat, q):
        row_norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
        dots = mat.dot(q)
        sims = np.zeros_like(dots, dtype=float)
        nz = row_norms > 0
        sims[nz] = dots[nz] / row_norms[nz]
        return sims

    sims_train = cosine_rowwise(tfidf_train, q_vec)
    sims_eval = cosine_rowwise(tfidf_eval, q_vec)
    return sims_train, sims_eval


def main():
    parser = argparse.ArgumentParser(
        description="Heavy hybrid (SciBERT fine-tune + priors, recall-oriented thresholding + workload metrics)."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--recall_target", type=float, default=0.95)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight on transformer score in fusion [0..1].")
    parser.add_argument(
        "--domain_terms",
        default="glymphatic,perivascular,pvs,csf,diffusion,asl,flair,spectroscopy",
        help="Comma-separated domain vocab for TF-IDF/centroid priors.",
    )
    parser.add_argument("--calibrate_platt", action="store_true")
    parser.add_argument("--device", default=None,
                        help="'cuda' or 'cpu'. Default: auto.")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input_csv)
    texts_all = [
        normalize_text(t, a)
        for t, a in zip(df.get("title", ""), df.get("abstract", ""))
    ]
    y_all = df["label"].astype(int).to_numpy()
    ids_all = build_ids(df)

    # Deterministic split 70/10/20
    idx_all = np.arange(len(df))
    idx_trainval, idx_test, y_trainval, y_test = train_test_split(
        idx_all,
        y_all,
        test_size=0.2,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_trainval,
        y_trainval,
        test_size=0.125,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )

    split_manifest = pd.DataFrame(
        {
            "record_id": [ids_all[i] for i in idx_train]
            + [ids_all[i] for i in idx_val]
            + [ids_all[i] for i in idx_test],
            "split": (["train"] * len(idx_train))
            + (["val"] * len(idx_val))
            + (["test"] * len(idx_test)),
        }
    )
    split_manifest.to_csv(outdir / "split_manifest.csv", index=False)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL)
    enc_train = tokenizer(
        [texts_all[i] for i in idx_train],
        truncation=True,
        padding=True,
        max_length=args.max_len,
        return_tensors="pt",
    )
    enc_val = tokenizer(
        [texts_all[i] for i in idx_val],
        truncation=True,
        padding=True,
        max_length=args.max_len,
        return_tensors="pt",
    )
    enc_test = tokenizer(
        [texts_all[i] for i in idx_test],
        truncation=True,
        padding=True,
        max_length=args.max_len,
        return_tensors="pt",
    )

    ds_train = TextDataset(enc_train, y_train)
    ds_val = TextDataset(enc_val, y_val)
    ds_test = TextDataset(enc_test, y_test)

    # Init SciBERT classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        SCIBERT_MODEL,
        num_labels=2,
    )

    # class weights ~ inverse frequency
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    if pos_count > 0 and neg_count > 0:
        w_neg = 1.0 / neg_count
        w_pos = 1.0 / pos_count
        avg_w = (w_neg + w_pos) / 2.0
        class_weights = [w_neg / avg_w, w_pos / avg_w]
    else:
        class_weights = [1.0, 1.0]

    # device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir=str(outdir / "hf_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=RANDOM_STATE,
        report_to=[],
    )

    trainer = WeightedCETrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
    )

    trainer.train()

    # Get logits for train/val/test
    logits_train = get_logits(trainer, ds_train, device)
    logits_val = get_logits(trainer, ds_val, device)
    logits_test = get_logits(trainer, ds_test, device)

    probs_train_raw = softmax_prob(logits_train)
    probs_val_raw = softmax_prob(logits_val)
    probs_test_raw = softmax_prob(logits_test)

    if args.calibrate_platt:
        platt_clf = platt_scale_fit(logits_val, y_val)
        trans_train_score = platt_scale_apply(platt_clf, logits_train)
        trans_val_score = platt_scale_apply(platt_clf, logits_val)
        trans_test_score = platt_scale_apply(platt_clf, logits_test)
        calibration = "platt"
    else:
        trans_train_score = probs_train_raw
        trans_val_score = probs_val_raw
        trans_test_score = probs_test_raw
        calibration = "none"

    # CLS embeddings for centroid prior
    emb_train_unit = get_cls_embeddings(trainer, ds_train, device)
    emb_val_unit = get_cls_embeddings(trainer, ds_val, device)
    emb_test_unit = get_cls_embeddings(trainer, ds_test, device)

    pos_mask = (y_train == 1)
    if np.any(pos_mask):
        centroid_vec = emb_train_unit[pos_mask].mean(axis=0)
        norm_c = np.linalg.norm(centroid_vec)
        if norm_c > 0:
            centroid_vec = centroid_vec / norm_c
    else:
        centroid_vec = np.zeros(emb_train_unit.shape[1], dtype=float)

    def centroid_sim(Xu, c):
        return np.dot(Xu, c)

    cent_train = centroid_sim(emb_train_unit, centroid_vec)
    cent_val = centroid_sim(emb_val_unit, centroid_vec)
    cent_test = centroid_sim(emb_test_unit, centroid_vec)

    # TF-IDF prior (fit only on TRAIN text)
    domain_terms = [w.strip() for w in args.domain_terms.split(",") if w.strip()]
    tfidf_train_sim, tfidf_val_sim = tfidf_similarity(
        [texts_all[i] for i in idx_train],
        [texts_all[i] for i in idx_val],
        domain_terms,
    )
    _, tfidf_test_sim = tfidf_similarity(
        [texts_all[i] for i in idx_train],
        [texts_all[i] for i in idx_test],
        domain_terms,
    )

    # normalize components on TRAIN
    trans_tr_n, trans_val_n, trans_te_n = minmax_fit_apply(
        trans_train_score, [trans_val_score, trans_test_score]
    )
    cent_tr_n, cent_val_n, cent_te_n = minmax_fit_apply(
        cent_train, [cent_val, cent_test]
    )
    tfidf_tr_n, tfidf_val_n, tfidf_te_n = minmax_fit_apply(
        tfidf_train_sim, [tfidf_val_sim, tfidf_test_sim]
    )

    # fuse
    alpha = float(args.alpha)
    beta = (1.0 - alpha) / 2.0
    fuse_train = alpha * trans_tr_n + beta * cent_tr_n + beta * tfidf_tr_n
    fuse_val = alpha * trans_val_n + beta * cent_val_n + beta * tfidf_val_n
    fuse_test = alpha * trans_te_n + beta * cent_te_n + beta * tfidf_te_n

    # choose τ* on validation
    thr_star = select_threshold(y_val, fuse_val, recall_target=args.recall_target)

    # final eval on untouched test
    binary_metrics = compute_binary_metrics(y_test, fuse_test, thr_star)
    workload = compute_workload_metrics(
        y_test, fuse_test, recall_target=args.recall_target
    )

    scored_test = pd.DataFrame(
        {
            "record_id": [ids_all[i] for i in idx_test],
            "y_true": y_test,
            "score_fused": fuse_test,
            "pred_binary": (fuse_test >= thr_star).astype(int),
            "score_transformer_norm": trans_te_n,
            "score_centroid_norm": cent_te_n,
            "score_tfidf_norm": tfidf_te_n,
        }
    ).sort_values("score_fused", ascending=False)
    scored_test.to_csv(outdir / "test_scores_heavyhybrid.csv", index=False)

    summary_row = {
        **binary_metrics,
        **workload,
        "model": "HeavyHybrid_SciBERT+priors",
        "recall_target": args.recall_target,
        "random_state": RANDOM_STATE,
        "alpha_transformer_weight": alpha,
        "calibration": calibration,
    }
    pd.DataFrame([summary_row]).to_csv(
        outdir / "summary_test_heavyhybrid.csv", index=False
    )

    manifest = {
        "model": "HeavyHybrid_SciBERT+priors",
        "scibert_model": SCIBERT_MODEL,
        "random_state": RANDOM_STATE,
        "recall_target": args.recall_target,
        "threshold_star": thr_star,
        "alpha_transformer_weight": alpha,
        "calibration": calibration,
        "domain_terms": domain_terms,
        "split_counts": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "metrics_test": summary_row,
        "files": {
            "split_manifest": "split_manifest.csv",
            "per_item_scores": "test_scores_heavyhybrid.csv",
            "summary_csv": "summary_test_heavyhybrid.csv",
        },
    }
    with open(outdir / "manifest_heavyhybrid.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== Heavy hybrid: done ===")
    for k, v in summary_row.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
