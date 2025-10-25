#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

from metrics import (
    select_threshold,
    compute_binary_metrics,
    compute_workload_metrics,
)

RANDOM_STATE = 42


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


def embed_texts(model, texts, batch_size=32):
    embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        vec = model.encode(chunk, show_progress_bar=False, normalize_embeddings=True)
        embs.append(vec)
    return np.vstack(embs)


def main():
    parser = argparse.ArgumentParser(
        description="MiniLM-only baseline: LR classifier + recall-oriented thresholding + workload metrics."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--recall_target", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iter", type=int, default=5000)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    texts = [
        normalize_text(t, a)
        for t, a in zip(df.get("title", ""), df.get("abstract", ""))
    ]
    y_all = df["label"].astype(int).to_numpy()
    ids_all = build_ids(df)

    # 70/10/20 split with fixed seed, stratified
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
        test_size=0.125,  # 0.125 of 0.8 = 0.1 overall
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

    # embeddings
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X_all = embed_texts(st_model, texts, batch_size=args.batch_size)
    X_train = X_all[idx_train]
    X_val = X_all[idx_val]
    X_test = X_all[idx_test]

    # logistic regression
    lr = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=args.max_iter,
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train, y_train)

    p_train = lr.predict_proba(X_train)[:, 1]
    p_val = lr.predict_proba(X_val)[:, 1]
    p_test = lr.predict_proba(X_test)[:, 1]

    # choose τ* on validation (maximize precision s.t. Recall ≥ recall_target)
    thr_star = select_threshold(y_val, p_val, recall_target=args.recall_target)

    # final eval on held-out test
    binary_metrics = compute_binary_metrics(y_test, p_test, thr_star)
    workload = compute_workload_metrics(
        y_test, p_test, recall_target=args.recall_target
    )

    scored_test = pd.DataFrame(
        {
            "record_id": [ids_all[i] for i in idx_test],
            "y_true": y_test,
            "score": p_test,
            "pred_binary": (p_test >= thr_star).astype(int),
        }
    ).sort_values("score", ascending=False)
    scored_test.to_csv(outdir / "test_scores_minilm.csv", index=False)

    summary_row = {
        **binary_metrics,
        **workload,
        "model": "MiniLM_LR",
        "recall_target": args.recall_target,
        "random_state": RANDOM_STATE,
    }
    pd.DataFrame([summary_row]).to_csv(
        outdir / "summary_test_minilm.csv", index=False
    )

    manifest = {
        "model": "MiniLM_LR",
        "sentence_transformer": "sentence-transformers/all-MiniLM-L6-v2",
        "random_state": RANDOM_STATE,
        "recall_target": args.recall_target,
        "threshold_star": thr_star,
        "split_counts": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "metrics_test": summary_row,
        "files": {
            "split_manifest": "split_manifest.csv",
            "per_item_scores": "test_scores_minilm.csv",
            "summary_csv": "summary_test_minilm.csv",
        },
    }
    with open(outdir / "manifest_minilm.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== MiniLM baseline: done ===")
    for k, v in summary_row.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
