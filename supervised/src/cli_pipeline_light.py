#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

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


def cosine_sim_to_centroid(X_unit, centroid_unit):
    return np.dot(X_unit, centroid_unit)


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


def parse_weights(s):
    parts = [float(x.strip()) for x in s.split(",")]
    w = np.array(parts, dtype=float)
    if np.any(w < 0):
        raise ValueError("All weights must be non-negative.")
    if w.sum() == 0:
        raise ValueError("Weights must not all be zero.")
    return w / w.sum()


def tfidf_similarity(train_texts, eval_texts, domain_terms):
    """
    TF-IDF fit on train_texts only -> cosine sim to query built from domain_terms.
    """
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
        description="Light hybrid: LR prob + centroid sim + TF-IDF prior, fused; recall-oriented thresholding + workload metrics."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--recall_target", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument(
        "--domain_terms",
        default="glymphatic,perivascular,pvs,csf,diffusion,asl,flair,spectroscopy",
        help="Comma-separated domain vocab for TF-IDF prior.",
    )
    parser.add_argument(
        "--weights",
        default="0.5,0.3,0.2",
        help="Non-negative fusion weights for [LR_prob, centroid_sim, tfidf_sim]. Will be L1-normalized.",
    )
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

    # deterministic 70/10/20 split
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

    # embeddings (MiniLM)
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X_all = embed_texts(st_model, texts, batch_size=args.batch_size)
    X_train = X_all[idx_train]
    X_val = X_all[idx_val]
    X_test = X_all[idx_test]

    # --- LR prob component
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

    # --- centroid sim prior
    pos_mask = (y_train == 1)
    if np.any(pos_mask):
        centroid = X_train[pos_mask].mean(axis=0)
        norm_c = np.linalg.norm(centroid)
        if norm_c > 0:
            centroid = centroid / norm_c
    else:
        centroid = np.zeros(X_train.shape[1], dtype=float)

    sim_train = cosine_sim_to_centroid(X_train, centroid)
    sim_val = cosine_sim_to_centroid(X_val, centroid)
    sim_test = cosine_sim_to_centroid(X_test, centroid)

    # --- TF-IDF prior (fit only on TRAIN text!)
    domain_terms = [w.strip() for w in args.domain_terms.split(",") if w.strip()]
    tfidf_train_sim, tfidf_val_sim = tfidf_similarity(
        [texts[i] for i in idx_train],
        [texts[i] for i in idx_val],
        domain_terms,
    )
    _, tfidf_test_sim = tfidf_similarity(
        [texts[i] for i in idx_train],
        [texts[i] for i in idx_test],
        domain_terms,
    )

    # --- min-max normalize each component based on TRAIN
    lr_tr_n, lr_val_n, lr_te_n = minmax_fit_apply(p_train, [p_val, p_test])
    sim_tr_n, sim_val_n, sim_te_n = minmax_fit_apply(sim_train, [sim_val, sim_test])
    tfidf_tr_n, tfidf_val_n, tfidf_te_n = minmax_fit_apply(
        tfidf_train_sim, [tfidf_val_sim, tfidf_test_sim]
    )

    # fuse
    W = parse_weights(args.weights)  # e.g. [0.5,0.3,0.2] normalized to sum=1
    fuse_train = W[0] * lr_tr_n + W[1] * sim_tr_n + W[2] * tfidf_tr_n
    fuse_val = W[0] * lr_val_n + W[1] * sim_val_n + W[2] * tfidf_val_n
    fuse_test = W[0] * lr_te_n + W[1] * sim_te_n + W[2] * tfidf_te_n

    # choose τ* on validation
    thr_star = select_threshold(y_val, fuse_val, recall_target=args.recall_target)

    # evaluate on held-out test
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
            "lr_prob_component": lr_te_n,
            "centroid_sim_component": sim_te_n,
            "tfidf_sim_component": tfidf_te_n,
        }
    ).sort_values("score_fused", ascending=False)
    scored_test.to_csv(outdir / "test_scores_lighthybrid.csv", index=False)

    summary_row = {
        **binary_metrics,
        **workload,
        "model": "LightHybrid_LR+centroid+tfidf",
        "recall_target": args.recall_target,
        "random_state": RANDOM_STATE,
        "fusion_weights": W.tolist(),
    }
    pd.DataFrame([summary_row]).to_csv(
        outdir / "summary_test_lighthybrid.csv", index=False
    )

    manifest = {
        "model": "LightHybrid_LR+centroid+tfidf",
        "sentence_transformer": "sentence-transformers/all-MiniLM-L6-v2",
        "random_state": RANDOM_STATE,
        "recall_target": args.recall_target,
        "threshold_star": thr_star,
        "fusion_weights": W.tolist(),
        "domain_terms": domain_terms,
        "split_counts": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "metrics_test": summary_row,
        "files": {
            "split_manifest": "split_manifest.csv",
            "per_item_scores": "test_scores_lighthybrid.csv",
            "summary_csv": "summary_test_lighthybrid.csv",
        },
    }
    with open(outdir / "manifest_lighthybrid.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== Light hybrid: done ===")
    for k, v in summary_row.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
