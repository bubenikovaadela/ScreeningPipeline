#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


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


def minmax_fit_apply_single(vec):
    lo = np.min(vec)
    hi = np.max(vec)
    if hi > lo:
        return (vec - lo) / (hi - lo)
    else:
        return np.zeros_like(vec)


def main():
    parser = argparse.ArgumentParser(
        description="Label-free pipeline: TF-IDF prior + MiniLM semantic similarity to domain terms; outputs ranked list."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--domain_terms",
        default="glymphatic,perivascular,pvs,csf,diffusion,asl,flair,spectroscopy",
        help="Comma-separated domain vocab (query) for TF-IDF and semantic similarity.",
    )
    parser.add_argument(
        "--weights",
        default="0.5,0.5",
        help="Two non-negative weights for [semantic_sim, tfidf_sim]. Will be L1-normalized.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Optional: also output a binary suggestion of top_k to screen first.",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    texts_all = [
        normalize_text(t, a)
        for t, a in zip(df.get("title", ""), df.get("abstract", ""))
    ]
    ids_all = build_ids(df)

    domain_terms = [w.strip() for w in args.domain_terms.split(",") if w.strip()]
    query_text = " ".join(domain_terms)

    # --- TF-IDF prior
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_mat = tfidf.fit_transform(texts_all)

    vocab = tfidf.vocabulary_
    q_vec = np.zeros((len(vocab),), dtype=float)
    for term in domain_terms:
        if term in vocab:
            q_vec[vocab[term]] = 1.0
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm

    row_norms = np.sqrt(tfidf_mat.multiply(tfidf_mat).sum(axis=1)).A1
    dots = tfidf_mat.dot(q_vec)
    tfidf_sim = np.zeros_like(dots, dtype=float)
    nz = row_norms > 0
    tfidf_sim[nz] = dots[nz] / row_norms[nz]

    # --- semantic similarity prior via MiniLM
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb_all = st_model.encode(
        texts_all, show_progress_bar=False, normalize_embeddings=True
    )  # N x D, unit norm
    q_emb = st_model.encode(
        [query_text], show_progress_bar=False, normalize_embeddings=True
    )[0]  # D, unit norm
    sem_sim = np.dot(emb_all, q_emb)  # cosine similarity

    # normalize both components [0,1]
    sem_sim_n = minmax_fit_apply_single(sem_sim)
    tfidf_sim_n = minmax_fit_apply_single(tfidf_sim)

    # fuse
    w_parts = np.array([float(x) for x in args.weights.split(",")], dtype=float)
    if np.any(w_parts < 0):
        raise ValueError("All weights must be non-negative.")
    if w_parts.sum() == 0:
        raise ValueError("Weights must not all be zero.")
    w_parts = w_parts / w_parts.sum()
    fused = w_parts[0] * sem_sim_n + w_parts[1] * tfidf_sim_n

    # rank
    order = np.argsort(fused)[::-1]
    ranked = pd.DataFrame(
        {
            "record_id": [ids_all[i] for i in order],
            "rank": np.arange(1, len(order) + 1),
            "fused_score": fused[order],
            "semantic_component": sem_sim_n[order],
            "tfidf_component": tfidf_sim_n[order],
            "title": [str(df.get("title", "")[i]) for i in order],
            "abstract": [str(df.get("abstract", "")[i]) for i in order],
        }
    )
    ranked.to_csv(outdir / "ranked_unsupervised.csv", index=False)

    # optional binary suggestion: "screen these first"
    top_k = int(args.top_k)
    k_eff = min(top_k, len(order))
    suggest = ranked.iloc[:k_eff].copy()
    suggest["screen_flag"] = 1
    suggest.to_csv(outdir / "topk_binary.csv", index=False)

    manifest = {
        "mode": "unsupervised_fusion",
        "domain_terms": domain_terms,
        "weights": w_parts.tolist(),
        "top_k": top_k,
        "files": {
            "ranked": "ranked_unsupervised.csv",
            "topk_binary": "topk_binary.csv",
        },
    }
    with open(outdir / "manifest_unsupervised.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== Unsupervised fusion: done ===")
    print(f"Top {k_eff} suggested for manual screening.")


if __name__ == "__main__":
    main()
