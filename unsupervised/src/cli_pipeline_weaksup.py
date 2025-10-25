#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


def make_regex(patterns):
    """
    Turn a 'a|b|c' style string into a compiled case-insensitive regex.
    Empty -> None.
    """
    pat = patterns.strip()
    if not pat:
        return None
    return re.compile(pat, flags=re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser(
        description="Weak supervision: keyword seeds -> pseudo-labels -> LR on MiniLM -> ranked output."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--pos_keywords",
        required=True,
        help="Regex OR pattern for positive seeds, e.g. 'glymphatic|perivascular|pvs|csf circulation|phase-contrast mri'",
    )
    parser.add_argument(
        "--neg_keywords",
        default="",
        help="Regex OR pattern for obvious exclusions, e.g. 'cardiac|renal|liver'",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Top-k suggestions for manual screening.",
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

    rx_pos = make_regex(args.pos_keywords)
    rx_neg = make_regex(args.neg_keywords)

    pseudo_y = []
    for txt in texts_all:
        hit_pos = (rx_pos.search(txt) is not None) if rx_pos else False
        hit_neg = (rx_neg.search(txt) is not None) if rx_neg else False
        label = 1 if (hit_pos and not hit_neg) else 0
        pseudo_y.append(label)
    pseudo_y = np.array(pseudo_y, dtype=int)

    # Train LR on MiniLM embeddings using pseudo-labels
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X_all = st_model.encode(
        texts_all, show_progress_bar=False, normalize_embeddings=True
    )

    lr = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=42,
    )
    lr.fit(X_all, pseudo_y)
    p_all = lr.predict_proba(X_all)[:, 1]

    # Rank by pseudo-prob
    order = np.argsort(p_all)[::-1]
    ranked = pd.DataFrame(
        {
            "record_id": [ids_all[i] for i in order],
            "rank": np.arange(1, len(order) + 1),
            "pseudo_score": p_all[order],
            "pseudo_label": pseudo_y[order],
            "title": [str(df.get("title", "")[i]) for i in order],
            "abstract": [str(df.get("abstract", "")[i]) for i in order],
        }
    )
    ranked.to_csv(outdir / "ranked_weaksup.csv", index=False)

    # top-k screening suggestion
    top_k = int(args.top_k)
    k_eff = min(top_k, len(order))
    suggest = ranked.iloc[:k_eff].copy()
    suggest["screen_flag"] = 1
    suggest.to_csv(outdir / "topk_binary.csv", index=False)

    manifest = {
        "mode": "weak_supervision",
        "pos_keywords": args.pos_keywords,
        "neg_keywords": args.neg_keywords,
        "top_k": top_k,
        "files": {
            "ranked": "ranked_weaksup.csv",
            "topk_binary": "topk_binary.csv",
        },
    }
    with open(outdir / "manifest_weaksup.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== Weak supervision: done ===")
    print(f"Top {k_eff} suggested for manual screening.")


if __name__ == "__main__":
    main()
