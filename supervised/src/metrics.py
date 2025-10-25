import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

RECALL_AT_K_SET = [50, 100, 200, 500, 1000]


def select_threshold(y_val, scores_val, recall_target=0.95):
    """
    Highest-precision threshold that still hits Recall >= recall_target.
    Ties in precision -> pick higher threshold.
    """
    y_val = np.asarray(y_val)
    scores_val = np.asarray(scores_val)

    thr_candidates = np.unique(scores_val)[::-1]
    best_thr = None
    best_prec = -1.0

    for thr in thr_candidates:
        y_pred = (scores_val >= thr).astype(int)
        tp = np.sum((y_pred == 1) & (y_val == 1))
        fp = np.sum((y_pred == 1) & (y_val == 0))
        fn = np.sum((y_pred == 0) & (y_val == 1))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if recall < recall_target:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if (precision > best_prec) or (
            np.isclose(precision, best_prec) and (best_thr is not None and thr > best_thr)
        ):
            best_prec = precision
            best_thr = thr

    if best_thr is None:
        best_thr = thr_candidates[-1]

    return float(best_thr)


def compute_binary_metrics(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc_roc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc_roc = np.nan
    try:
        ap = average_precision_score(y_true, y_score)
    except ValueError:
        ap = np.nan

    return {
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "accuracy": float(acc),
        "roc_auc": float(auc_roc),
        "average_precision": float(ap),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": float(thr),
    }


def compute_workload_metrics(y_true, y_score, recall_target=0.95, ks=RECALL_AT_K_SET):
    """
    Workload metrics:
    - k_R_for_recall_target: min #top docs to reach Recall >= target
    - WSS_at_recall_target = 1 - (k_R/N)
    - Recall@k for selected k values
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    N = len(y_true)
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]

    cum_rel = np.cumsum(y_sorted)
    total_rel = cum_rel[-1] if len(cum_rel) > 0 else 0

    if total_rel > 0:
        recall_curve = cum_rel / total_rel
        hits = np.where(recall_curve >= recall_target)[0]
        if len(hits) > 0:
            k_R = int(hits[0] + 1)
        else:
            k_R = N
    else:
        k_R = N

    wss_at = 1.0 - (k_R / N) if N > 0 else 0.0

    recall_at_k = {}
    for k in ks:
        k_eff = min(k, N)
        if k_eff == 0:
            recall_at_k[f"recall@{k}"] = 0.0
            continue
        recovered_rel = cum_rel[k_eff - 1] if total_rel > 0 else 0
        recall_at_k[f"recall@{k}"] = float(
            recovered_rel / total_rel if total_rel > 0 else 0.0
        )

    return {
        "k_R_for_recall_target": int(k_R),
        "WSS_at_recall_target": float(wss_at),
        "recall_target": float(recall_target),
        **recall_at_k,
    }
