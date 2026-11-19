import pandas as pd
import json
import pathlib

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

RESULTS_JSON = pathlib.Path("reports/val_results.json")
RESULTS_MD = pathlib.Path("reports/val_results.md")


def recall_at_fpr(y, p, target_fpr: float) -> float:
    """Highest recall achievable while keeping false-positive rate <= target."""
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def precision_at_prevalence(y, p, target_fpr: float, prevalence: float = 0.01) -> float:
    """Precision you'd see if phishing were `prevalence` of real traffic, at the operating
    point with FPR <= target. Your val set is far more balanced than real traffic."""
    fpr, tpr, _ = roc_curve(y, p)
    i = np.flatnonzero(fpr <= target_fpr)[-1]
    t, f = tpr[i], fpr[i]
    denom = t * prevalence + f * (1 - prevalence)
    return float(t * prevalence / denom) if denom > 0 else float("nan")


def summarize(y, p) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "recall_fpr_0.1pct": recall_at_fpr(y, p, 0.001),
        "recall_fpr_1pct": recall_at_fpr(y, p, 0.01),
        "precision_at_1pct_prevalence": precision_at_prevalence(y, p, 0.001, 0.01),
    }


def record(name: str, y, p, note: str = "") -> dict:
    """Compute validation metrics and add/replace this model's row in reports/val_results.*"""
    RESULTS_JSON.parent.mkdir(exist_ok=True)
    results = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}
    results[name] = {**summarize(y, p), "note": note}
    RESULTS_JSON.write_text(json.dumps(results, indent=2))

    lines = ["# Validation results", "",
             "| Model | PR-AUC | ROC-AUC | Recall @0.1% FPR | Recall @1% FPR | "
             "Precision @1% prevalence (0.1% FPR) | Notes |",
             "|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        lines.append(f"| {k} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['recall_fpr_0.1pct']:.3f} | "
                     f"{r['recall_fpr_1pct']:.3f} | {r['precision_at_1pct_prevalence']:.3f} | {r['note']} |")
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"{name}: PR-AUC {results[name]['pr_auc']:.4f} | "
          f"recall@1%FPR {results[name]['recall_fpr_1pct']:.3f}")
    return results[name]

def threshold_at_fpr(y, p, target_fpr: float) -> float:
    """Score threshold whose false-positive rate on (y, p) is <= target. Flag when score >= threshold."""
    fpr, tpr, thr = roc_curve(y, p)
    i = np.flatnonzero(fpr <= target_fpr)[-1]
    return float(thr[i])


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson confidence interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def cluster_bootstrap(y, scores: dict, groups, n_boot: int = 1000, seed: int = 0,
                      fpr_target: float = 0.01) -> dict:
    """Bootstrap that resamples whole domains (clusters), not single URLs.
    URLs from one domain are correlated, so row-level resampling gives CIs that are too narrow.
    Returns {model: {"ap": array, "rec": array}}; arrays are aligned across models."""
    y = np.asarray(y)
    codes, _ = pd.factorize(pd.Series(groups))
    assert (codes >= 0).all(), "groups contains missing values"
    order = np.argsort(codes, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(codes))[:-1])
    G = len(members)
    rng = np.random.default_rng(seed)
    out = {n: {"ap": [], "rec": []} for n in scores}
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, G, G)])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for n, p in scores.items():
            pb = np.asarray(p)[idx]
            out[n]["ap"].append(average_precision_score(yb, pb))
            out[n]["rec"].append(recall_at_fpr(yb, pb, fpr_target))
    return {n: {k: np.array(v) for k, v in d.items()} for n, d in out.items()}