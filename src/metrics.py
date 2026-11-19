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