import datetime as dt
import hashlib
import json
import pathlib
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.common import load_split
from src.metrics import cluster_bootstrap, recall_at_fpr, threshold_at_fpr
from src.scoring import MODELS, load_models, score

N_BOOT = 1000                 # lower to ~200 while debugging
PREVALENCES = (0.001, 0.01, 0.05)
ML_MODELS = ("n-gram LR", "LightGBM", "Blend")
FROZEN_TAG = "v0.1-frozen"
FROZEN_FILES = ["models/ngram_lr.joblib", "models/gbm.joblib", "models/blend.joblib"]


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_check() -> bool:
    """First run records model hashes. Later runs report whether the models are unchanged."""
    now = {f: sha256(f) for f in FROZEN_FILES}
    rec = pathlib.Path("reports/frozen_models.json")
    if rec.exists():
        same = json.loads(rec.read_text())["hashes"] == now
        if not same:
            print("WARNING: models differ from those first evaluated on the test set.")
        return same
    rec.parent.mkdir(exist_ok=True)
    rec.write_text(json.dumps({
        "first_test_evaluation_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "hashes": now}, indent=2))
    return True


def precision_at(rec, fpr, prev):
    d = rec * prev + fpr * (1 - prev)
    return rec * prev / d if d > 0 else float("nan")


def plots(y, P):
    fig, ax = plt.subplots(figsize=(6, 5))
    for n in MODELS:
        pr, rc, _ = precision_recall_curve(y, P[n])
        ax.plot(rc, pr, label=n)
    ax.axhline(y.mean(), ls=":", c="gray", label="no skill")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)", ylim=(0, 1.02))
    ax.legend()
    fig.savefig("reports/pr_curve_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for n in MODELS:
        fpr, tpr, _ = roc_curve(y, P[n])
        ax.plot(np.maximum(fpr, 1e-4), tpr, label=n)
    ax.set(xscale="log", xlabel="False-positive rate (log scale)", ylabel="Recall",
           title="ROC (test)", ylim=(0, 1.02))
    ax.axvline(0.01, ls=":", c="gray")
    ax.legend()
    fig.savefig("reports/roc_curve_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    if FROZEN_TAG not in git("tag", "--list", FROZEN_TAG).split():
        raise SystemExit(f"Freeze first: git tag -a {FROZEN_TAG} -m 'Models frozen before test evaluation'")
    unchanged = frozen_check()

    X, y, urls = load_split("test")
    meta = pd.read_csv("data/processed/test.csv")
    assert len(meta) == len(y) and (meta["url"].to_numpy() == urls.to_numpy()).all(), \
        "test.csv and features_test.joblib are out of sync; rerun build_features"
    Xv, yv, uv = load_split("val")
    y, yv = y.to_numpy(), yv.to_numpy()

    models = load_models()
    P = score(urls.tolist(), X, models)
    Pv = score(uv.tolist(), Xv, models)

    print(f"Bootstrapping ({N_BOOT} resamples of whole domains)... this takes a few minutes")
    boot = cluster_bootstrap(y, P, meta["reg_domain"], n_boot=N_BOOT)
    diff = boot["Blend"]["ap"] - boot["n-gram LR"]["ap"]
    dlo, dhi = np.percentile(diff, [2.5, 97.5])

    L = ["# Test-set results", ""]
    if not unchanged:
        L += ["> **WARNING:** the model files differ from those first evaluated on the test set, "
              "so these numbers are no longer a clean estimate.", ""]
    L += [f"Test set: {len(y):,} URLs ({int(y.sum()):,} phishing, {int((y == 0).sum()):,} benign; "
          f"{y.mean():.1%} phishing, from {meta['reg_domain'].nunique():,} domains). "
          "Models were trained on train only; thresholds were chosen on validation.", "",
          f"## Headline (95% CIs from resampling whole domains, {len(diff)} resamples)", "",
          "| Model | Val PR-AUC | Test PR-AUC (95% CI) | Test ROC-AUC | Recall @1% FPR (95% CI) | Recall @0.1% FPR |",
          "|---|---|---|---|---|---|"]
    for n in MODELS:
        ap_lo, ap_hi = np.percentile(boot[n]["ap"], [2.5, 97.5])
        r_lo, r_hi = np.percentile(boot[n]["rec"], [2.5, 97.5])
        L.append(f"| {n} | {average_precision_score(yv, Pv[n]):.4f} | "
                 f"{average_precision_score(y, P[n]):.4f} ({ap_lo:.4f}-{ap_hi:.4f}) | "
                 f"{roc_auc_score(y, P[n]):.4f} | {recall_at_fpr(y, P[n], 0.01):.3f} ({r_lo:.3f}-{r_hi:.3f}) | "
                 f"{recall_at_fpr(y, P[n], 0.001):.3f} |")
    verdict = "the CI excludes 0, so the gain is real" if dlo > 0 or dhi < 0 else \
        "the CI includes 0, so the gain is not distinguishable from noise"
    L += ["", f"Blend minus n-gram LR (paired, PR-AUC): {diff.mean():+.4f} "
              f"(95% CI {dlo:+.4f} to {dhi:+.4f}); {verdict}.", "",
          "Recall at a fixed FPR above uses the test set's own ROC curve, which is slightly optimistic. "
          "The next table is the realistic version.", "",
          "## Operating points: threshold chosen on validation, applied to test", "",
          "| Model | Target FPR | Test FPR | Test recall | Precision @0.1% prev | @1% prev | @5% prev |",
          "|---|---|---|---|---|---|---|"]
    for n in ML_MODELS:
        for target in (0.01, 0.001):
            thr = threshold_at_fpr(yv, Pv[n], target)
            flagged = P[n] >= thr
            rec, fpr = flagged[y == 1].mean(), flagged[y == 0].mean()
            ps = " | ".join(f"{precision_at(rec, fpr, pv):.3f}" for pv in PREVALENCES)
            L.append(f"| {n} | {target:.1%} | {fpr:.2%} | {rec:.3f} | {ps} |")
    L += ["", "Precision columns recompute precision for a realistic share of phishing in traffic "
              "(0.1%, 1%, 5%). The test set itself is much more phishing-heavy than real traffic, "
              "which makes its raw precision misleading.", ""]
    pathlib.Path("reports/test_results.md").write_text("\n".join(L))

    out = meta[["url", "label", "reg_domain"]].copy()
    for n in MODELS:
        out[n] = P[n]
    out.to_csv("data/processed/test_predictions.csv", index=False)   # gitignored; used in Phase 8
    plots(y, P)
    print("\n".join(L))


if __name__ == "__main__":
    main()