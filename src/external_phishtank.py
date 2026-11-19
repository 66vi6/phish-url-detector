import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common import load_split
from src.data import normalize, reg_domain, template
from src.features import FEATURE_NAMES, extract
from src.metrics import threshold_at_fpr, wilson
from src.scoring import load_models, score

ML = ("n-gram LR", "LightGBM", "Blend")


def main():
    models = load_models()
    Xv, yv, uv = load_split("val")
    Xt, yt, ut = load_split("test")
    yv, yt = yv.to_numpy(), yt.to_numpy()
    Pv, Pt = score(uv.tolist(), Xv, models), score(ut.tolist(), Xt, models)

    ext = pd.read_csv("data/raw/phishtank_external.csv")
    log = [("PhishTank rows", len(ext))]
    ext["url"] = ext["url"].map(normalize)
    ext = ext[ext["url"] != ""].drop_duplicates("url")
    log.append(("after exact dedupe", len(ext)))
    ext = ext.drop_duplicates("tmpl" if "tmpl" in ext else ext["url"].map(template).name) \
        if False else ext.assign(tmpl=ext["url"].map(template)).drop_duplicates("tmpl")
    log.append(("after template dedupe", len(ext)))
    ext["reg_domain"] = ext["url"].map(reg_domain)

    seen = pd.concat([pd.read_csv(f"data/processed/{s}.csv", usecols=["url", "reg_domain"])
                      for s in ("train", "val")])
    ext = ext[~ext["url"].isin(set(seen["url"]))]
    log.append(("after removing URLs seen in train/val", len(ext)))
    ext = ext[~ext["reg_domain"].isin(set(seen["reg_domain"]))]
    log.append(("after removing domains seen in train/val", len(ext)))
    n = len(ext)
    if n < 100:
        print(f"WARNING: only {n} usable PhishTank URLs; intervals will be wide. Download more snapshots.")
    if n == 0:
        raise SystemExit("No usable PhishTank URLs left after leakage removal.")

    Xe = pd.DataFrame([extract(u) for u in ext["url"]], columns=FEATURE_NAMES)
    Pe = score(ext["url"].tolist(), Xe, models)

    tr_dates = pd.to_datetime(pd.concat([pd.read_csv(f"data/processed/{s}.csv") for s in ("train", "val", "test")])
                              .query("label == 1")["collected_at"], utc=True, format="ISO8601")
    ext_dates = pd.to_datetime(ext["collected_at"], utc=True, format="ISO8601")

    L = ["# External check: held-out PhishTank phishing", "",
         "| Step | Rows |", "|---|---|"] + [f"| {a} | {b:,} |" for a, b in log]
    L += ["", f"PhreshPhish phishing dates: {tr_dates.min():%Y-%m-%d} to {tr_dates.max():%Y-%m-%d}. "
              f"PhishTank dates: {ext_dates.min():%Y-%m-%d} to {ext_dates.max():%Y-%m-%d}. "
              "This test mixes a new feed with a later time period, and the two effects can't be separated.", "",
          "Thresholds are the validation thresholds at 1% FPR. Negatives for AUC are the test-split benign URLs.", "",
          "| Model | Recall on PhishTank (95% CI) | Recall on PhreshPhish test phish | Test FPR | ROC-AUC (PhishTank vs test benign) |",
          "|---|---|---|---|---|"]
    for m in ML:
        thr = threshold_at_fpr(yv, Pv[m], 0.01)
        k = int((Pe[m] >= thr).sum())
        lo, hi = wilson(k, n)
        internal = (Pt[m][yt == 1] >= thr).mean()
        fpr = (Pt[m][yt == 0] >= thr).mean()
        auc = roc_auc_score(np.r_[np.ones(n), np.zeros((yt == 0).sum())],
                            np.r_[Pe[m], Pt[m][yt == 0]])
        L.append(f"| {m} | {k / n:.3f} ({lo:.3f}-{hi:.3f}) | {internal:.3f} | {fpr:.2%} | {auc:.4f} |")
    open("reports/external_phishtank.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()