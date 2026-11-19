import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score

from src.common import load_split
from src.features import FEATURE_NAMES
from src.metrics import recall_at_fpr

GROUPS = {
    "size": ["len_url", "len_host", "len_path", "len_query"],
    "structure": ["n_dots_host", "n_hyphens_host", "n_digits_host", "n_slashes",
                  "n_params", "sub_depth", "tld_len"],
    "host_tricks": ["is_ip", "has_userinfo", "has_port", "is_punycode"],
    "randomness": ["ent_host", "ent_path", "digit_ratio", "longest_token", "longest_digit_run"],
    "keywords": ["kw_total", "kw_in_host", "kw_in_path"],
    "brand": ["brand_in_sub", "brand_in_domain", "brand_in_path", "lookalike_dist", "is_lookalike"],
    "misc": ["n_pct", "embedded_http"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURE_NAMES), "groups must cover every feature exactly once"


def fit_eval(Xtr, ytr, Xva, yva, cols):
    gbm = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=31, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1)
    gbm.fit(Xtr[cols], ytr, eval_set=[(Xva[cols], yva)], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    p = gbm.predict_proba(Xva[cols])[:, 1]
    return average_precision_score(yva, p), recall_at_fpr(yva, p, 0.01)


def main():
    Xtr, ytr, _ = load_split("train")
    Xva, yva, _ = load_split("val")
    rows = []
    base_ap, base_rec = fit_eval(Xtr, ytr, Xva, yva, FEATURE_NAMES)
    rows.append(("All features", len(FEATURE_NAMES), base_ap, base_rec, 0.0))
    print(f"All features: PR-AUC {base_ap:.4f}")

    for g, cols in GROUPS.items():
        keep = [c for c in FEATURE_NAMES if c not in cols]
        ap, rec = fit_eval(Xtr, ytr, Xva, yva, keep)
        rows.append((f"Drop {g}", len(keep), ap, rec, ap - base_ap))
        print(f"Drop {g:<12} PR-AUC {ap:.4f} ({ap - base_ap:+.4f})")
    for g, cols in GROUPS.items():
        ap, rec = fit_eval(Xtr, ytr, Xva, yva, cols)
        rows.append((f"Only {g}", len(cols), ap, rec, ap - base_ap))
        print(f"Only {g:<12} PR-AUC {ap:.4f}")

    lines = ["# Ablation (LightGBM, validation set)", "",
             "| Setting | # features | Val PR-AUC | Recall @1% FPR | Change vs all |", "|---|---|---|---|---|"]
    lines += [f"| {n} | {k} | {ap:.4f} | {rec:.3f} | {d:+.4f} |" for n, k, ap, rec, d in rows]
    open("reports/ablation_val.md", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()