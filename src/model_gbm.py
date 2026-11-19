import joblib
import pathlib

import lightgbm as lgb
import pandas as pd

from src.common import load_split, require_both_classes
from src.metrics import record


def main():
    Xtr, ytr, _ = load_split("train")
    Xva, yva, _ = load_split("val")
    require_both_classes(ytr, "train")
    require_both_classes(yva, "val")

    gbm = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=31, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        importance_type="gain", random_state=42, n_jobs=-1, verbose=-1,
    )
    gbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    p = gbm.predict_proba(Xva)[:, 1]

    imp = pd.Series(gbm.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    print("Top features by gain:\n", (imp / imp.sum()).head(10).round(3).to_string())

    pathlib.Path("models").mkdir(exist_ok=True)
    joblib.dump(gbm, "models/gbm.joblib")
    joblib.dump(p, "models/val_preds_gbm.joblib")
    record("LightGBM (handcrafted features)", yva, p, note=f"{gbm.best_iteration_} trees")


if __name__ == "__main__":
    main()