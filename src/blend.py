import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from src.common import load_split
from src.metrics import record


def main():
    _, yva, _ = load_split("val")
    p_lr = joblib.load("models/val_preds_ngram.joblib")
    p_gb = joblib.load("models/val_preds_gbm.joblib")
    print("Score correlation between models:", round(float(np.corrcoef(p_lr, p_gb)[0, 1]), 3))

    best = (-1.0, 0.0)
    for w in np.linspace(0, 1, 11):                    # w = weight on LightGBM
        ap = average_precision_score(yva, w * p_gb + (1 - w) * p_lr)
        print(f"  w_gbm={w:.1f}  val PR-AUC {ap:.4f}")
        if ap > best[0]:
            best = (ap, w)

    w = best[1]
    p = w * p_gb + (1 - w) * p_lr
    joblib.dump({"w_gbm": float(w)}, "models/blend.joblib")
    record("Blend (n-gram LR + LightGBM)", yva, p, note=f"w_gbm={w:.1f}")


if __name__ == "__main__":
    main()