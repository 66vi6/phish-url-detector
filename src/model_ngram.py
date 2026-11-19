import joblib
import pathlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.common import load_split, require_both_classes
from src.metrics import record

C_GRID = [0.5, 1, 4, 16]


def main():
    _, ytr, utr = load_split("train")
    _, yva, uva = load_split("val")
    require_both_classes(ytr, "train")
    require_both_classes(yva, "val")

    # Fit the vectoriser on TRAIN only; val is just transformed.
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=5,
                          max_features=200_000, sublinear_tf=True)
    Xtr = vec.fit_transform(utr)
    Xva = vec.transform(uva)
    print(f"n-gram matrix: {Xtr.shape[0]:,} x {Xtr.shape[1]:,}")

    best = None
    for C in C_GRID:
        lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        lr.fit(Xtr, ytr)
        p = lr.predict_proba(Xva)[:, 1]
        ap = average_precision_score(yva, p)
        print(f"  C={C:<5} val PR-AUC {ap:.4f}")
        if best is None or ap > best[0]:
            best = (ap, C, lr, p)

    _, C, lr, p = best
    pathlib.Path("models").mkdir(exist_ok=True)
    joblib.dump({"vec": vec, "lr": lr, "C": C}, "models/ngram_lr.joblib")
    joblib.dump(p, "models/val_preds_ngram.joblib")
    record("TF-IDF char n-grams + LR", yva, p, note=f"C={C}")


if __name__ == "__main__":
    main()