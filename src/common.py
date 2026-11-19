import joblib


def load_split(split: str):
    """Returns (X features DataFrame, y labels Series, url Series) for train/val/test."""
    d = joblib.load(f"data/processed/features_{split}.joblib")
    return d["X"], d["y"], d["url"]


def require_both_classes(y, name: str):
    if y.nunique() < 2:
        raise SystemExit(
            f"The {name} split has only one class ({y.unique().tolist()}). "
            "Add benign URLs and rebuild: collect -> data -> split -> build_features."
        )