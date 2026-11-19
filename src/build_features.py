import joblib
import pandas as pd

from src.features import extract, FEATURE_NAMES


def features_df(urls) -> pd.DataFrame:
    rows, failures = [], 0
    for u in urls:
        try:
            rows.append(extract(u))
        except Exception:
            failures += 1
            rows.append({k: float("nan") for k in FEATURE_NAMES})
    if failures:
        print(f"  {failures} URLs failed feature extraction (filled with NaN)")
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        df = pd.read_csv(f"data/processed/{split}.csv")
        X = features_df(df["url"])
        y = df["label"].reset_index(drop=True)
        joblib.dump({"X": X, "y": y, "url": df["url"].reset_index(drop=True)},
                    f"data/processed/features_{split}.joblib")
        print(f"{split}: {X.shape[0]:,} rows x {X.shape[1]} features")