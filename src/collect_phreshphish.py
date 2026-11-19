import pathlib
import pandas as pd

REPO = "phreshphish/phreshphish"
OUT = pathlib.Path("data/raw/phreshphish_urls.csv")
WANT = ["url", "label", "date"]       # edit if the printed column names differ
MAX_PER_CLASS = 150_000               # keep the dataset a manageable size


def via_parquet():
    """Read only the needed columns from the Parquet files (skips the large HTML column)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{REPO}/**/*.parquet"))
    if not files:
        return None
    with fs.open(files[0]) as fh:
        print("Columns in the dataset:", pq.ParquetFile(fh).schema_arrow.names)
    frames = []
    for f in files:
        print("reading", f)
        frames.append(pd.read_parquet("hf://" + f, columns=WANT))
    return pd.concat(frames, ignore_index=True)


def via_streaming():
    """Slower fallback: streams every row (including HTML) but never stores it."""
    from datasets import load_dataset

    rows = []
    for split in ("train", "test"):
        ds = load_dataset(REPO, split=split, streaming=True)
        for i, r in enumerate(ds):
            if i == 0:
                print("Columns in the dataset:", list(r.keys()))
            rows.append({c: r.get(c) for c in WANT})
            if i % 50_000 == 0:
                print(split, i)
    return pd.DataFrame(rows)


def main():
    try:
        df = via_parquet()
    except Exception as e:
        print("Parquet route failed:", repr(e))
        df = None
    if df is None:
        print("Falling back to streaming (slower)...")
        df = via_streaming()

    labels = df["label"].astype(str).str.lower().map(
        {"phish": 1, "phishing": 1, "1": 1, "benign": 0, "0": 0})
    if labels.isna().all():
        raise SystemExit(f"Unrecognised label values: {df['label'].unique()[:10]}")
    df["label"] = labels
    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)

    # Cap each class with a uniform random sample (keeps the time distribution intact)
    parts = [g.sample(min(len(g), MAX_PER_CLASS), random_state=42) for _, g in df.groupby("label")]
    df = pd.concat(parts, ignore_index=True)

    ts = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["collected_at"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")      # one uniform format
    df["source"] = "phreshphish"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[["url", "label", "source", "collected_at"]].to_csv(OUT, index=False)

    print(f"\nWrote {OUT}: {len(df):,} rows")
    print(df["label"].value_counts().rename({0: "benign", 1: "phish"}).to_string())
    print("Phishing date range:", ts[df["label"] == 1].min(), "to", ts[df["label"] == 1].max())
    print("Missing dates (phish):", int(ts[df["label"] == 1].isna().sum()))


if __name__ == "__main__":
    main()