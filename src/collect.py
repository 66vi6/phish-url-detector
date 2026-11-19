import glob
import pandas as pd


def build_raw() -> pd.DataFrame:
    """Main dataset: PhreshPhish only (both classes, one consistent collection process)."""
    raw = pd.read_csv("data/raw/phreshphish_urls.csv")
    return raw[["url", "label", "source", "collected_at"]]


def build_phishtank_external() -> pd.DataFrame:
    """PhishTank snapshots, held out for the cross-source test in Phase 7."""
    files = sorted(glob.glob("data/raw/phishtank/phishtank_*.csv.gz"))
    if not files:
        return pd.DataFrame(columns=["url", "label", "source", "collected_at"])
    pt = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates("phish_id")
    ts = pd.to_datetime(pt["submission_time"], utc=True, errors="coerce")
    pt["collected_at"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    pt["label"], pt["source"] = 1, "phishtank"
    return pt[["url", "label", "source", "collected_at"]]


if __name__ == "__main__":
    raw = build_raw()
    raw.to_csv("data/raw/raw.csv", index=False)
    print(f"Wrote data/raw/raw.csv: {len(raw):,} rows")
    print(raw.groupby(["source", "label"]).size().to_string())

    ext = build_phishtank_external()
    ext.to_csv("data/raw/phishtank_external.csv", index=False)
    print(f"Wrote data/raw/phishtank_external.csv: {len(ext):,} rows (held out for Phase 7)")