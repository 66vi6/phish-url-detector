import glob
import pandas as pd


def build_raw() -> pd.DataFrame:
    files = sorted(glob.glob("data/raw/phishtank/phishtank_*.csv.gz"))
    if not files:
        raise SystemExit("No PhishTank snapshots found in data/raw/phishtank/")

    pt = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pt = pt.drop_duplicates("phish_id")                     # same phish appears in many snapshots
    pt = pt[["url", "submission_time"]].rename(columns={"submission_time": "collected_at"})
    pt["label"] = 1
    pt["source"] = "phishtank"
    return pt[["url", "label", "source", "collected_at"]]


if __name__ == "__main__":
    raw = build_raw()
    raw.to_csv("data/raw/raw.csv", index=False)
    print(f"Wrote data/raw/raw.csv: {len(raw):,} rows")
    print(f"collected_at range: {raw.collected_at.min()}  to  {raw.collected_at.max()}")