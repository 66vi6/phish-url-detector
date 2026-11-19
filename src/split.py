import pathlib
import numpy as np
import pandas as pd

SEED = 42
FRACS = (0.70, 0.15, 0.15)           # train, val, test
ORDER = {"train": 0, "val": 1, "test": 2}


def _assign_benign_domains(benign: pd.DataFrame, rng) -> dict:
    """Randomly assign whole domains to splits, balancing by ROW count (not domain count),
    so one huge site can't skew the proportions."""
    sizes = benign.groupby("reg_domain").size()
    doms = sizes.index.to_numpy().copy()
    rng.shuffle(doms)
    cum = sizes.loc[doms].cumsum().to_numpy() / sizes.sum()
    split = np.where(cum <= FRACS[0], "train",
             np.where(cum <= FRACS[0] + FRACS[1], "val", "test"))
    return dict(zip(doms, split))


def make_splits(df: pd.DataFrame, seed: int = SEED):
    """Returns (split_df, dropped_df). split_df has a 'split' column."""
    df = df.copy()
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True, errors="coerce")

    # Phishing rows need a timestamp for the time split
    df = df[~((df["label"] == 1) & df["collected_at"].isna())]

    rng = np.random.default_rng(seed)
    df["split"] = ""

    # 1. Phishing: split by time
    phish = df[df["label"] == 1]
    if len(phish) >= 3:
        ts = phish["collected_at"].sort_values().reset_index(drop=True)
        n = len(ts)
        t_train = ts.iloc[max(int(n * FRACS[0]) - 1, 0)]
        t_val = ts.iloc[max(int(n * (FRACS[0] + FRACS[1])) - 1, 0)]
        df.loc[phish.index, "split"] = np.where(
            phish["collected_at"] <= t_train, "train",
            np.where(phish["collected_at"] <= t_val, "val", "test"))

    # 2. Benign: split by domain
    benign = df[df["label"] == 0]
    if len(benign):
        mapping = _assign_benign_domains(benign, rng)
        df.loc[benign.index, "split"] = benign["reg_domain"].map(mapping)

    # 3. A domain may exist in only ONE split: the earliest split it appears in wins
    df["_rank"] = df["split"].map(ORDER)
    earliest = df.groupby("reg_domain")["_rank"].transform("min")
    keep = df["_rank"] == earliest
    dropped = df[~keep].drop(columns="_rank")
    out = df[keep].drop(columns="_rank")
    return out, dropped


def summarize(df: pd.DataFrame) -> str:
    lines = ["| Split | Rows | Phishing | Benign | Phishing share | Phish date range |",
             "|---|---|---|---|---|---|"]
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        p = d[d["label"] == 1]
        rng_ = (f"{p['collected_at'].min():%Y-%m-%d %H:%M} to {p['collected_at'].max():%Y-%m-%d %H:%M}"
                if len(p) else "n/a")
        share = f"{len(p) / len(d):.1%}" if len(d) else "n/a"
        lines.append(f"| {s} | {len(d):,} | {len(p):,} | {len(d) - len(p):,} | {share} | {rng_} |")
    return "\n".join(lines)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/clean.csv")
    if (df["label"] == 0).sum() == 0:
        print("WARNING: no benign rows found. This output is for testing the code only.\n")

    out, dropped = make_splits(df)

    for s in ("train", "val", "test"):
        out[out["split"] == s].to_csv(f"data/processed/{s}.csv", index=False)

    report = ["# Split summary", "", summarize(out), "",
              f"Rows dropped to keep domains in one split only: {len(dropped):,} "
              f"({len(dropped) / max(len(df), 1):.1%} of cleaned data)", ""]
    if len(dropped):
        report += ["Dropped by label:", "",
                   dropped.groupby("label").size().to_string(), "",
                   "Top domains causing drops:", "",
                   dropped["reg_domain"].value_counts().head(10).to_string()]
    pathlib.Path("reports").mkdir(exist_ok=True)
    pathlib.Path("reports/split_summary.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))