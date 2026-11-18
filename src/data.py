import re, pathlib
import pandas as pd
import tldextract

_extract = tldextract.TLDExtract(suffix_list_urls=())   # offline, no network needed

def normalize(u: str) -> str:
    u = str(u).strip()
    u = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", u)
    u = u.split("#")[0]
    m = re.match(r"([^/?]*)(.*)", u)
    host, rest = m.group(1).lower(), m.group(2)
    if rest == "/":
        rest = ""
    return host + rest

def template(u: str) -> str:
    u = re.sub(r"[a-f0-9]{16,}", "H", u)
    u = re.sub(r"\d+", "0", u)
    return u

def reg_domain(u: str) -> str:
    return _extract(u).registered_domain or u.split("/")[0].split("?")[0]

def clean(df: pd.DataFrame):
    log = []
    def step(name, before, after):
        log.append((name, before, after, before - after))

    df = df.copy()
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True, errors="coerce")
    df = df.sort_values("collected_at")

    n = len(df); log.append(("Raw rows", n, n, 0))

    # 1. bad rows
    n = len(df)
    df = df.dropna(subset=["url", "label"])
    df = df[df["url"].astype(str).str.len().between(4, 2048)]
    df = df[~df["url"].astype(str).str.contains(r"\s")]
    step("Drop empty/invalid/over-long", n, len(df))

    # 2. normalise
    df["url"] = df["url"].map(normalize)
    df = df[df["url"] != ""]

    # 4 (before dedupe). label conflicts
    n = len(df)
    conflict = df.groupby("url")["label"].nunique()
    df = df[~df["url"].isin(conflict[conflict > 1].index)]
    step("Drop URLs with conflicting labels", n, len(df))

    # 3. exact dedupe (keeps earliest because sorted by collected_at)
    n = len(df)
    df = df.drop_duplicates("url")
    step("Exact duplicates", n, len(df))

    # 5. template dedupe
    n = len(df)
    df["tmpl"] = df["url"].map(template)
    df = df.drop_duplicates(["tmpl", "label"])
    step("Near-duplicate templates", n, len(df))

    # 6. registered domain
    df["reg_domain"] = df["url"].map(reg_domain)

    return df, log

def write_log(log, path="reports/cleaning_log.md"):
    pathlib.Path(path).parent.mkdir(exist_ok=True)
    lines = ["| Step | Before | After | Removed |", "|---|---|---|---|"]
    lines += [f"| {a} | {b:,} | {c:,} | {d:,} |" for a, b, c, d in log]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    raw = pd.read_csv("data/raw/raw.csv")
    df, log = clean(raw)
    df.to_csv("data/processed/clean.csv", index=False)
    write_log(log)
    print(pd.read_csv("data/processed/clean.csv").groupby(["source", "label"]).size())