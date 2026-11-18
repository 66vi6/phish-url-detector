import glob, pandas as pd

files = sorted(glob.glob("data/raw/phishtank/phishtank_*.csv.gz"))
frames = []
for f in files:
    d = pd.read_csv(f)
    d["snapshot_file"] = f
    frames.append(d)

pt = pd.concat(frames, ignore_index=True)
pt = pt.drop_duplicates("phish_id")                     # same phish appears in many snapshots
pt = pt[["phish_id", "url", "submission_time", "target"]]
pt["label"] = 1
pt["source"] = "phishtank"
pt["collected_at"] = pd.to_datetime(pt["submission_time"], utc=True)   # use for the time split
pt.to_csv("data/raw/phishtank_merged.csv", index=False)
print(len(pt), "unique phishes;", pt.collected_at.min(), "to", pt.collected_at.max())

print(pt.url.str.len().describe())
print(pt.url.str.startswith("https").mean())     # share of https vs http
print(pt.target.value_counts().head(10))          # which brands are most targeted
print(pt.url.sample(5, random_state=1).tolist())  # eyeball a few as text only