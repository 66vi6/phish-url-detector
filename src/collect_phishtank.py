import datetime as dt, pathlib, glob
import httpx

UA = "phishtank/raghad-deakin-project"
URL = "http://data.phishtank.com/data/online-valid.csv.gz"
OUT = pathlib.Path("data/raw/phishtank")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)

    existing = sorted(glob.glob(str(OUT / "phishtank_*.csv.gz")))
    if existing:
        last = dt.datetime.strptime(pathlib.Path(existing[-1]).stem.split("_", 1)[1].split(".")[0],
                                    "%Y%m%d_%H%M").replace(tzinfo=dt.timezone.utc)
        if now - last < dt.timedelta(hours=20):
            raise SystemExit(f"Last download was {now - last} ago; skipping to respect rate limits.")

    dest = OUT / f"phishtank_{now.strftime('%Y%m%d_%H%M')}.csv.gz"
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=120) as c:
        r = c.get(URL)
    if r.status_code != 200 or r.content[:2] != b"\x1f\x8b":
        raise SystemExit(f"Download failed: HTTP {r.status_code} (not gzip). Try again tomorrow.")
    dest.write_bytes(r.content)
    print(f"Saved {dest} ({len(r.content)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()