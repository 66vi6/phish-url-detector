import pandas as pd
from src.split import make_splits


def _toy():
    rows = []
    t0 = pd.Timestamp("2026-10-01", tz="UTC")
    for i in range(200):                                   # phishing, one domain each
        rows.append(dict(url=f"bad{i}.com/x", label=1, source="phishtank",
                         collected_at=t0 + pd.Timedelta(hours=i), reg_domain=f"bad{i}.com"))
    for d in range(60):                                    # benign, 5 URLs per domain
        for j in range(5):
            rows.append(dict(url=f"good{d}.com/{j}", label=0, source="tranco",
                             collected_at=pd.NaT, reg_domain=f"good{d}.com"))
    # a LATE phish on a domain that also appears early (in train) -> must be dropped
    rows.append(dict(url="late.bad0.com/z", label=1, source="phishtank",
                     collected_at=t0 + pd.Timedelta(hours=500), reg_domain="bad0.com"))
    return pd.DataFrame(rows)


def test_all_splits_non_empty():
    out, _ = make_splits(_toy())
    assert set(out["split"]) == {"train", "val", "test"}


def test_no_domain_in_two_splits():
    out, _ = make_splits(_toy())
    assert out.groupby("reg_domain")["split"].nunique().max() == 1


def test_phishing_is_time_ordered():
    out, _ = make_splits(_toy())
    p = out[out["label"] == 1]
    tr = p[p.split == "train"].collected_at.max()
    va = p[p.split == "val"]
    te = p[p.split == "test"]
    assert tr <= va.collected_at.min()
    assert va.collected_at.max() <= te.collected_at.min()


def test_overlapping_domain_row_is_dropped():
    out, dropped = make_splits(_toy())
    assert "late.bad0.com/z" not in set(out["url"])
    assert "late.bad0.com/z" in set(dropped["url"])


def test_no_url_in_two_splits():
    out, _ = make_splits(_toy())
    assert out.groupby("url")["split"].nunique().max() == 1