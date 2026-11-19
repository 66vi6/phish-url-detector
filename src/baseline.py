from src.common import load_split, require_both_classes
from src.metrics import record


def rule_score(X):
    """Count of red flags. Higher = more suspicious."""
    return (X["is_ip"] + X["has_userinfo"] + X["brand_in_sub"] + X["brand_in_domain"]
            + (X["kw_in_host"] > 0).astype(int) + X["is_punycode"]).astype(float)


def main():
    Xva, yva, _ = load_split("val")
    require_both_classes(yva, "val")
    s = rule_score(Xva)
    flagged = s > 0
    tp = int((flagged & (yva == 1)).sum())
    fp = int((flagged & (yva == 0)).sum())
    fn = int((~flagged & (yva == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"Rule (flag if any red flag): precision {prec:.3f}, recall {rec:.3f}")
    record("Rule baseline", yva, s, note=f"flag if any red flag: precision {prec:.2f}, recall {rec:.2f}")


if __name__ == "__main__":
    main()