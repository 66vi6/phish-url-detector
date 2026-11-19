import math
import re
from collections import Counter

import tldextract
from rapidfuzz.distance import Levenshtein

# Offline mode: use the bundled public-suffix list, no network calls.
_extract = tldextract.TLDExtract(suffix_list_urls=())

# brand token -> set of its REAL registered domains. Extend this to ~100-200 brands.
BRANDS = {
    "paypal": {"paypal.com"},
    "google": {"google.com"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "outlook.com"},
    "apple": {"apple.com", "icloud.com"},
    "amazon": {"amazon.com", "amazon.com.au"},
    "facebook": {"facebook.com"},
    "netflix": {"netflix.com"},
    "instagram": {"instagram.com"},
    "linkedin": {"linkedin.com"},
    "dropbox": {"dropbox.com"},
    "chase": {"chase.com"},
    "wellsfargo": {"wellsfargo.com"},
    "ebay": {"ebay.com", "ebay.com.au"},
    "dhl": {"dhl.com"},
    "fedex": {"fedex.com"},
    "commbank": {"commbank.com.au"},
    "westpac": {"westpac.com.au"},
    "anz": {"anz.com", "anz.com.au"},
    "nab": {"nab.com.au"},
    "auspost": {"auspost.com.au"},
    "mygov": {"my.gov.au"},
    "telstra": {"telstra.com", "telstra.com.au"},
    "optus": {"optus.com.au"},
    "centrelink": {"servicesaustralia.gov.au"},
}
# Only fuzzy-match brands with 5+ letters; short names cause false positives.
FUZZY_BRANDS = [b for b in BRANDS if len(b) >= 5]

KEYWORDS = ("login", "signin", "verify", "secure", "account", "update", "confirm",
            "bank", "wallet", "password", "billing", "support", "invoice")

_HOMOGLYPHS = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "$": "s", "@": "a"})


def entropy(s: str) -> float:
    if not s:
        return 0.0
    counts, n = Counter(s), len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def tokens(s: str) -> list:
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def deglyph(s: str) -> str:
    """paypa1 -> paypal, rnicrosoft -> microsoft"""
    return s.translate(_HOMOGLYPHS).replace("rn", "m").replace("vv", "w")


def split_url(u: str):
    """Input is a NORMALISED url (no scheme). Returns userinfo, host, port, path, query."""
    m = re.match(r"([^/?]*)(.*)", u)
    authority, rest = m.group(1), m.group(2)
    userinfo, _, hostport = authority.rpartition("@")   # no '@' -> userinfo == ""
    host, _, port = hostport.partition(":")
    path, _, query = rest.partition("?")
    return userinfo, host, port, path, query


def _fake_brand(toks, reg) -> int:
    """1 if any token is a brand name but the registered domain isn't that brand's real one."""
    return int(any(t in BRANDS and reg not in BRANDS[t] for t in toks))


def _lookalike_dist(label: str, reg: str) -> int:
    """Smallest edit distance (capped at 5) between the de-glyphed domain label and a brand name,
    ignoring the brand the URL really belongs to."""
    label = deglyph(label)
    if not label:
        return 5
    best = 5
    for b in FUZZY_BRANDS:
        if reg in BRANDS[b]:
            continue
        best = min(best, Levenshtein.distance(label, b))
    return best


def extract(url: str) -> dict:
    userinfo, host, port, path, query = split_url(url)
    ext = _extract(host)
    sub, dom = ext.subdomain.lower(), ext.domain.lower()
    suffix, reg = ext.suffix.lower(), ext.registered_domain.lower()
    low = url.lower()
    pq = (path + "?" + query).lower()
    dist = _lookalike_dist(dom, reg)

    return {
        # size
        "len_url": len(url),
        "len_host": len(host),
        "len_path": len(path),
        "len_query": len(query),
        # structure
        "n_dots_host": host.count("."),
        "n_hyphens_host": host.count("-"),
        "n_digits_host": sum(c.isdigit() for c in host),
        "n_slashes": path.count("/"),
        "n_params": (query.count("&") + 1) if query else 0,
        "sub_depth": len(sub.split(".")) if sub else 0,
        "tld_len": len(suffix),
        # host tricks
        "is_ip": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host)) or host.startswith("[")),
        "has_userinfo": int(bool(userinfo)),
        "has_port": int(bool(port)),
        "is_punycode": int("xn--" in host),
        # randomness
        "ent_host": entropy(host),
        "ent_path": entropy(path),
        "digit_ratio": sum(c.isdigit() for c in url) / max(len(url), 1),
        "longest_token": max((len(t) for t in re.split(r"[./\-_?=&]", url)), default=0),
        "longest_digit_run": max((len(m) for m in re.findall(r"\d+", url)), default=0),
        # keywords, by location
        "kw_total": sum(k in low for k in KEYWORDS),
        "kw_in_host": sum(k in host.lower() for k in KEYWORDS),
        "kw_in_path": sum(k in pq for k in KEYWORDS),
        # brand abuse
        "brand_in_sub": _fake_brand(tokens(sub), reg),
        "brand_in_domain": _fake_brand(tokens(dom), reg),
        "brand_in_path": _fake_brand(tokens(pq), reg),
        "lookalike_dist": dist,
        "is_lookalike": int(dist <= 2),
        # misc obfuscation
        "n_pct": url.count("%"),
        "embedded_http": int("http" in pq),
    }


FEATURE_NAMES = list(extract("example.com/a").keys())