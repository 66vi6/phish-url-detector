from src.features import extract, split_url, deglyph, FEATURE_NAMES


def test_split_url_userinfo():
    userinfo, host, port, path, query = split_url("paypal.com@evil.com:8080/login?a=1")
    assert (userinfo, host, port, path, query) == ("paypal.com", "evil.com", "8080", "/login", "a=1")


def test_split_url_plain():
    assert split_url("example.com") == ("", "example.com", "", "", "")


def test_deglyph():
    assert deglyph("paypa1") == "paypal"
    assert deglyph("rnicrosoft") == "microsoft"


def test_lookalike_flagged():
    f = extract("paypa1.com/login")
    assert f["lookalike_dist"] == 0 and f["is_lookalike"] == 1


def test_real_brand_not_flagged():
    f = extract("www.paypal.com/signin")
    assert f["is_lookalike"] == 0 and f["brand_in_sub"] == 0 and f["brand_in_domain"] == 0


def test_brand_in_subdomain_of_other_domain():
    f = extract("paypal.com.secure-login.example.xyz/verify")
    assert f["brand_in_sub"] == 1
    assert f["sub_depth"] >= 2
    assert f["kw_in_host"] >= 1


def test_brand_in_fake_domain():
    assert extract("paypal-secure.com/x")["brand_in_domain"] == 1


def test_brand_in_real_subdomain_is_fine():
    assert extract("accounts.google.com/login")["brand_in_sub"] == 0


def test_ip_host():
    f = extract("192.0.2.5/login")
    assert f["is_ip"] == 1 and f["sub_depth"] == 0


def test_userinfo_flag():
    assert extract("paypal.com@evil.com/login")["has_userinfo"] == 1


def test_punycode():
    assert extract("xn--pypal-4ve.com/")["is_punycode"] == 1


def test_short_brand_no_false_positive():
    # "nab"/"anz" are too short to fuzzy-match, and "organize" shouldn't count as "anz"
    assert extract("organize-events.com/home")["brand_in_domain"] == 0


def test_empty_path_and_query_ok():
    f = extract("example.com")
    assert f["len_path"] == 0 and f["n_params"] == 0


def test_all_features_numeric_and_stable():
    f = extract("a.b.example.co.uk/x/y?z=1&w=2")
    assert list(f.keys()) == FEATURE_NAMES
    assert all(isinstance(v, (int, float)) for v in f.values())