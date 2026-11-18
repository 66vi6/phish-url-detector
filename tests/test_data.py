from src.data import normalize, template, reg_domain

def test_normalize_scheme_case_fragment():
    assert normalize("HTTPS://Example.com/Path#frag") == "example.com/Path"

def test_normalize_trailing_slash():
    assert normalize("http://example.com/") == "example.com"

def test_normalize_query_without_path_keeps_case():
    assert normalize("http://Example.com?ID=5") == "example.com?ID=5"

def test_template_collapses_tokens():
    a = template("x.com/a1b2c3d4e5f60789/v?id=48213")
    b = template("x.com/9f8e7d6c5b4a3210/v?id=77104")
    assert a == b

def test_reg_domain_subdomains_and_ip():
    assert reg_domain("login.secure.example.co.uk/a") == "example.co.uk"
    assert reg_domain("192.0.2.5/login") == "192.0.2.5"