from bire_repro.report import render_html


def test_report_escapes_content():
    report = {
        "generated_utc": "now",
        "config_sha256": "abc<def",
        "upstream": {},
        "data": {"valid": False},
        "figures": {},
        "known_limitations": ["x < y"],
    }
    rendered = render_html(report)
    assert "abc&lt;def" in rendered
    assert "x &lt; y" in rendered

