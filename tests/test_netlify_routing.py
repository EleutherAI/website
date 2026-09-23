from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def production_site(tmp_path_factory):
    destination = tmp_path_factory.mktemp("netlify-site")
    subprocess.run(
        ["hugo", "--config", "hugo.toml,hugo-netlify.toml",
         "--destination", str(destination), "--panicOnWarning"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return destination


def test_production_aliases_do_not_shadow_pages(production_site):
    redirects = (production_site / "_redirects").read_text().splitlines()
    assert "/community.html /community/ 301" in redirects
    assert "/research/evaluation-approach /research/evaluation/ 301" in redirects
    for rule in redirects:
        source, target, status = rule.split()
        assert status == "301"
        assert source.rstrip("/") != target.rstrip("/")
        assert not (production_site / source.lstrip("/")).is_file()
        assert (production_site / target.lstrip("/") / "index.html").is_file()
    for slug in ("community", "about", "papers", "staff", "research", "soar"):
        assert not (production_site / f"{slug}.html").exists()
        page = (production_site / slug / "index.html").read_text()
        assert '<meta http-equiv="refresh"' not in page
        assert "<main" in page


def test_local_preview_keeps_html_aliases(tmp_path, production_site):
    subprocess.run(
        ["hugo", "--config", "hugo.toml", "--destination", str(tmp_path)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert 'url=/community/' in (tmp_path / "community.html").read_text()
    local_page = (tmp_path / "community/index.html").read_text()
    production_page = (production_site / "community/index.html").read_text()
    assert local_page.split("<body>", 1)[1] == production_page.split("<body>", 1)[1]
