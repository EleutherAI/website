import pytest
from urllib.parse import parse_qs, urlsplit

import generate_hugo_data as generator


def test_papers_export_targets_research_papers_tab():
    query = parse_qs(urlsplit(generator.PAPERS_SHEET_CSV_URL).query)
    assert query["gid"] == ["2053751678"]
    assert query["format"] == ["csv"]


def test_strict_freshness_accepts_live_and_provided_sources():
    generator.require_fresh_sources(
        {
            "sheet": {"status": "live"},
            "downloads": {"status": "provided"},
            "blog": {"status": "local"},
        }
    )


def test_strict_freshness_rejects_cache_and_missing_sources():
    with pytest.raises(SystemExit, match="scholar, downloads"):
        generator.require_fresh_sources(
            {
                "sheet": {"status": "live"},
                "scholar": {"status": "cache"},
                "downloads": {"status": "missing"},
            }
        )
