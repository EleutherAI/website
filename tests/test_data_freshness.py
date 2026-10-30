import pytest
import io
import json
from urllib.parse import parse_qs, urlsplit

import generate_hugo_data as generator


def test_hf_downloads_paginates_and_applies_offset_once(monkeypatch):
    monkeypatch.delenv("HF_MODEL_DOWNLOADS", raising=False)
    next_url = "https://huggingface.co/api/models?page=2"
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        response = io.StringIO(json.dumps([
            {"id": "EleutherAI/a" if len(calls) == 1 else "EleutherAI/b",
             "downloadsAllTime": 140_000_000 if len(calls) == 1 else 3_999_999}
        ]))
        response.headers = {"Link": f'<{next_url}>; rel="next"'} if len(calls) == 1 else {}
        return response

    monkeypatch.setattr(generator, "urlopen", fetch)
    assert generator.fetch_hf_model_downloads() == "154M+"
    assert calls == [generator.HF_MODELS_API_URL, next_url]


def test_hf_manual_override_is_not_adjusted(monkeypatch):
    monkeypatch.setenv("HF_MODEL_DOWNLOADS", "154M+")
    assert generator.fetch_hf_model_downloads() == "154M+"


@pytest.mark.parametrize("count", [None, -1, "123", True])
def test_hf_invalid_count_preserves_cache(monkeypatch, tmp_path, count):
    monkeypatch.delenv("HF_MODEL_DOWNLOADS", raising=False)
    monkeypatch.setattr(generator, "HF_MODEL_DOWNLOADS_CACHE", tmp_path / "hf.json")
    generator.write_hf_downloads_cache("154M+")

    def fetch(url, timeout):
        response = io.StringIO(json.dumps([{"id": "EleutherAI/a", "downloadsAllTime": count}]))
        response.headers = {}
        return response

    monkeypatch.setattr(generator, "urlopen", fetch)
    assert generator.hf_model_downloads_payload() == ("154M+", "cache")
    assert generator.hf_model_downloads_payload(offline=True) == ("154M+", "offline")


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
