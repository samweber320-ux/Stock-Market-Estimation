from pathlib import Path

from stock_estimation_agent.offline_recommendations import (
    load_offline_recommendations,
    offline_symbols,
)


def test_load_offline_recommendations(tmp_path: Path):
    payload = tmp_path / "snapshot.json"
    payload.write_text(
        """
        [
          {
            "symbol": "TEST",
            "company": "Test Corp",
            "pattern_confidence": 0.42,
            "thesis": "Sample thesis",
            "technical_snapshot": ["Tech detail"],
            "fundamental_snapshot": ["Fundamental"],
            "news_snapshot": ["News"],
            "sources": [
              {"name": "Source", "url": "http://example.com", "description": "Desc"}
            ]
          }
        ]
        """.strip()
    )

    recommendations = load_offline_recommendations(payload)
    assert len(recommendations) == 1
    rec = recommendations[0]
    assert rec.symbol == "TEST"
    assert rec.company == "Test Corp"
    assert rec.pattern_confidence == 0.42
    assert rec.technical_snapshot == ["Tech detail"]
    assert rec.fundamental_snapshot == ["Fundamental"]
    assert rec.news_snapshot == ["News"]
    assert rec.sources[0].name == "Source"
    assert rec.sources[0].url == "http://example.com"


def test_offline_symbols_reads_default(tmp_path: Path, monkeypatch):
    payload = tmp_path / "snapshot.json"
    payload.write_text(
        """
        [
          {"symbol": "AAA", "sources": []},
          {"symbol": "BBB", "sources": []}
        ]
        """.strip()
    )

    from stock_estimation_agent import offline_recommendations as module

    monkeypatch.setattr(module, "_default_path", lambda: payload)

    symbols = list(offline_symbols())
    assert symbols == ["AAA", "BBB"]
