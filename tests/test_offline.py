import json
from datetime import datetime, timezone

from stock_estimation_agent.offline import (
    OfflineRecommendation,
    OfflineNewsSource,
    load_offline_recommendations,
)


def test_load_offline_recommendations(tmp_path):
    payload = {
        "market_date": "2025-10-10",
        "universe": [
            {
                "symbol": "TEST",
                "name": "Test Corp",
                "close": 123.45,
                "pattern_confidence": 0.75,
                "potential_upside_pct": 0.2,
                "risk_level": "Medium",
                "indicator_alignment": ["SMA", "RSI"],
                "historical_response": "Median +2% next day",
                "narrative": "Constructive setup",
                "news_sources": [
                    {
                        "title": "Test headline",
                        "url": "https://example.com/article",
                        "publisher": "Reuters",
                        "publishedAt": "2025-10-09T15:00:00Z",
                    }
                ],
            }
        ],
    }
    json_path = tmp_path / "offline.json"
    json_path.write_text(json.dumps(payload))

    recommendations = load_offline_recommendations(json_path)

    assert len(recommendations) == 1
    rec = recommendations[0]
    assert isinstance(rec, OfflineRecommendation)
    assert rec.symbol == "TEST"
    assert rec.name == "Test Corp"
    assert rec.market_date == datetime(2025, 10, 10)
    assert abs(rec.close - 123.45) < 1e-6
    assert abs(rec.pattern_confidence - 0.75) < 1e-6
    assert rec.indicator_alignment == ("SMA", "RSI")
    assert rec.news_sources and isinstance(rec.news_sources[0], OfflineNewsSource)
    assert rec.news_sources[0].published_at == datetime(2025, 10, 9, 15, 0, tzinfo=timezone.utc)
