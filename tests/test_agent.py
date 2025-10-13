from datetime import datetime, timedelta

import pytest

pandas = pytest.importorskip("pandas")
pd = pandas

from stock_estimation_agent.agent import AgentConfig, StockEstimationAgent
from stock_estimation_agent.data_sources import (
    HistoricalDataFetcher,
    MarketNewsFetcher,
    StaticTopGainersFetcher,
    TopGainer,
    VerifiedSource,
)
from stock_estimation_agent.estimation import EstimationEngine
from stock_estimation_agent.indicators import IndicatorCalculator
from stock_estimation_agent.top_gainers import TopGainerAnalytics


class DummyHistorical(HistoricalDataFetcher):
    verified_sources = [VerifiedSource(name="Dummy", url="http://example.com", description="Test source")]

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def fetch(self, symbol: str, *, start: datetime, end: datetime) -> pd.DataFrame:
        return self._frame

    def available_symbols(self):
        return ["TEST", "ABC", "XYZ"]


class DummyNews(MarketNewsFetcher):
    verified_sources = [VerifiedSource(name="Dummy News", url="http://example.com/news", description="Test news")]

    def fetch(self, symbol: str, *, as_of: datetime, window_days: int):
        return [
            {"title": "Positive outlook", "publishedAt": as_of.isoformat(), "sentiment": "positive", "symbols": [symbol]},
            {"title": "Caution on supply chain", "publishedAt": (as_of - timedelta(days=1)).isoformat(), "sentiment": "negative", "symbols": [symbol]},
        ]

    def available_symbols(self):
        return ["TEST"]


def test_agent_generates_estimation():
    index = pd.date_range(end=datetime.utcnow(), periods=120, freq="B")
    data = pd.DataFrame(
        {
            "Open": range(120),
            "High": range(1, 121),
            "Low": range(0, 120),
            "Close": range(1, 121),
            "Volume": [1_000_000] * 120,
        },
        index=index,
    )

    historical = DummyHistorical(data)
    news = DummyNews()
    indicators = IndicatorCalculator()
    estimation = EstimationEngine()
    gainers = [
        TopGainer(symbol="ABC", name="Alpha Brands", last_price=50.0, percent_change=8.5, volume=2_000_000, average_volume=1_000_000, source="Webull"),
        TopGainer(symbol="XYZ", name="Xylon Corp", last_price=32.0, percent_change=6.2, volume=1_500_000, average_volume=700_000, source="Robinhood"),
    ]

    agent = StockEstimationAgent(
        historical_fetcher=historical,
        news_fetcher=news,
        indicator_calculator=indicators,
        estimation_engine=estimation,
        top_gainers_fetcher=StaticTopGainersFetcher(gainers),
        top_gainer_analytics=TopGainerAnalytics(),
        config=AgentConfig(history_lookback=365, news_window_days=7, min_data_points=60),
    )

    result = agent.estimate("TEST")

    assert result.symbol == "TEST"
    assert result.target_price > 0
    assert 0.05 <= result.confidence <= 0.95
    assert result.supporting_indicators["sma_20"] > 0
    assert result.news_headlines
    assert isinstance(result.top_gainer_factors, dict)
    assert result.potential_top_gainers
    assert all(item.get("pattern_confidence", 0) >= 0.3 for item in result.potential_top_gainers)
    assert result.top_gainer_narrative
