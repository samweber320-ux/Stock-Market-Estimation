from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

pandas = pytest.importorskip("pandas")
pd = pandas

from stock_estimation_agent.agent import AgentConfig, StockEstimationAgent, AgentRecommendation
from stock_estimation_agent.data_sources import (
    HistoricalDataFetcher,
    MarketNewsFetcher,
    StaticTopGainersFetcher,
    TopGainer,
)
from stock_estimation_agent.sources import VerifiedSource
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
    analysis_as_of = datetime(2025, 10, 13, 12, 0, 0)
    index = pd.date_range(end=datetime(2025, 10, 10), periods=120, freq="B")
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

    result = agent.estimate("TEST", as_of=analysis_as_of)

    assert result.symbol == "TEST"
    assert result.market_date.date() == datetime(2025, 10, 10).date()
    assert result.reference_close == pytest.approx(float(data["Close"].iloc[-1]))
    assert result.target_price > 0
    assert 0.05 <= result.confidence <= 0.95
    assert result.supporting_indicators["sma_20"] > 0
    assert result.news_headlines
    assert isinstance(result.top_gainer_factors, dict)
    assert result.potential_top_gainers
    assert all(item.get("pattern_confidence", 0) >= 0.7 for item in result.potential_top_gainers)
    assert result.top_gainer_narrative


def test_detect_new_market_day_tracks_central_time_rollover():
    analysis_as_of = datetime(2025, 10, 13, 12, 0, 0)
    index = pd.date_range(end=datetime(2025, 10, 10), periods=120, freq="B")
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
    agent = StockEstimationAgent(
        historical_fetcher=historical,
        news_fetcher=news,
        indicator_calculator=indicators,
        estimation_engine=estimation,
        top_gainers_fetcher=StaticTopGainersFetcher([]),
        top_gainer_analytics=TopGainerAnalytics(),
        config=AgentConfig(history_lookback=365, news_window_days=7, min_data_points=60),
    )

    is_new, market_date = agent.detect_new_market_day(as_of=analysis_as_of)
    assert is_new is True
    assert market_date.date() == datetime(2025, 10, 10).date()

    central_midday = datetime(2025, 10, 14, 12, 0, 0, tzinfo=ZoneInfo("America/Chicago"))
    is_new, market_date = agent.detect_new_market_day(as_of=central_midday)
    assert is_new is True
    assert market_date.date() == datetime(2025, 10, 13).date()

    another_check = datetime(2025, 10, 14, 13, 0, 0, tzinfo=ZoneInfo("America/Chicago"))
    is_new, market_date = agent.detect_new_market_day(as_of=another_check)
    assert is_new is False
    assert market_date.date() == datetime(2025, 10, 13).date()

    post_close = datetime(2025, 10, 14, 16, 5, 0, tzinfo=ZoneInfo("America/Chicago"))
    is_new, market_date = agent.detect_new_market_day(as_of=post_close)
    assert is_new is True
    assert market_date.date() == datetime(2025, 10, 14).date()


def test_recommend_top_symbols_filters_by_confidence():
    analysis_as_of = datetime(2025, 10, 13, 12, 0, 0)
    index = pd.date_range(end=datetime(2025, 10, 10), periods=120, freq="B")
    closes = list(range(1, 121))
    data = pd.DataFrame(
        {
            "Open": closes,
            "High": [value + 1 for value in closes],
            "Low": [max(value - 1, 0) for value in closes],
            "Close": closes,
            "Volume": [1_500_000] * 120,
        },
        index=index,
    )

    historical = DummyHistorical(data)
    news = DummyNews()
    indicators = IndicatorCalculator()
    estimation = EstimationEngine()
    gainers = [
        TopGainer(symbol="ABC", name="Alpha Brands", last_price=50.0, percent_change=9.5, volume=2_000_000, average_volume=1_000_000, source="Webull"),
        TopGainer(symbol="XYZ", name="Xylon Corp", last_price=32.0, percent_change=6.2, volume=1_500_000, average_volume=700_000, source="Robinhood"),
        TopGainer(symbol="LMN", name="Lumina Tech", last_price=78.0, percent_change=5.9, volume=1_200_000, average_volume=600_000, source="Webull"),
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

    recommendations = agent.recommend_top_symbols(
        limit=5,
        min_confidence=0.6,
        as_of=analysis_as_of,
        extra_symbols=["TEST"],
    )

    assert recommendations
    assert all(isinstance(rec, AgentRecommendation) for rec in recommendations)
    assert all(rec.confidence >= 0.6 for rec in recommendations)
    assert all(rec.potential_upside_pct > -5 for rec in recommendations)
    assert recommendations[0].market_date.date() == datetime(2025, 10, 10).date()
