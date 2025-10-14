"""Factory helpers for constructing :class:`StockEstimationAgent` instances."""
from __future__ import annotations

from typing import Optional

from .agent import StockEstimationAgent
from .data_sources import BrokerageTopGainersFetcher, NewsAPIFetcher, YFinanceHistoricalFetcher
from .estimation import EstimationEngine
from .indicators import IndicatorCalculator
from .top_gainers import TopGainerAnalytics


def create_default_agent(*, news_api_key: Optional[str] = None) -> StockEstimationAgent:
    """Create an agent wired with Yahoo! Finance and NewsAPI data sources."""

    if news_api_key is None:
        raise ValueError("news_api_key must be provided to construct the default agent.")

    historical_fetcher = YFinanceHistoricalFetcher()
    news_fetcher = NewsAPIFetcher(api_key=news_api_key)
    indicator_calculator = IndicatorCalculator()
    estimation_engine = EstimationEngine()
    top_gainers_fetcher = BrokerageTopGainersFetcher()
    top_gainer_analytics = TopGainerAnalytics()

    return StockEstimationAgent(
        historical_fetcher=historical_fetcher,
        news_fetcher=news_fetcher,
        indicator_calculator=indicator_calculator,
        estimation_engine=estimation_engine,
        top_gainers_fetcher=top_gainers_fetcher,
        top_gainer_analytics=top_gainer_analytics,
    )
