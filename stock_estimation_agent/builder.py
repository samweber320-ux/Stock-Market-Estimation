"""Factory helpers for constructing :class:`StockEstimationAgent` instances."""
from __future__ import annotations

from typing import Optional

from .agent import StockEstimationAgent
from .data_sources import NewsAPIFetcher, YFinanceHistoricalFetcher
from .estimation import EstimationEngine
from .indicators import IndicatorCalculator


def create_default_agent(*, news_api_key: Optional[str] = None) -> StockEstimationAgent:
    """Create an agent wired with Yahoo! Finance and NewsAPI data sources."""

    if news_api_key is None:
        raise ValueError("news_api_key must be provided to construct the default agent.")

    historical_fetcher = YFinanceHistoricalFetcher()
    news_fetcher = NewsAPIFetcher(api_key=news_api_key)
    indicator_calculator = IndicatorCalculator()
    estimation_engine = EstimationEngine()

    return StockEstimationAgent(
        historical_fetcher=historical_fetcher,
        news_fetcher=news_fetcher,
        indicator_calculator=indicator_calculator,
        estimation_engine=estimation_engine,
    )
