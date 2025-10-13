"""Core agent orchestrating data collection, analysis, and estimation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .data_sources import (
    HistoricalDataFetcher,
    MarketNewsFetcher,
    VerifiedSource,
)
from .indicators import IndicatorCalculator
from .estimation import EstimationEngine, EstimationResult
from .user_data import UserDatasetRegistry


@dataclass
class AgentConfig:
    """Configuration controlling the behaviour of :class:`StockEstimationAgent`."""

    history_lookback: int = 365
    news_window_days: int = 7
    min_data_points: int = 60


@dataclass
class StockEstimationAgent:
    """Agent that fuses market data, technical indicators, and news to estimate prices."""

    historical_fetcher: HistoricalDataFetcher
    news_fetcher: MarketNewsFetcher
    indicator_calculator: IndicatorCalculator
    estimation_engine: EstimationEngine
    user_registry: UserDatasetRegistry = field(default_factory=UserDatasetRegistry)
    config: AgentConfig = field(default_factory=AgentConfig)

    def estimate(self, symbol: str, *, as_of: Optional[datetime] = None) -> EstimationResult:
        """Return an estimation for ``symbol`` using the configured data sources."""

        if as_of is None:
            as_of = datetime.utcnow()

        start_date = as_of - timedelta(days=self.config.history_lookback)
        price_history = self.historical_fetcher.fetch(symbol, start=start_date, end=as_of)

        if len(price_history) < self.config.min_data_points:
            raise ValueError(
                f"Insufficient history for {symbol}. "
                f"Expected at least {self.config.min_data_points} data points, "
                f"received {len(price_history)}."
            )

        user_augmented_history = self.user_registry.merge(symbol, price_history)
        indicators = self.indicator_calculator.compute_all(user_augmented_history)

        news = self.news_fetcher.fetch(symbol, as_of=as_of, window_days=self.config.news_window_days)

        return self.estimation_engine.estimate(
            symbol=symbol,
            price_history=user_augmented_history,
            indicators=indicators,
            news=news,
            sources=self.collate_sources(),
        )

    def collate_sources(self) -> List[VerifiedSource]:
        """Combine sources from every component for transparency."""

        sources: List[VerifiedSource] = []
        sources.extend(self.historical_fetcher.verified_sources)
        sources.extend(self.news_fetcher.verified_sources)
        sources.extend(self.indicator_calculator.verified_sources)
        sources.extend(self.estimation_engine.verified_sources)
        return sources

    def register_user_dataset(self, symbol: str, dataset: pd.DataFrame) -> None:
        """Add a user supplied dataset (e.g. proprietary indicators)."""

        self.user_registry.register(symbol, dataset)

    def available_symbols(self) -> Iterable[str]:
        """Return symbols with either historical data or user uploads."""

        symbols = set(self.user_registry.available_symbols())
        symbols.update(self.historical_fetcher.available_symbols())
        return symbols

    def explain_configuration(self) -> Dict[str, int]:
        """Expose key configuration parameters for observability."""

        return {
            "history_lookback_days": self.config.history_lookback,
            "news_window_days": self.config.news_window_days,
            "min_data_points": self.config.min_data_points,
        }
