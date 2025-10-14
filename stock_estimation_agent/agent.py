"""Core agent orchestrating data collection, analysis, and estimation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
from pandas.tseries.offsets import BDay
from zoneinfo import ZoneInfo

US_CENTRAL = ZoneInfo("America/Chicago")

from .data_sources import HistoricalDataFetcher, MarketNewsFetcher, TopGainersFetcher
from .indicators import IndicatorCalculator
from .estimation import EstimationEngine, EstimationResult
from .sources import VerifiedSource
from .user_data import UserDatasetRegistry
from .top_gainers import TopGainerAnalytics


@dataclass
class AgentConfig:
    """Configuration controlling the behaviour of :class:`StockEstimationAgent`."""

    history_lookback: int = 365
    news_window_days: int = 7
    min_data_points: int = 60
    top_gainers_lookback_days: int = 5


@dataclass(frozen=True)
class AgentRecommendation:
    symbol: str
    name: str
    market_date: datetime
    confidence: float
    reference_close: float
    target_price: float
    potential_upside_pct: float
    risk_level: str
    narrative: str
    news_headlines: Sequence[str]
    sources: Sequence[VerifiedSource]
    top_gainer_factors: Dict[str, float | str]
    top_gainer_narrative: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "market_date": self.market_date.isoformat(),
            "confidence": self.confidence,
            "reference_close": self.reference_close,
            "target_price": self.target_price,
            "potential_upside_pct": self.potential_upside_pct,
            "risk_level": self.risk_level,
            "narrative": self.narrative,
            "news_headlines": list(self.news_headlines),
            "sources": [source.__dict__ for source in self.sources],
            "top_gainer_factors": dict(self.top_gainer_factors),
            "top_gainer_narrative": self.top_gainer_narrative,
        }


@dataclass
class StockEstimationAgent:
    """Agent that fuses market data, technical indicators, and news to estimate prices."""

    historical_fetcher: HistoricalDataFetcher
    news_fetcher: MarketNewsFetcher
    indicator_calculator: IndicatorCalculator
    estimation_engine: EstimationEngine
    top_gainers_fetcher: TopGainersFetcher
    top_gainer_analytics: TopGainerAnalytics
    user_registry: UserDatasetRegistry = field(default_factory=UserDatasetRegistry)
    config: AgentConfig = field(default_factory=AgentConfig)
    _last_market_date: Optional[pd.Timestamp] = field(default=None, init=False, repr=False)

    def estimate(self, symbol: str, *, as_of: Optional[datetime] = None) -> EstimationResult:
        """Return an estimation for ``symbol`` using the configured data sources."""

        if as_of is None:
            as_of = datetime.utcnow()

        _, market_date = self.detect_new_market_day(as_of=as_of)
        start_date = market_date - timedelta(days=self.config.history_lookback)
        price_history = self.historical_fetcher.fetch(
            symbol,
            start=start_date,
            end=(market_date + timedelta(days=1)),
        )
        price_history = price_history.loc[:pd.Timestamp(market_date)]

        if len(price_history) < self.config.min_data_points:
            raise ValueError(
                f"Insufficient history for {symbol}. "
                f"Expected at least {self.config.min_data_points} data points, "
                f"received {len(price_history)}."
            )

        user_augmented_history = self.user_registry.merge(symbol, price_history)
        indicators = self.indicator_calculator.compute_all(user_augmented_history)

        news = self.news_fetcher.fetch(
            symbol,
            as_of=market_date.to_pydatetime(),
            window_days=self.config.news_window_days,
        )

        try:
            top_gainers = self.top_gainers_fetcher.fetch(
                as_of=market_date.to_pydatetime(),
                lookback_days=self.config.top_gainers_lookback_days,
            )
        except Exception:  # pragma: no cover - defensive against network failures
            top_gainers = []
        candidate_symbols = set(self.available_symbols())
        candidate_symbols.update(g.symbol for g in top_gainers)
        candidate_symbols.discard(symbol)
        gainer_context = self.top_gainer_analytics.build_context(
            symbol=symbol,
            price_history=user_augmented_history,
            indicators=indicators,
            top_gainers=top_gainers,
            historical_fetcher=self.historical_fetcher,
            indicator_calculator=self.indicator_calculator,
            as_of=market_date.to_pydatetime(),
            universe_symbols=candidate_symbols,
        )

        return self.estimation_engine.estimate(
            symbol=symbol,
            price_history=user_augmented_history,
            indicators=indicators,
            news=news,
            sources=self.collate_sources(),
            gainer_context=gainer_context,
            market_date=market_date.to_pydatetime(),
        )

    def recommend_top_symbols(
        self,
        *,
        limit: int = 10,
        min_confidence: float = 0.7,
        as_of: Optional[datetime] = None,
        extra_symbols: Optional[Iterable[str]] = None,
    ) -> List[AgentRecommendation]:
        """Screen for the strongest opportunities using previous-day analytics.

        The routine evaluates the aggregated brokerage top gainers alongside any
        ``extra_symbols`` and returns the highest-confidence setups that align
        with the learned breakout pattern while satisfying ``min_confidence``.
        """

        if limit <= 0:
            return []

        _, market_date = self.detect_new_market_day(as_of=as_of)

        try:
            top_gainers = self.top_gainers_fetcher.fetch(
                as_of=market_date.to_pydatetime(),
                lookback_days=self.config.top_gainers_lookback_days,
            )
        except Exception:
            top_gainers = []

        top_gainer_lookup = {g.symbol: g for g in top_gainers}

        candidate_symbols: List[str] = []
        seen: set[str] = set()

        def _add_candidate(value: Optional[str]) -> None:
            if not value:
                return
            symbol = value.upper()
            if symbol in seen:
                return
            seen.add(symbol)
            candidate_symbols.append(symbol)

        sorted_gainers = sorted(top_gainers, key=lambda g: g.percent_change or 0.0, reverse=True)
        for gainer in sorted_gainers:
            _add_candidate(gainer.symbol)

        if extra_symbols:
            for symbol in extra_symbols:
                _add_candidate(symbol)

        for symbol in self.available_symbols():
            _add_candidate(symbol)

        recommendations: List[AgentRecommendation] = []

        for symbol in candidate_symbols:
            if len(recommendations) >= limit * 2:
                break
            try:
                result = self.estimate(symbol, as_of=market_date.to_pydatetime())
            except Exception:
                continue

            if result.confidence < min_confidence:
                continue

            recommendation = self._build_recommendation(
                estimation=result,
                name=getattr(top_gainer_lookup.get(symbol), "name", symbol),
            )
            recommendations.append(recommendation)

        recommendations.sort(key=lambda rec: rec.confidence, reverse=True)
        return recommendations[:limit]

    def _build_recommendation(
        self,
        *,
        estimation: EstimationResult,
        name: str,
    ) -> AgentRecommendation:
        reference_close = estimation.reference_close
        target_price = estimation.target_price
        potential_upside_pct = (target_price / reference_close - 1.0) * 100 if reference_close else 0.0

        band_upper = estimation.supporting_indicators.get("bollinger_upper")
        band_lower = estimation.supporting_indicators.get("bollinger_lower")
        band_width_pct = None
        if band_upper is not None and band_lower is not None and not math.isnan(band_upper) and not math.isnan(band_lower):
            band_width = band_upper - band_lower
            if reference_close:
                band_width_pct = band_width / reference_close

        risk_level = "Medium"
        if band_width_pct is not None:
            if band_width_pct >= 0.15 or estimation.confidence < 0.75:
                risk_level = "High"
            elif band_width_pct <= 0.05 and estimation.confidence >= 0.85:
                risk_level = "Low"
        elif estimation.confidence >= 0.85:
            risk_level = "Low"

        return AgentRecommendation(
            symbol=estimation.symbol,
            name=name or estimation.symbol,
            market_date=estimation.market_date,
            confidence=estimation.confidence,
            reference_close=reference_close,
            target_price=target_price,
            potential_upside_pct=potential_upside_pct,
            risk_level=risk_level,
            narrative=estimation.narrative,
            news_headlines=estimation.news_headlines,
            sources=estimation.sources,
            top_gainer_factors=estimation.top_gainer_factors,
            top_gainer_narrative=estimation.top_gainer_narrative,
        )

    def detect_new_market_day(self, *, as_of: Optional[datetime] = None) -> tuple[bool, pd.Timestamp]:
        """Return whether a new previous-day session has begun in U.S. Central Time.

        The check aligns to America/Chicago, matching the U.S. equities trading calendar. It
        tracks the most recent trading session the agent has analysed so callers can decide
        when to refresh cached datasets or rerun estimations with newly available prices and
        headlines.
        """

        reference = as_of or datetime.utcnow()
        market_date = self._previous_trading_day(reference)
        is_new = self._last_market_date is None or market_date > self._last_market_date
        if is_new:
            self._last_market_date = market_date
        return is_new, market_date

    def collate_sources(self) -> List[VerifiedSource]:
        """Combine sources from every component for transparency."""

        sources: List[VerifiedSource] = []
        sources.extend(self.historical_fetcher.verified_sources)
        sources.extend(self.news_fetcher.verified_sources)
        sources.extend(self.indicator_calculator.verified_sources)
        sources.extend(self.estimation_engine.verified_sources)
        sources.extend(self.top_gainers_fetcher.verified_sources)
        sources.extend(self.top_gainer_analytics.verified_sources)
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
            "top_gainers_lookback_days": self.config.top_gainers_lookback_days,
        }

    @staticmethod
    def _previous_trading_day(as_of: datetime) -> pd.Timestamp:
        """Return the most recent completed U.S. trading day prior to ``as_of``."""

        timestamp = pd.Timestamp(as_of)
        if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        central_timestamp = timestamp.tz_convert(US_CENTRAL)
        central_datetime = central_timestamp.to_pydatetime()
        central_date = pd.Timestamp(central_datetime.date())
        is_business_day = central_date.dayofweek < 5

        market_close = central_datetime.replace(hour=15, minute=0, second=0, microsecond=0)
        if is_business_day and central_datetime >= market_close:
            candidate = central_date
        else:
            candidate = central_date - BDay(1)

        return pd.Timestamp(candidate.normalize())
