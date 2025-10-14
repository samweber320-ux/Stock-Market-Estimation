"""Utilities for analysing top gainer lists from retail brokerages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence

import math

import pandas as pd

from .data_sources import HistoricalDataFetcher, TopGainer
from .sources import VerifiedSource
from .indicators import IndicatorCalculator


@dataclass
class TopGainerContext:
    """Structured summary of top gainer dynamics used to augment estimations."""

    factors: Dict[str, float | str]
    narrative: str
    sidebar: List[Dict[str, float | str]]
    alignment_bias: float


@dataclass
class TopGainerAnalytics:
    """Derive common traits from top gainers and evaluate alignment for a symbol."""

    lookback_window_days: int = 120
    breakout_analysis_days: int = 100
    pre_breakout_window: int = 3
    min_pattern_confidence: float = 0.3
    max_sidebar: int = 5
    verified_sources: List[VerifiedSource] = field(
        default_factory=lambda: [
            VerifiedSource(
                name="Investopedia Technical Analysis",
                url="https://www.investopedia.com/terms/t/technicalanalysis.asp",
                description="Best practices on leveraging momentum, volume, and sector leadership signals.",
            ),
            VerifiedSource(
                name="CMT Association Momentum Study",
                url="https://cmtassociation.org/knowledge/technical-analysis-of-stocks-trends/",
                description="Professional research on recognising pre-breakout momentum patterns across leading equities.",
            ),
        ]
    )

    def build_context(
        self,
        *,
        symbol: str,
        price_history: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        top_gainers: Sequence[TopGainer],
        historical_fetcher: Optional[HistoricalDataFetcher] = None,
        indicator_calculator: Optional[IndicatorCalculator] = None,
        as_of: Optional[datetime] = None,
        universe_symbols: Optional[Iterable[str]] = None,
    ) -> TopGainerContext:
        if not top_gainers:
            return TopGainerContext(factors={}, narrative="", sidebar=[], alignment_bias=0.0)

        factors = self._identify_common_factors(top_gainers)
        narrative = self._compose_narrative(factors, top_gainers)
        sidebar = self._build_sidebar(top_gainers, exclude_symbol=symbol)
        alignment_bias = self._alignment_score(price_history, indicators, factors)

        if historical_fetcher and indicator_calculator:
            pattern_summary = self._learn_pre_breakout_pattern(
                top_gainers=top_gainers,
                historical_fetcher=historical_fetcher,
                indicator_calculator=indicator_calculator,
                as_of=as_of or datetime.utcnow(),
            )
            if pattern_summary:
                factors.update(pattern_summary)
                pattern_candidates = self._screen_for_pattern_alignment(
                    symbol=symbol,
                    pattern_summary=pattern_summary,
                    historical_fetcher=historical_fetcher,
                    indicator_calculator=indicator_calculator,
                    as_of=as_of or datetime.utcnow(),
                    universe_symbols=universe_symbols,
                    seed_sidebar=sidebar,
                )
                if pattern_candidates:
                    sidebar = pattern_candidates

        return TopGainerContext(
            factors=factors,
            narrative=narrative,
            sidebar=sidebar,
            alignment_bias=alignment_bias,
        )

    def _identify_common_factors(self, top_gainers: Sequence[TopGainer]) -> Dict[str, float | str]:
        percent_changes = [g.percent_change for g in top_gainers if g.percent_change is not None]
        relative_volumes: List[float] = []
        market_caps = [g.market_cap for g in top_gainers if g.market_cap]

        for gainer in top_gainers:
            if gainer.volume and gainer.average_volume and gainer.average_volume > 0:
                relative_volumes.append(gainer.volume / gainer.average_volume)

        factors: Dict[str, float | str] = {}
        if percent_changes:
            factors["avg_percent_change"] = float(mean(percent_changes))
            factors["median_percent_change"] = float(median(percent_changes))
        if relative_volumes:
            factors["avg_relative_volume"] = float(mean(relative_volumes))
        if market_caps:
            factors["median_market_cap"] = float(median(market_caps))

        momentum_ratio = sum(1 for g in top_gainers if g.percent_change and g.percent_change > 5) / max(len(top_gainers), 1)
        factors["momentum_share"] = float(momentum_ratio)

        sector_counts: Dict[str, int] = {}
        for gainer in top_gainers:
            if gainer.sector:
                sector_counts[gainer.sector] = sector_counts.get(gainer.sector, 0) + 1
        if sector_counts:
            dominant_sector, count = max(sector_counts.items(), key=lambda item: item[1])
            factors["dominant_sector_weight"] = count / len(top_gainers)
            factors["dominant_sector_score"] = float(count)
            factors["dominant_sector_name"] = dominant_sector
        return factors

    def _compose_narrative(self, factors: Dict[str, float | str], gainers: Sequence[TopGainer]) -> str:
        pieces: List[str] = []

        if "avg_percent_change" in factors:
            avg_move = float(factors["avg_percent_change"])
            median_move = float(factors.get("median_percent_change", avg_move))
            pieces.append(
                f"Recent brokerage top gainers advanced an average of {avg_move:.2f}% with a median move of {median_move:.2f}%."
            )
        if "avg_relative_volume" in factors:
            avg_rel_volume = float(factors["avg_relative_volume"])
            pieces.append(
                f"Relative volume averaged {avg_rel_volume:.2f}x normal turnover, signalling strong participation."
            )
        if "dominant_sector_weight" in factors:
            sector_weight = float(factors["dominant_sector_weight"])
            sector_name = factors.get("dominant_sector_name")
            if sector_name:
                pieces.append(
                    f"{sector_name} leadership made up {sector_weight * 100:.0f}% of tracked movers, highlighting where risk appetite is strongest."
                )
            else:
                pieces.append(
                    "Sector leadership concentration is evident, suggesting momentum can persist when confirmed by volume and breadth indicators."
                )

        if not pieces:
            pieces.append(
                "Recent top gainers share elevated momentum characteristics despite varied sector exposure."
            )

        return " ".join(pieces)

    def _build_sidebar(
        self, top_gainers: Sequence[TopGainer], *, exclude_symbol: Optional[str]
    ) -> List[Dict[str, float | str]]:
        sidebar: List[Dict[str, float | str]] = []
        sorted_gainers = sorted(top_gainers, key=lambda g: g.percent_change or 0.0, reverse=True)

        for gainer in sorted_gainers:
            if exclude_symbol and gainer.symbol == exclude_symbol:
                continue
            last_price = float(gainer.last_price) if gainer.last_price is not None else 0.0
            sidebar.append(
                {
                    "symbol": gainer.symbol,
                    "name": gainer.name,
                    "percent_change": round(gainer.percent_change, 2) if gainer.percent_change is not None else None,
                    "last_price": round(last_price, 2),
                    "source": gainer.source,
                }
            )
            if len(sidebar) == 5:
                break
        return sidebar

    def _learn_pre_breakout_pattern(
        self,
        *,
        top_gainers: Sequence[TopGainer],
        historical_fetcher: HistoricalDataFetcher,
        indicator_calculator: IndicatorCalculator,
        as_of: datetime,
    ) -> Dict[str, float | str]:
        window_start = as_of - timedelta(days=self.lookback_window_days)
        rsi_samples: List[float] = []
        macd_samples: List[float] = []
        sma_spread_samples: List[float] = []
        volume_ratio_samples: List[float] = []

        for gainer in top_gainers:
            try:
                history = historical_fetcher.fetch(gainer.symbol, start=window_start, end=as_of)
            except Exception:
                continue
            if history.empty or len(history) < self.pre_breakout_window + 5:
                continue
            history = history.sort_index()

            closes = history["Close"] if "Close" in history else history.iloc[:, 0]
            returns = closes.pct_change().dropna()
            if returns.empty:
                continue

            analysis_window = returns.tail(self.breakout_analysis_days)
            if analysis_window.empty:
                continue
            breakout_idx = analysis_window.idxmax()
            if breakout_idx is None:
                continue
            breakout_return = analysis_window.loc[breakout_idx]
            if breakout_return <= 0:
                continue

            try:
                breakout_position = history.index.get_loc(breakout_idx)
            except KeyError:
                continue
            if breakout_position == 0:
                continue

            pre_breakout_position = max(0, breakout_position - 1)
            pre_breakout_index = history.index[pre_breakout_position]

            indicators = indicator_calculator.compute_all(history)

            rsi_series = indicators.get("rsi_14")
            macd_series = indicators.get("macd")
            sma_20_series = indicators.get("sma_20")
            sma_50_series = indicators.get("sma_50")

            rsi_value = self._safe_float_from_series(rsi_series, pre_breakout_index)
            macd_value = self._safe_float_from_series(macd_series, pre_breakout_index)
            sma_spread = None
            if sma_20_series is not None and sma_50_series is not None:
                sma_short = self._safe_float_from_series(sma_20_series, pre_breakout_index)
                sma_long = self._safe_float_from_series(sma_50_series, pre_breakout_index)
                if sma_short is not None and sma_long is not None:
                    sma_spread = sma_short - sma_long

            if "Volume" in history and breakout_position > self.pre_breakout_window:
                recent_volume = (
                    history["Volume"].iloc[breakout_position - self.pre_breakout_window : breakout_position].mean()
                )
                long_volume = history["Volume"].iloc[:breakout_position].tail(20).mean()
                volume_ratio = (recent_volume / long_volume) if long_volume and long_volume > 0 else None
            else:
                volume_ratio = None

            if rsi_value is not None and not math.isnan(rsi_value):
                rsi_samples.append(float(rsi_value))
            if macd_value is not None and not math.isnan(macd_value):
                macd_samples.append(float(macd_value))
            if sma_spread is not None and not math.isnan(sma_spread):
                sma_spread_samples.append(float(sma_spread))
            if volume_ratio is not None and not math.isnan(volume_ratio):
                volume_ratio_samples.append(float(volume_ratio))

        samples = max(len(rsi_samples), len(macd_samples), len(sma_spread_samples), len(volume_ratio_samples))
        if samples == 0:
            return {}

        summary: Dict[str, float | str] = {"pattern_sample_size": float(samples)}
        if rsi_samples:
            summary["pattern_avg_rsi"] = float(mean(rsi_samples))
        if macd_samples:
            summary["pattern_avg_macd"] = float(mean(macd_samples))
        if sma_spread_samples:
            summary["pattern_avg_sma_spread"] = float(mean(sma_spread_samples))
        if volume_ratio_samples:
            summary["pattern_avg_volume_ratio"] = float(mean(volume_ratio_samples))

        return summary

    def _screen_for_pattern_alignment(
        self,
        *,
        symbol: str,
        pattern_summary: Dict[str, float | str],
        historical_fetcher: HistoricalDataFetcher,
        indicator_calculator: IndicatorCalculator,
        as_of: datetime,
        universe_symbols: Optional[Iterable[str]],
        seed_sidebar: List[Dict[str, float | str]],
    ) -> List[Dict[str, float | str]]:
        candidate_symbols: List[str] = []
        if universe_symbols:
            candidate_symbols.extend(s for s in universe_symbols if s and isinstance(s, str))
        candidate_symbols.extend(item.get("symbol") for item in seed_sidebar if item.get("symbol"))
        # Remove duplicates while preserving order
        seen: set[str] = set()
        ordered_candidates: List[str] = []
        for candidate in candidate_symbols:
            if not candidate or candidate in seen or candidate == symbol:
                continue
            seen.add(candidate)
            ordered_candidates.append(candidate)

        sidebar: List[Dict[str, float | str]] = []
        window_start = as_of - timedelta(days=self.lookback_window_days)

        for candidate_symbol in ordered_candidates:
            if len(sidebar) >= self.max_sidebar:
                break
            try:
                history = historical_fetcher.fetch(candidate_symbol, start=window_start, end=as_of)
            except Exception:
                continue
            if history.empty:
                continue
            history = history.sort_index()
            indicators = indicator_calculator.compute_all(history)

            rsi_current = self._safe_float_from_series(indicators.get("rsi_14"), history.index[-1])
            macd_current = self._safe_float_from_series(indicators.get("macd"), history.index[-1])
            sma_spread_current = None
            sma_20_series = indicators.get("sma_20")
            sma_50_series = indicators.get("sma_50")
            if sma_20_series is not None and sma_50_series is not None:
                sma_20_value = self._safe_float_from_series(sma_20_series, history.index[-1])
                sma_50_value = self._safe_float_from_series(sma_50_series, history.index[-1])
                if sma_20_value is not None and sma_50_value is not None:
                    sma_spread_current = sma_20_value - sma_50_value

            if "Volume" in history and len(history) >= 25:
                recent_volume = history["Volume"].tail(self.pre_breakout_window).mean()
                long_volume = history["Volume"].tail(20).mean()
                volume_ratio_current = (recent_volume / long_volume) if long_volume and long_volume > 0 else None
            else:
                volume_ratio_current = None

            confidence = self._score_pattern_alignment(
                pattern_summary=pattern_summary,
                rsi_current=rsi_current,
                macd_current=macd_current,
                sma_spread_current=sma_spread_current,
                volume_ratio_current=volume_ratio_current,
            )

            if confidence < self.min_pattern_confidence:
                continue

            latest_close = float(history["Close"].iloc[-1]) if "Close" in history else float(history.iloc[-1, 0])
            sidebar.append(
                {
                    "symbol": candidate_symbol,
                    "last_price": round(latest_close, 2),
                    "pattern_confidence": round(confidence, 2),
                    "pattern_alignment": self._pattern_alignment_description(pattern_summary),
                    "source": "Pattern Screening",
                }
            )

        return sidebar

    def _score_pattern_alignment(
        self,
        *,
        pattern_summary: Dict[str, float | str],
        rsi_current: Optional[float],
        macd_current: Optional[float],
        sma_spread_current: Optional[float],
        volume_ratio_current: Optional[float],
    ) -> float:
        score = 0.0
        max_score = 0.0

        pattern_rsi = self._safe_float(pattern_summary.get("pattern_avg_rsi"))
        if pattern_rsi is not None and rsi_current is not None and not math.isnan(rsi_current):
            max_score += 0.25
            if abs(rsi_current - pattern_rsi) <= 10:
                score += 0.25
            elif rsi_current > pattern_rsi and rsi_current <= 80:
                score += 0.2

        pattern_macd = self._safe_float(pattern_summary.get("pattern_avg_macd"))
        if pattern_macd is not None and macd_current is not None and not math.isnan(macd_current):
            max_score += 0.25
            if pattern_macd > 0 and macd_current > 0:
                score += 0.25
            elif pattern_macd <= 0 and macd_current <= 0:
                score += 0.15

        pattern_sma_spread = self._safe_float(pattern_summary.get("pattern_avg_sma_spread"))
        if pattern_sma_spread is not None and sma_spread_current is not None and not math.isnan(sma_spread_current):
            max_score += 0.25
            if pattern_sma_spread >= 0 and sma_spread_current >= 0:
                score += 0.25
            elif abs(sma_spread_current - pattern_sma_spread) <= abs(pattern_sma_spread) * 0.25 if pattern_sma_spread else 0.05:
                score += 0.18

        pattern_volume_ratio = self._safe_float(pattern_summary.get("pattern_avg_volume_ratio"))
        if pattern_volume_ratio is not None and volume_ratio_current is not None and not math.isnan(volume_ratio_current):
            max_score += 0.25
            if volume_ratio_current >= pattern_volume_ratio:
                score += 0.25
            elif volume_ratio_current >= max(pattern_volume_ratio - 0.3, 1.0):
                score += 0.18

        if max_score == 0:
            return 0.0

        normalised = score  # already scaled to <= 1.0 as cumulative weights total 1.0
        return max(0.0, min(0.95, normalised))

    @staticmethod
    def _pattern_alignment_description(pattern_summary: Dict[str, float | str]) -> str:
        pieces: List[str] = []
        rsi_value = pattern_summary.get("pattern_avg_rsi")
        if rsi_value is not None:
            pieces.append(f"RSI≈{float(rsi_value):.1f}")
        macd_value = pattern_summary.get("pattern_avg_macd")
        if macd_value is not None:
            pieces.append("MACD>0" if float(macd_value) > 0 else "MACD≈0")
        sma_spread = pattern_summary.get("pattern_avg_sma_spread")
        if sma_spread is not None:
            pieces.append("SMA20>SMA50" if float(sma_spread) > 0 else "SMA20≈SMA50")
        volume_ratio = pattern_summary.get("pattern_avg_volume_ratio")
        if volume_ratio is not None:
            pieces.append(f"Vol×{float(volume_ratio):.1f}")
        return ", ".join(pieces) if pieces else "Pattern alignment"

    @staticmethod
    def _safe_float_from_series(series: Optional[pd.Series], index) -> Optional[float]:
        if series is None:
            return None
        try:
            value = series.loc[index]
        except KeyError:
            return None
        return TopGainerAnalytics._safe_float(value)

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result

    def _alignment_score(
        self,
        price_history: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        factors: Dict[str, float | str],
    ) -> float:
        if price_history.empty:
            return 0.0

        macd = indicators.get("macd")
        rsi = indicators.get("rsi_14")

        close_series = price_history["Close"] if "Close" in price_history else price_history.iloc[:, 0]
        recent_returns = close_series.pct_change().tail(5)
        avg_recent_return = float(recent_returns.mean() * 100) if not recent_returns.empty else 0.0

        bias = 0.0
        if macd is not None and macd.iloc[-1] > 0:
            bias += 0.05
        if rsi is not None and 55 <= rsi.iloc[-1] <= 70:
            bias += 0.03
        if rsi is not None and rsi.iloc[-1] > 75:
            bias -= 0.02

        if "avg_percent_change" in factors and avg_recent_return > float(factors["avg_percent_change"]) * 0.5:
            bias += 0.04
        if "avg_relative_volume" in factors and "Volume" in price_history:
            recent_volume = price_history["Volume"].tail(5).mean()
            long_volume = price_history["Volume"].tail(30).mean()
            if long_volume and long_volume > 0:
                relative_volume = float(recent_volume / long_volume)
                benchmark = float(factors.get("avg_relative_volume", 1.0))
                if relative_volume > benchmark:
                    bias += 0.04

        return max(min(bias, 0.15), -0.1)
