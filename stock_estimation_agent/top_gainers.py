"""Utilities for analysing top gainer lists from retail brokerages."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .data_sources import TopGainer, VerifiedSource


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

    verified_sources: List[VerifiedSource] = field(
        default_factory=lambda: [
            VerifiedSource(
                name="Investopedia Technical Analysis",
                url="https://www.investopedia.com/terms/t/technicalanalysis.asp",
                description="Best practices on leveraging momentum, volume, and sector leadership signals.",
            )
        ]
    )

    def build_context(
        self,
        *,
        symbol: str,
        price_history: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        top_gainers: Sequence[TopGainer],
    ) -> TopGainerContext:
        if not top_gainers:
            return TopGainerContext(factors={}, narrative="", sidebar=[], alignment_bias=0.0)

        factors = self._identify_common_factors(top_gainers)
        narrative = self._compose_narrative(factors, top_gainers)
        sidebar = self._build_sidebar(top_gainers, exclude_symbol=symbol)
        alignment_bias = self._alignment_score(price_history, indicators, factors)

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
