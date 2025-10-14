from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from .sources import VerifiedSource
from .top_gainers import TopGainerContext


@dataclass
class EstimationResult:
    """Structured response returned by the estimation engine."""

    symbol: str
    generated_at: datetime
    market_date: datetime
    reference_close: float
    narrative: str
    target_price: float
    confidence: float
    supporting_indicators: Dict[str, float]
    news_headlines: List[str]
    sources: List[VerifiedSource]
    top_gainer_factors: Dict[str, float | str]
    top_gainer_narrative: str
    potential_top_gainers: List[Dict[str, float | str]]


@dataclass
class EstimationEngine:
    """Combines indicators, historical responses, and news to estimate prices."""

    verified_sources: List[VerifiedSource] = field(
        default_factory=lambda: [
            VerifiedSource(
                name="Chartered Financial Analyst Institute",
                url="https://www.cfainstitute.org/",
                description="Practitioner guidance on combining fundamental and technical inputs for investment decisions.",
            ),
            VerifiedSource(
                name="Federal Reserve Economic Data (FRED)",
                url="https://fred.stlouisfed.org/",
                description="Macroeconomic datasets to contextualize sector performance and risk factors.",
            ),
        ]
    )

    def estimate(
        self,
        *,
        symbol: str,
        price_history: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        news: List[dict],
        sources: List[VerifiedSource],
        gainer_context: TopGainerContext,
        market_date: datetime,
    ) -> EstimationResult:
        market_ts = pd.Timestamp(market_date).normalize()
        closes_series = price_history["Close"] if "Close" in price_history else price_history.iloc[:, 0]
        closes_up_to_market = closes_series.loc[:market_ts]
        if closes_up_to_market.empty:
            raise ValueError("No pricing data available for the requested market date.")

        latest_close = float(closes_up_to_market.iloc[-1])

        macd = self._latest_indicator(indicators.get("macd"), market_ts)
        rsi = self._latest_indicator(indicators.get("rsi_14"), market_ts)
        sma_20 = self._latest_indicator(indicators.get("sma_20"), market_ts)
        sma_50 = self._latest_indicator(indicators.get("sma_50"), market_ts)
        bollinger_upper = self._latest_indicator(indicators.get("bollinger_upper"), market_ts)
        bollinger_lower = self._latest_indicator(indicators.get("bollinger_lower"), market_ts)

        trend_bias, match_count = self._historical_response_bias(
            closes_series=closes_series,
            indicators=indicators,
            market_ts=market_ts,
        )

        volatility_penalty = 0.0
        if bollinger_upper is not None and bollinger_lower is not None:
            band_width = float(bollinger_upper - bollinger_lower)
            if band_width / latest_close > 0.1:
                volatility_penalty -= 0.05

        qualitative_adjustment = self._news_sentiment_adjustment(news)
        gainer_bias = gainer_context.alignment_bias

        base_confidence = 0.4
        if match_count >= 5:
            base_confidence += 0.15
        elif match_count >= 3:
            base_confidence += 0.08
        elif match_count == 0:
            base_confidence -= 0.05

        combined_bias = trend_bias + qualitative_adjustment + gainer_bias + volatility_penalty
        confidence = max(min(base_confidence + combined_bias, 0.95), 0.05)
        target_price = latest_close * (1 + combined_bias)

        narrative_parts = [
            f"Analysis anchored to the {market_ts.date()} close of ${latest_close:,.2f} for {symbol}.",
        ]
        if match_count > 0:
            narrative_parts.append(
                f"{match_count} historical periods with similar indicator readings averaged a {trend_bias * 100:.2f}% next-session move."
            )
        else:
            narrative_parts.append("No closely matching historical trend responses were found; directional bias stays muted.")

        if qualitative_adjustment > 0:
            narrative_parts.append("Recent news flow is net positive, supporting bullish sentiment.")
        elif qualitative_adjustment < 0:
            narrative_parts.append("Recent headlines lean negative, applying caution to the outlook.")

        if volatility_penalty < 0:
            narrative_parts.append("Elevated Bollinger band width suggests heightened volatility; confidence trimmed accordingly.")

        if gainer_bias > 0 and gainer_context.narrative:
            narrative_parts.append(
                f"Top gainer dynamics align with the current setup, adding conviction. {gainer_context.narrative}"
            )
        elif gainer_context.narrative:
            narrative_parts.append(gainer_context.narrative)

        supporting_indicators = {
            "sma_20": float(sma_20) if sma_20 is not None else float("nan"),
            "sma_50": float(sma_50) if sma_50 is not None else float("nan"),
            "macd": float(macd) if macd is not None else float("nan"),
            "rsi_14": float(rsi) if rsi is not None else float("nan"),
            "bollinger_upper": float(bollinger_upper) if bollinger_upper is not None else float("nan"),
            "bollinger_lower": float(bollinger_lower) if bollinger_lower is not None else float("nan"),
            "historical_response_mean": float(trend_bias),
            "historical_match_count": float(match_count),
            "top_gainer_alignment_bias": gainer_bias,
        }

        news_headlines = [article.get("title", "") for article in news][:5]

        return EstimationResult(
            symbol=symbol,
            generated_at=datetime.utcnow(),
            market_date=market_ts.to_pydatetime(),
            reference_close=latest_close,
            narrative=" ".join(narrative_parts),
            target_price=target_price,
            confidence=confidence,
            supporting_indicators=supporting_indicators,
            news_headlines=news_headlines,
            sources=sources,
            top_gainer_factors=gainer_context.factors,
            top_gainer_narrative=gainer_context.narrative,
            potential_top_gainers=gainer_context.sidebar,
        )

    @staticmethod
    def _news_sentiment_adjustment(news: List[dict]) -> float:
        if not news:
            return 0.0

        score = 0.0
        for article in news:
            sentiment = article.get("sentiment")
            if sentiment == "positive":
                score += 0.05
            elif sentiment == "negative":
                score -= 0.05

        return max(min(score, 0.15), -0.15)

    @staticmethod
    def _latest_indicator(series: pd.Series | None, market_ts: pd.Timestamp) -> float | None:
        if series is None or series.empty:
            return None
        trimmed = series.loc[:market_ts].dropna()
        if trimmed.empty:
            return None
        return float(trimmed.iloc[-1])

    def _historical_response_bias(
        self,
        *,
        closes_series: pd.Series,
        indicators: Dict[str, pd.Series],
        market_ts: pd.Timestamp,
    ) -> Tuple[float, int]:
        closes_trimmed = closes_series.loc[:market_ts]
        if closes_trimmed.empty or len(closes_trimmed) < 2:
            return 0.0, 0

        indicator_names = ["macd", "rsi_14", "sma_20", "sma_50"]
        snapshot: Dict[str, float] = {}
        indicator_frames: Dict[str, pd.Series] = {}
        for name in indicator_names:
            series = indicators.get(name)
            if series is None:
                continue
            trimmed = series.loc[:market_ts].dropna()
            if trimmed.empty:
                continue
            snapshot[name] = float(trimmed.iloc[-1])
            indicator_frames[name] = trimmed

        if not snapshot:
            return 0.0, 0

        common_index = None
        for series in indicator_frames.values():
            common_index = series.index if common_index is None else common_index.intersection(series.index)
        if common_index is None:
            return 0.0, 0

        common_index = common_index[common_index < market_ts]
        if len(common_index) == 0:
            return 0.0, 0

        analog_returns: List[float] = []
        tolerance = 0.15
        closes_lookup = closes_trimmed
        for idx in common_index:
            try:
                current_close = float(closes_lookup.loc[idx])
            except KeyError:
                continue

            loc = closes_lookup.index.get_loc(idx)
            if isinstance(loc, slice):
                loc = loc.stop - 1
            if isinstance(loc, (list, tuple)):
                loc = loc[-1]
            next_position = loc + 1
            if next_position >= len(closes_lookup):
                continue

            next_close = float(closes_lookup.iloc[next_position])

            deltas: List[float] = []
            for name, value in snapshot.items():
                series = indicator_frames[name]
                try:
                    history_value = float(series.loc[idx])
                except KeyError:
                    deltas = []
                    break
                scale = max(abs(value), 1e-9)
                deltas.append(abs(history_value - value) / scale)

            if not deltas:
                continue

            if sum(deltas) / len(deltas) <= tolerance:
                analog_returns.append((next_close / current_close) - 1.0)

        if not analog_returns:
            return 0.0, 0

        return float(sum(analog_returns) / len(analog_returns)), len(analog_returns)
