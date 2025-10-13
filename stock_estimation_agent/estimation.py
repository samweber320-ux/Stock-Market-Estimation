"""Price estimation logic that combines quantitative and qualitative signals."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import pandas as pd

from .data_sources import VerifiedSource
from .top_gainers import TopGainerContext


@dataclass
class EstimationResult:
    """Structured response returned by the estimation engine."""

    symbol: str
    generated_at: datetime
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
    """Combines indicators, historical trends, and news to estimate prices."""

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
    ) -> EstimationResult:
        latest_close = float(price_history["Close"].iloc[-1]) if "Close" in price_history else float(price_history.iloc[-1, 0])

        macd = indicators.get("macd")
        rsi = indicators.get("rsi_14")
        sma_20 = indicators.get("sma_20")
        sma_50 = indicators.get("sma_50")
        bollinger_upper = indicators.get("bollinger_upper")
        bollinger_lower = indicators.get("bollinger_lower")

        trend_bias = 0.0
        if sma_20 is not None and sma_50 is not None:
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                trend_bias += 0.1
            else:
                trend_bias -= 0.05

        if macd is not None:
            trend_bias += float(macd.iloc[-1]) / max(abs(macd).max(), 1e-9)

        if rsi is not None:
            latest_rsi = float(rsi.iloc[-1])
            if latest_rsi > 70:
                trend_bias -= 0.1
            elif latest_rsi < 30:
                trend_bias += 0.1

        volatility_penalty = 0.0
        if bollinger_upper is not None and bollinger_lower is not None:
            band_width = float(bollinger_upper.iloc[-1] - bollinger_lower.iloc[-1])
            if band_width / latest_close > 0.1:
                volatility_penalty -= 0.05

        qualitative_adjustment = self._news_sentiment_adjustment(news)
        gainer_bias = gainer_context.alignment_bias

        confidence = max(
            min(0.5 + trend_bias + qualitative_adjustment + volatility_penalty + gainer_bias, 0.95),
            0.05,
        )
        target_price = latest_close * (1 + trend_bias + qualitative_adjustment + volatility_penalty + gainer_bias)

        narrative_parts = [
            f"Latest close for {symbol} is ${latest_close:,.2f}.",
        ]
        if trend_bias > 0:
            narrative_parts.append("Short-term momentum and trend indicators skew positive.")
        elif trend_bias < 0:
            narrative_parts.append("Trend indicators point to potential weakness relative to longer-term averages.")

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
            "sma_20": float(sma_20.iloc[-1]) if sma_20 is not None else float("nan"),
            "sma_50": float(sma_50.iloc[-1]) if sma_50 is not None else float("nan"),
            "macd": float(macd.iloc[-1]) if macd is not None else float("nan"),
            "rsi_14": float(rsi.iloc[-1]) if rsi is not None else float("nan"),
            "bollinger_upper": float(bollinger_upper.iloc[-1]) if bollinger_upper is not None else float("nan"),
            "bollinger_lower": float(bollinger_lower.iloc[-1]) if bollinger_lower is not None else float("nan"),
            "top_gainer_alignment_bias": gainer_bias,
        }

        news_headlines = [article.get("title", "") for article in news][:5]

        return EstimationResult(
            symbol=symbol,
            generated_at=datetime.utcnow(),
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
