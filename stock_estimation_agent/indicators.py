"""Technical indicator calculations used by the estimation agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from .sources import VerifiedSource


@dataclass
class IndicatorCalculator:
    """Computes a suite of technical indicators using pandas/numpy."""

    verified_sources: List[VerifiedSource] = field(
        default_factory=lambda: [
            VerifiedSource(
                name="Investopedia",
                url="https://www.investopedia.com/terms/t/technicalindicator.asp",
                description="Definitions and methodologies for widely used technical indicators.",
            ),
            VerifiedSource(
                name="CMT Association",
                url="https://cmtassociation.org/",
                description="Professional body certifying Chartered Market Technicians and providing methodology guidance.",
            ),
        ]
    )

    def compute_all(self, history: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute a curated set of indicators required for the estimation engine."""

        if history.empty:
            raise ValueError("Cannot compute indicators on empty history.")

        closes = history["Close"] if "Close" in history else history.iloc[:, 0]

        indicators: Dict[str, pd.Series] = {
            "sma_20": closes.rolling(window=20).mean(),
            "sma_50": closes.rolling(window=50).mean(),
            "ema_12": closes.ewm(span=12, adjust=False).mean(),
            "ema_26": closes.ewm(span=26, adjust=False).mean(),
        }

        indicators["macd"] = indicators["ema_12"] - indicators["ema_26"]
        signal = indicators["macd"].ewm(span=9, adjust=False).mean()
        indicators["macd_signal"] = signal
        indicators["macd_histogram"] = indicators["macd"] - signal

        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        indicators["rsi_14"] = 100 - (100 / (1 + rs))

        rolling_std = closes.rolling(window=20).std()
        indicators["bollinger_mid"] = indicators["sma_20"]
        indicators["bollinger_upper"] = indicators["sma_20"] + (rolling_std * 2)
        indicators["bollinger_lower"] = indicators["sma_20"] - (rolling_std * 2)

        if "Volume" in history:
            indicators["on_balance_volume"] = self._compute_obv(closes, history["Volume"])

        return indicators

    @staticmethod
    def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=close.index)
