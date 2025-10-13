"""Utilities for retrieving historical market data and news from verified sources."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Protocol

import pandas as pd


@dataclass
class VerifiedSource:
    """Descriptor of a source used in the estimation process."""

    name: str
    url: str
    description: str


class HistoricalDataFetcher(Protocol):
    """Interface describing an object capable of retrieving historical prices."""

    verified_sources: List[VerifiedSource]

    def fetch(self, symbol: str, *, start: datetime, end: datetime) -> pd.DataFrame:
        ...

    def available_symbols(self) -> Iterable[str]:
        ...


class MarketNewsFetcher(Protocol):
    """Interface describing an object capable of retrieving relevant news."""

    verified_sources: List[VerifiedSource]

    def fetch(self, symbol: str, *, as_of: datetime, window_days: int) -> List[dict]:
        ...


class YFinanceHistoricalFetcher:
    """Historical data fetcher that relies on Yahoo! Finance via ``yfinance``."""

    verified_sources = [
        VerifiedSource(
            name="Yahoo! Finance",
            url="https://finance.yahoo.com/",
            description=(
                "Market data provided by Yahoo! Finance. Data includes adjusted close, "
                "volume, and high/low/close/open fields, aggregated daily."
            ),
        )
    ]

    def __init__(self, *, session=None) -> None:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - informative error
            raise RuntimeError(
                "yfinance is required to use YFinanceHistoricalFetcher. Install via ``pip install yfinance``"
            ) from exc

        self._yf = yf
        self._session = session

    def fetch(self, symbol: str, *, start: datetime, end: datetime) -> pd.DataFrame:
        ticker = self._yf.Ticker(symbol, session=self._session)
        data = ticker.history(start=start, end=end, auto_adjust=False)
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError(f"No historical data returned for symbol {symbol}.")
        data.index = pd.to_datetime(data.index)
        data.sort_index(inplace=True)
        return data

    def available_symbols(self) -> Iterable[str]:
        return []


class NewsAPIFetcher:
    """News fetcher using the NewsAPI service."""

    verified_sources = [
        VerifiedSource(
            name="NewsAPI",
            url="https://newsapi.org/",
            description=(
                "Curated collection of reputable financial publications such as The Wall Street Journal, "
                "Bloomberg, CNBC, and Reuters provided via the NewsAPI aggregator."
            ),
        )
    ]

    def __init__(self, api_key: str, *, session=None) -> None:
        self._api_key = api_key
        self._session = session

    def fetch(self, symbol: str, *, as_of: datetime, window_days: int) -> List[dict]:
        import requests

        if window_days <= 0:
            raise ValueError("window_days must be positive")

        query_params = {
            "q": symbol,
            "from": (as_of - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d"),
            "to": as_of.strftime("%Y-%m-%d"),
            "language": "en",
            "pageSize": 25,
            "sortBy": "relevancy",
            "apiKey": self._api_key,
        }

        response = (self._session or requests).get("https://newsapi.org/v2/everything", params=query_params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return payload.get("articles", [])

    def available_symbols(self) -> Iterable[str]:  # pragma: no cover - no inherent state
        return []


class StaticHistoricalFetcher:
    """Historical data fetcher backed by an in-memory mapping.

    Useful for testing or when the user uploads their own datasets.
    """

    def __init__(self, datasets: Optional[dict[str, pd.DataFrame]] = None) -> None:
        self._datasets: dict[str, pd.DataFrame] = datasets or {}
        self.verified_sources: List[VerifiedSource] = []

    def fetch(self, symbol: str, *, start: datetime, end: datetime) -> pd.DataFrame:
        data = self._datasets.get(symbol)
        if data is None:
            raise ValueError(f"Symbol {symbol} not available in static historical fetcher.")
        mask = (data.index >= pd.Timestamp(start)) & (data.index <= pd.Timestamp(end))
        filtered = data.loc[mask].copy()
        if filtered.empty:
            raise ValueError(f"No data available for {symbol} in the requested range.")
        return filtered

    def available_symbols(self) -> Iterable[str]:
        return self._datasets.keys()
