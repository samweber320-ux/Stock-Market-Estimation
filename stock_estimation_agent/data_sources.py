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


class TopGainersFetcher(Protocol):
    """Interface describing an object capable of retrieving recent top market gainers."""

    verified_sources: List[VerifiedSource]

    def fetch(self, *, as_of: datetime, lookback_days: int) -> List["TopGainer"]:
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


@dataclass
class TopGainer:
    """Snapshot of a recent top gaining equity sourced from a brokerage screener."""

    symbol: str
    name: str
    last_price: float
    percent_change: float
    volume: Optional[int] = None
    average_volume: Optional[int] = None
    market_cap: Optional[float] = None
    source: str = ""
    sector: Optional[str] = None


class BrokerageTopGainersFetcher:
    """Aggregates top gainer lists from Webull and Robinhood public endpoints."""

    verified_sources = [
        VerifiedSource(
            name="Webull Top Gainers",
            url="https://www.webull.com/quote/us/top-gainers",
            description="Webull brokerage screener highlighting daily top performing U.S. equities.",
        ),
        VerifiedSource(
            name="Robinhood Movers",
            url="https://robinhood.com/collections/top-movers",
            description="Robinhood retail brokerage data of the largest percentage risers across major indices.",
        ),
    ]

    def __init__(self, *, session=None) -> None:
        self._session = session

    def fetch(self, *, as_of: datetime, lookback_days: int) -> List[TopGainer]:
        import requests

        gainers: List[TopGainer] = []

        gainers.extend(self._fetch_webull(requests))
        gainers.extend(self._fetch_robinhood(requests))

        unique: dict[str, TopGainer] = {}
        for entry in gainers:
            if entry.symbol not in unique or unique[entry.symbol].percent_change < entry.percent_change:
                unique[entry.symbol] = entry

        return list(unique.values())

    def _fetch_webull(self, requests_module) -> List[TopGainer]:
        url = "https://quotes-gw.webullfintech.com/api/information/brief/market/rank"
        params = {
            "tickerType": 1,
            "rankType": 5,
            "pageIndex": 1,
            "pageSize": 50,
        }
        try:
            response = (self._session or requests_module).get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception:
            return []

        items = data.get("data") or data.get("rankList") or []
        gainers: List[TopGainer] = []
        for item in items:
            try:
                gainers.append(
                    TopGainer(
                        symbol=item.get("tickerSymbol") or item.get("symbol", ""),
                        name=item.get("tickerName") or item.get("name", ""),
                        last_price=float(item.get("latestPrice") or item.get("close", 0.0)),
                        percent_change=float(item.get("changeRate") or item.get("percent", 0.0)) * 100
                        if abs(float(item.get("changeRate") or 0.0)) < 5
                        else float(item.get("changeRate") or item.get("percent", 0.0)),
                        volume=self._safe_int(item.get("volume")),
                        average_volume=self._safe_int(item.get("avgVolume")),
                        market_cap=self._safe_float(item.get("marketValue")),
                        source="Webull",
                        sector=item.get("industry", ""),
                    )
                )
            except Exception:
                continue
        return gainers

    def _fetch_robinhood(self, requests_module) -> List[TopGainer]:
        url = "https://api.robinhood.com/midlands/movers/sp500/"
        params = {"direction": "up"}
        try:
            response = (self._session or requests_module).get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []

        instruments = payload if isinstance(payload, list) else payload.get("results", [])
        gainers: List[TopGainer] = []
        for item in instruments:
            try:
                movement = item.get("price_movement", {})
                gainers.append(
                    TopGainer(
                        symbol=item.get("symbol", ""),
                        name=item.get("name", ""),
                        last_price=float(movement.get("market_price") or movement.get("price", 0.0)),
                        percent_change=float(movement.get("percent_change") or 0.0) * 100,
                        volume=None,
                        average_volume=None,
                        market_cap=None,
                        source="Robinhood",
                        sector=item.get("sector", ""),
                    )
                )
            except Exception:
                continue
        return gainers

    @staticmethod
    def _safe_int(value) -> Optional[int]:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class StaticTopGainersFetcher:
    """In-memory top gainer feed useful for testing or offline usage."""

    def __init__(self, gainers: Optional[List[TopGainer]] = None) -> None:
        self._gainers = gainers or []
        self.verified_sources: List[VerifiedSource] = []

    def fetch(self, *, as_of: datetime, lookback_days: int) -> List[TopGainer]:
        return list(self._gainers)
