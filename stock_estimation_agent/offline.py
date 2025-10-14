"""Helpers to run the agent without external network calls."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Sequence

from .sources import VerifiedSource


@dataclass
class LocalNewsFetcher:
    """News fetcher that reads from a JSON file created by the user."""

    path: Path

    verified_sources = [
        VerifiedSource(
            name="User supplied news",
            url="file://local-news",
            description="News headlines uploaded by the user for offline estimation.",
        )
    ]

    def fetch(self, symbol: str, *, as_of: datetime, window_days: int) -> List[dict]:
        import pandas as pd  # Imported lazily to keep optional dependency lightweight.

        data = json.loads(self.path.read_text())
        articles: List[dict] = []
        lower_bound = as_of - timedelta(days=window_days)
        for article in data:
            published = pd.to_datetime(article.get("publishedAt"))
            if published is pd.NaT:
                continue
            if lower_bound <= published <= as_of and symbol.upper() in article.get("symbols", [symbol.upper()]):
                articles.append(article)
        return articles

    def available_symbols(self) -> Iterable[str]:  # pragma: no cover - static dataset
        return []


@dataclass(frozen=True)
class OfflineNewsSource:
    """Metadata describing a verified article used in offline recommendations."""

    title: str
    url: str
    publisher: str
    published_at: datetime


@dataclass(frozen=True)
class OfflineRecommendation:
    """Pre-computed recommendation built from previous trading-day analytics."""

    symbol: str
    name: str
    market_date: datetime
    close: float
    pattern_confidence: float
    potential_upside_pct: float
    risk_level: str
    indicator_alignment: Sequence[str]
    historical_response: str
    narrative: str
    news_sources: Sequence[OfflineNewsSource]

    verified_sources = [
        VerifiedSource(
            name="Offline research snapshot",
            url="file://offline-recommendations",
            description="Curated recommendations generated from the agent using the prior U.S. trading session.",
        )
    ]


def _parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:  # pragma: no cover - defensive fallback
        raise ValueError(f"Invalid datetime string: {value}") from exc


def load_offline_recommendations(path: Path) -> List[OfflineRecommendation]:
    """Load a set of offline recommendations from ``path``.

    Supply a JSON payload generated from your most recent run of the agent so the
    embedded ``market_date`` reflects the last completed U.S. trading session.
    Each entry in the ``universe`` list describes a recommended symbol. News
    metadata is preserved so consumers can surface the verified sources alongside
    the technical summary.
    """

    payload = json.loads(path.read_text())
    default_market_date = payload.get("market_date")
    if not default_market_date:
        raise ValueError("offline recommendation payload missing 'market_date'")

    market_date = datetime.strptime(default_market_date, "%Y-%m-%d")
    recommendations: List[OfflineRecommendation] = []
    for entry in payload.get("universe", []):
        entry_market_date = entry.get("market_date")
        resolved_market_date = (
            datetime.strptime(entry_market_date, "%Y-%m-%d")
            if entry_market_date
            else market_date
        )
        news_sources = [
            OfflineNewsSource(
                title=article["title"],
                url=article["url"],
                publisher=article["publisher"],
                published_at=_parse_datetime(article["publishedAt"]),
            )
            for article in entry.get("news_sources", [])
        ]
        recommendations.append(
            OfflineRecommendation(
                symbol=entry["symbol"],
                name=entry.get("name", entry["symbol"]),
                market_date=resolved_market_date,
                close=float(entry["close"]),
                pattern_confidence=float(entry["pattern_confidence"]),
                potential_upside_pct=float(entry["potential_upside_pct"]),
                risk_level=entry.get("risk_level", "Unknown"),
                indicator_alignment=tuple(entry.get("indicator_alignment", [])),
                historical_response=entry.get("historical_response", ""),
                narrative=entry.get("narrative", ""),
                news_sources=news_sources,
            )
        )
    return recommendations
