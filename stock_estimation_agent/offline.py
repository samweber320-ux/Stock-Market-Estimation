"""Helpers to run the agent without external network calls."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .data_sources import MarketNewsFetcher
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
