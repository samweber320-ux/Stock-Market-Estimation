"""Offline recommendations compiled from verified research snapshots."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .sources import VerifiedSource


@dataclass
class OfflineRecommendation:
    """Pre-computed recommendation that mirrors the agent output structure."""

    symbol: str
    company: str
    pattern_confidence: float
    thesis: str
    technical_snapshot: List[str]
    fundamental_snapshot: List[str]
    news_snapshot: List[str]
    sources: List[VerifiedSource]

    def to_dict(self) -> dict:
        """Serialize the recommendation for downstream consumption."""

        return {
            "symbol": self.symbol,
            "company": self.company,
            "pattern_confidence": self.pattern_confidence,
            "thesis": self.thesis,
            "technical_snapshot": list(self.technical_snapshot),
            "fundamental_snapshot": list(self.fundamental_snapshot),
            "news_snapshot": list(self.news_snapshot),
            "sources": [source.__dict__ for source in self.sources],
        }


def _default_path() -> Path:
    return Path(__file__).resolve().parent.parent / "research" / "offline_recommendations.json"


def load_offline_recommendations(path: Path | None = None) -> List[OfflineRecommendation]:
    """Load curated offline recommendations from disk.

    Parameters
    ----------
    path:
        Optional override path. When omitted the loader falls back to the
        repository-provided snapshot stored under ``research/offline_recommendations.json``.
    """

    payload_path = path or _default_path()
    data = json.loads(payload_path.read_text())
    recommendations: List[OfflineRecommendation] = []

    for entry in data:
        sources = [
            VerifiedSource(
                name=item.get("name", ""),
                url=item.get("url", ""),
                description=item.get("description", ""),
            )
            for item in entry.get("sources", [])
        ]
        recommendations.append(
            OfflineRecommendation(
                symbol=entry.get("symbol", ""),
                company=entry.get("company", ""),
                pattern_confidence=float(entry.get("pattern_confidence", 0.0)),
                thesis=entry.get("thesis", ""),
                technical_snapshot=list(entry.get("technical_snapshot", [])),
                fundamental_snapshot=list(entry.get("fundamental_snapshot", [])),
                news_snapshot=list(entry.get("news_snapshot", [])),
                sources=sources,
            )
        )

    return recommendations


def offline_symbols(path: Path | None = None) -> Iterable[str]:
    """Convenience helper returning the symbols included in the offline snapshot."""

    for recommendation in load_offline_recommendations(path):
        yield recommendation.symbol
