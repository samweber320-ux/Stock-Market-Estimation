"""Registry for user provided datasets to augment the agent's knowledge."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import pandas as pd


@dataclass
class UserDatasetRegistry:
    """Stores datasets indexed by symbol and merges them with market history."""

    _datasets: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def register(self, symbol: str, dataset: pd.DataFrame) -> None:
        if dataset.empty:
            raise ValueError("User dataset cannot be empty.")
        if not isinstance(dataset.index, pd.DatetimeIndex):
            raise TypeError("Dataset index must be a pandas.DatetimeIndex.")
        self._datasets[symbol] = dataset.sort_index()

    def merge(self, symbol: str, history: pd.DataFrame) -> pd.DataFrame:
        user_dataset = self._datasets.get(symbol)
        if user_dataset is None:
            return history
        return history.join(user_dataset, how="outer").sort_index().fillna(method="ffill")

    def available_symbols(self) -> Iterable[str]:
        return self._datasets.keys()
