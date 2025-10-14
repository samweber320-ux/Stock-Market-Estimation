"""Top-level package for the stock estimation agent."""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["StockEstimationAgent"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module("stock_estimation_agent.agent")
        return getattr(module, name)
    raise AttributeError(f"module 'stock_estimation_agent' has no attribute {name!r}")
