"""Shared source descriptors used across the agent."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VerifiedSource:
    """Descriptor of a source used in the estimation process."""

    name: str
    url: str
    description: str
