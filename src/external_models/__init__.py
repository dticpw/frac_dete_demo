"""External model adapter interfaces and registry."""

from .base import ExternalCandidate, ExternalModelAdapter
from .registry import get_enabled_adapters, run_enabled_adapters

__all__ = [
    "ExternalCandidate",
    "ExternalModelAdapter",
    "get_enabled_adapters",
    "run_enabled_adapters",
]
