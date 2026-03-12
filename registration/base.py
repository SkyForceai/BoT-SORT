from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseRegistration(ABC):
    """Abstract base class for track-to-identity registration."""

    @abstractmethod
    def register(self, track_id: int, features: Any) -> None:
        """Register or update an identity for *track_id*."""
        ...

    @abstractmethod
    def query(self, features: Any) -> List[Dict[str, Any]]:
        """Query the gallery and return ranked matches."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear the gallery."""
        ...
