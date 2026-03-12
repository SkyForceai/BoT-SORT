from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class BaseEvaluator(ABC):
    """Abstract base class for tracking evaluation."""

    @abstractmethod
    def evaluate(
        self,
        predictions_path: Path,
        ground_truth_path: Path,
    ) -> Dict[str, Any]:
        """Run evaluation and return a metrics dictionary."""
        ...

    @abstractmethod
    def summary(self, metrics: Dict[str, Any]) -> str:
        """Return a human-readable summary string."""
        ...
