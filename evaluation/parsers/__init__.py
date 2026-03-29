"""Parser registry for reading prediction and ground-truth files.

Parsers convert raw files into the canonical :class:`SequenceData`
representation.  Register new formats with :func:`register_parser`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Type

from evaluation.schema import AnnotationMask, SequenceData


class ParserBase(ABC):
    """Convert a raw file (or directory) into canonical evaluation types."""

    @abstractmethod
    def parse_predictions(self, path: Path) -> SequenceData:
        """Read tracker output and return a :class:`SequenceData`."""
        ...

    @abstractmethod
    def parse_ground_truth(
        self,
        path: Path,
    ) -> Tuple[SequenceData, AnnotationMask]:
        """Read ground-truth annotations.

        Returns both the data and an :class:`AnnotationMask` that
        indicates which frames are annotated.  The default strategy is
        to treat every frame present in the file as annotated, but
        parsers may override this (e.g. reading a sidecar mask file).
        """
        ...


_PARSER_REGISTRY: Dict[str, Type[ParserBase]] = {}


def register_parser(name: str):
    """Class decorator that registers a parser under *name*."""

    def decorator(cls: Type[ParserBase]) -> Type[ParserBase]:
        if name in _PARSER_REGISTRY:
            raise ValueError(f"Parser '{name}' already registered")
        _PARSER_REGISTRY[name] = cls
        return cls

    return decorator


def build_parser(name: str, **kwargs) -> ParserBase:
    """Instantiate a registered parser by name."""
    if name not in _PARSER_REGISTRY:
        available = list(_PARSER_REGISTRY.keys())
        raise KeyError(f"Unknown parser: '{name}'. Available: {available}")
    return _PARSER_REGISTRY[name](**kwargs)


def _discover_parsers() -> None:
    """Import all parser modules so their ``@register_parser`` decorators run."""
    import importlib
    import pkgutil

    package = importlib.import_module(__name__)
    for _importer, modname, _ispkg in pkgutil.iter_modules(package.__path__):
        if modname == "base":
            continue
        importlib.import_module(f"{__name__}.{modname}")


_discover_parsers()
