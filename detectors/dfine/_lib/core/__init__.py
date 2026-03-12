"""
Core building blocks for config-driven training.

Note: Keep this module import-light so utilities like `core.yaml_utils.load_config`
can be used without requiring heavy deps (e.g. `torch`) to be installed.
"""

from .workspace import GLOBAL_CONFIG, create, register
from .yaml_utils import *  # noqa: F403

# Heavy imports (torch) are optional at import time.
try:  # pragma: no cover
    from ._config import BaseConfig
    from .yaml_config import YAMLConfig
except ModuleNotFoundError:
    BaseConfig = None  # type: ignore[assignment]
    YAMLConfig = None  # type: ignore[assignment]
