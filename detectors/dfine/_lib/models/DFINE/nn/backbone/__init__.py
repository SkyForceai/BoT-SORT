"""
DFINE backbone module (inference-only subset).

Only HGNetv2 is included; other backbones are stripped to keep the
local copy minimal.
"""

from .common import (
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
    get_activation,
)
from .hgnetv2 import DFINEHGNetv2
