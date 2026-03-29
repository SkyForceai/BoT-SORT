from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from registration.base import BaseRegistration


class IdentityRegistration(BaseRegistration):
    """No-op registration — always returns the identity transform."""

    def apply(self, raw_frame: np.ndarray, detections=None) -> np.ndarray:
        return np.eye(2, 3)


def build_registration(config: Dict[str, Any]) -> Optional[BaseRegistration]:
    """Instantiate a registration module from the ``registration`` config section.

    Returns ``None`` when registration is disabled.
    """
    if not config.get("enabled", False):
        return None

    method = config.get("type", "sparseOptFlow")
    downscale = config.get("downscale", 2)

    if method in ("none", "None"):
        return IdentityRegistration()

    if method == "sparseOptFlow":
        from registration.sparse_optflow import SparseOptFlowRegistration

        return SparseOptFlowRegistration(downscale=downscale)

    if method == "ecc":
        from registration.ecc import ECCRegistration

        return ECCRegistration(
            downscale=downscale,
            num_iterations=config.get("num_iterations", 5000),
            eps=config.get("eps", 1e-6),
        )

    if method in ("orb", "sift"):
        from registration.feature import FeatureRegistration

        return FeatureRegistration(method=method, downscale=downscale)

    if method in ("file", "files"):
        from registration.file_based import FileRegistration

        return FileRegistration(file_path=config["file_path"])

    raise ValueError(
        f"Unknown registration type '{method}'. "
        f"Available: sparseOptFlow, ecc, orb, sift, file, none"
    )
