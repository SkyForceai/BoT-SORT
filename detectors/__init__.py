from __future__ import annotations

from typing import Dict

from detectors.base import BaseDetector


def build_detector(config: Dict) -> BaseDetector:
    """Instantiate a detector from a config dict.

    The ``type`` key selects the concrete class.  All remaining keys are
    forwarded as constructor kwargs.
    """
    config = config.copy()
    detector_type = config.pop("type")

    if detector_type == "dfine":
        from detectors.dfine import DFINEDetector

        return DFINEDetector(
            backend=config.get("backend", "onnx"),
            model_path=config.get("model_path"),
            config_path=config.get("config_path"),
            target_size=tuple(config.get("target_size", [720, 1280])),
            num_classes=config.get("num_classes", 80),
            conf_threshold=config.get("conf_threshold", 0.3),
            num_top_queries=config.get("num_top_queries", 300),
            device=config.get("device", "cuda:0"),
            fp16=config.get("fp16", False),
            use_ema=config.get("use_ema", True),
        )

    raise ValueError(
        f"Unknown detector type '{detector_type}'. Available: dfine"
    )
