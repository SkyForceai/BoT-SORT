from __future__ import annotations

from typing import Dict, Optional

from reid.base import BaseReID


def build_reid(config: Dict) -> Optional[BaseReID]:
    """Instantiate a ReID extractor from the ``reid`` config section.

    Returns ``None`` when ReID is disabled.
    """
    if not config.get("enabled", False):
        return None

    reid_type = config.get("type", "fast_reid")

    if reid_type == "fast_reid":
        from reid.fast_reid_wrapper import FastReIDExtractor

        return FastReIDExtractor(
            config_path=config["config_path"],
            weights_path=config["weights_path"],
            device=config.get("device", "cuda"),
        )

    raise ValueError(f"Unknown reid type '{reid_type}'. Available: fast_reid")
