"""BoT-SORT tracking runner.

Usage::

    python runner.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
import types
from pathlib import Path
from typing import Dict, Iterator, Tuple

import cv2
import numpy as np
import yaml

from detectors import build_detector
from recorder import RunRecorder
from reid import build_reid
from tracker.mc_bot_sort import BoTSORT

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ------------------------------------------------------------------
# Config / data helpers
# ------------------------------------------------------------------

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_tracker_args(tracker_cfg: Dict, reid_enabled: bool) -> types.SimpleNamespace:
    """Build the attribute-access object that ``BoTSORT.__init__`` expects.

    ``with_reid`` is derived from ``reid.enabled``, not from the tracker
    section, so there is a single source of truth.
    """
    return types.SimpleNamespace(
        track_high_thresh=tracker_cfg.get("track_high_thresh", 0.6),
        track_low_thresh=tracker_cfg.get("track_low_thresh", 0.1),
        new_track_thresh=tracker_cfg.get("new_track_thresh", 0.7),
        track_buffer=tracker_cfg.get("track_buffer", 30),
        match_thresh=tracker_cfg.get("match_thresh", 0.8),
        proximity_thresh=tracker_cfg.get("proximity_thresh", 0.5),
        appearance_thresh=tracker_cfg.get("appearance_thresh", 0.25),
        with_reid=reid_enabled,
    )


def iter_frames(data_cfg: Dict) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(frame_id, bgr_frame)`` from a video file or image directory."""
    source_type = data_cfg["source_type"]
    path = Path(data_cfg["path"])

    if source_type == "video":
        yield from _iter_video(path)
    elif source_type == "image_dir":
        yield from _iter_image_dir(path)
    else:
        raise ValueError(f"Unknown source_type '{source_type}'. Use 'video' or 'image_dir'.")


def _iter_video(video_path: Path) -> Iterator[Tuple[int, np.ndarray]]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            yield frame_id, frame
    finally:
        cap.release()

    logger.info("Video exhausted after %d frames", frame_id)


def _iter_image_dir(dir_path: Path) -> Iterator[Tuple[int, np.ndarray]]:
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {dir_path}")

    image_files = sorted(
        p for p in dir_path.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise FileNotFoundError(f"No images found in {dir_path}")

    logger.info("Found %d images in %s", len(image_files), dir_path)

    for frame_id, img_path in enumerate(image_files, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Skipping unreadable image: %s", img_path)
            continue
        yield frame_id, frame


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run(config: Dict) -> None:
    output_dir = Path(config["output"]["dir"]) / config["output"]["experiment_name"]
    min_box_area = config["tracker"].get("min_box_area", 10.0)

    output_cfg = config["output"]
    save_video = output_cfg.get("save_video", False)
    video_fps = output_cfg.get("video_fps", 30.0)

    recorder = RunRecorder(output_dir, save_video=save_video, video_fps=video_fps)
    recorder.save_config(config)

    detector = build_detector(config["detector"])
    reid_model = build_reid(config.get("reid", {}))
    reid_enabled = reid_model is not None
    tracker = BoTSORT(make_tracker_args(config["tracker"], reid_enabled))

    t_start = time.time()
    frame_count = 0

    for frame_id, frame_bgr in iter_frames(config["data"]):
        frame_count += 1

        detections = detector.detect(frame_bgr)

        if reid_model is not None and len(detections) > 0:
            features = reid_model.extract(frame_bgr, detections[:, :4])
            detections = np.concatenate(
                [detections, features.astype(np.float32)], axis=1,
            )

        online_targets = tracker.update(detections)

        recorder.add_detections(frame_id, detections)
        recorder.add_tracks(frame_id, online_targets, min_box_area)
        recorder.write_video_frame(frame_bgr, online_targets, frame_id, min_box_area)

        if frame_count % 20 == 0:
            elapsed = time.time() - t_start
            fps = frame_count / max(elapsed, 1e-6)
            logger.info(
                "Frame %d | %d dets | %d tracks | %.1f fps",
                frame_id, len(detections), len(online_targets), fps,
            )

    elapsed = time.time() - t_start
    logger.info(
        "Done. %d frames in %.1fs (%.1f fps)",
        frame_count, elapsed, frame_count / max(elapsed, 1e-6),
    )

    recorder.save()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BoT-SORT tracking runner")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
