"""Run recorder -- accumulates per-frame detections and tracks, writes CSVs.

Also owns Python logging setup, resolved-config persistence, and optional
video output with track visualisation.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

_PALETTE = (
    np.array(
        [
            [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
            [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
            [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
            [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
            [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255],
        ],
        dtype=np.int32,
    )
)


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    c = _PALETTE[track_id % len(_PALETTE)]
    return int(c[0]), int(c[1]), int(c[2])


# Per-frame track entries for drawing: x1,y1,x2,y2 (pixels), track_id, class_id.
_TrackDrawEntry = Tuple[float, float, float, float, int, int]


def _draw_track_overlay(
    frame_bgr: np.ndarray,
    frame_id: int,
    min_box_axis: float,
    entries: Sequence[_TrackDrawEntry],
) -> np.ndarray:
    """Return a copy of *frame_bgr* with boxes/labels for each track entry."""
    vis = frame_bgr.copy()

    for x1f, y1f, x2f, y2f, tid, cls_id in entries:
        bw = x2f - x1f
        bh = y2f - y1f
        if min(bw, bh) <= min_box_axis:
            continue
        x1, y1, x2, y2 = map(int, (x1f, y1f, x2f, y2f))
        color = _color_for_id(int(tid))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"{int(tid)} c{int(cls_id)}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)

    cv2.putText(vis, f"Frame {frame_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                cv2.LINE_AA)
    return vis


class RunRecorder:
    """Collects detections and tracks during a run and flushes to CSV.

    Optionally writes an annotated video with track overlays.

    Usage::

        recorder = RunRecorder(output_dir, save_video=True)
        recorder.save_config(config)

        for frame_id, frame in frames:
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            recorder.add_detections(frame_id, detections)
            recorder.add_tracks(frame_id, tracks, min_box_axis)
            recorder.write_video_frame(frame, tracks, frame_id, min_box_axis)

        recorder.save()
    """

    def __init__(
        self,
        output_dir: Path,
        log_level: int = logging.INFO,
        save_video: bool = False,
        video_fps: float = 24.0,
        *,
        persist_csv: bool = True,
        configure_logging: bool = True,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._det_rows: List[List] = []
        self._trk_rows: List[List] = []

        self._save_video = save_video
        self._video_fps = video_fps
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._persist_csv = persist_csv

        if configure_logging:
            self._configure_logging(log_level)

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------

    def _configure_logging(self, level: int) -> None:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        handlers: list[logging.Handler] = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(self._output_dir / "run.log", mode="w"),
        ]

        logging.basicConfig(
            level=level,
            format=fmt,
            datefmt=datefmt,
            handlers=handlers,
            force=True,
        )

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def save_config(self, config: Dict) -> None:
        with open(self._output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info("Config: %s", config)

    # ------------------------------------------------------------------
    # Per-frame collection
    # ------------------------------------------------------------------

    def add_detections(self, frame_id: int, detections: np.ndarray) -> None:
        """Record raw detector output for one frame.

        Args:
            frame_id:   1-based frame index.
            detections: (N, 6+) array ``[x1, y1, x2, y2, score, class_id, ...]``
                        or empty array.
        """
        if not hasattr(detections, "__len__") or len(detections) == 0:
            return

        for det in detections:
            self._det_rows.append([
                int(frame_id),
                float(det[0]),
                float(det[1]),
                float(det[2]),
                float(det[3]),
                float(det[4]),
                float(det[5]),
            ])

    def add_tracks(
        self,
        frame_id: int,
        online_targets: list,
        min_box_axis: float = 10.0,
    ) -> None:
        """Record tracker output for one frame.

        Args:
            frame_id:       1-based frame index.
            online_targets: list of STrack from ``tracker.update()``.
            min_box_axis:   Skip tracks whose shorter side is below this.
        """
        for t in online_targets:
            tlwh = t.tlwh
            if min(tlwh[2], tlwh[3]) <= min_box_axis:
                continue
            tlbr = t.tlbr
            self._trk_rows.append([
                int(frame_id),
                int(t.track_id),
                float(tlbr[0]),
                float(tlbr[1]),
                float(tlbr[2]),
                float(tlbr[3]),
                float(t.score),
                float(t.cls),
            ])

    # ------------------------------------------------------------------
    # Video output
    # ------------------------------------------------------------------

    def _ensure_video_writer(self, width: int, height: int) -> None:
        if self._video_writer is not None:
            return
        video_path = str(self._output_dir / "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            video_path, fourcc, self._video_fps, (width, height),
        )
        logger.info(
            "Video writer opened: %s (%dx%d @ %.1f fps)",
            video_path, width, height, self._video_fps,
        )

    def _write_annotated_video_frame(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        min_box_axis: float,
        entries: Sequence[_TrackDrawEntry],
    ) -> None:
        if not self._save_video:
            return
        h, w = frame_bgr.shape[:2]
        self._ensure_video_writer(w, h)
        vis = _draw_track_overlay(frame_bgr, frame_id, min_box_axis, entries)
        self._video_writer.write(vis)

    def write_video_frame(
        self,
        frame_bgr: np.ndarray,
        online_targets: list,
        frame_id: int,
        min_box_axis: float = 10.0,
    ) -> None:
        """Draw tracks on *frame_bgr* and write to the output video.

        No-op when ``save_video=False``.
        """
        entries: List[_TrackDrawEntry] = []
        for t in online_targets:
            tlwh = t.tlwh
            if min(tlwh[2], tlwh[3]) <= min_box_axis:
                continue
            br = t.tlbr
            entries.append(
                (
                    float(br[0]),
                    float(br[1]),
                    float(br[2]),
                    float(br[3]),
                    int(t.track_id),
                    int(t.cls),
                ),
            )
        self._write_annotated_video_frame(
            frame_bgr, frame_id, min_box_axis, entries,
        )

    def write_video_frame_from_track_rows(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        min_box_axis: float,
        rows: Sequence[Tuple[float, float, float, float, int, float, int]],
    ) -> None:
        """Draw track boxes from plain rows and append one video frame.

        Each row is ``(x1, y1, x2, y2, track_id, score, class_id)`` in pixel
        coordinates (same semantics as ``tracks.csv``). Used by offline tools
        (e.g. ``evaluate.py``) that replay predictions without live STrack objects.

        No-op when ``save_video=False``.
        """
        entries: List[_TrackDrawEntry] = [
            (float(x1), float(y1), float(x2), float(y2), int(tid), int(cls_id))
            for x1, y1, x2, y2, tid, _score, cls_id in rows
        ]
        self._write_annotated_video_frame(
            frame_bgr, frame_id, min_box_axis, entries,
        )

    # ------------------------------------------------------------------
    # Flush to disk
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write ``detections.csv``, ``tracks.csv`` (unless disabled), release video."""
        if self._persist_csv:
            det_path = self._output_dir / "detections.csv"
            trk_path = self._output_dir / "tracks.csv"

            self._write_csv(det_path, self._det_rows)
            logger.info("Detections saved to %s (%d rows)", det_path, len(self._det_rows))

            self._write_csv(trk_path, self._trk_rows)
            logger.info("Tracks saved to %s (%d rows)", trk_path, len(self._trk_rows))

        if self._video_writer is not None:
            self._video_writer.release()
            logger.info("Video saved to %s", self._output_dir / "output.mp4")

    @staticmethod
    def _write_csv(path: Path, rows: List[List]) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
