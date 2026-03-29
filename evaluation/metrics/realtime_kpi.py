"""Real-Time KPI: Track Initiation Delay (TID).

Measures how quickly the tracker reacts to new ground-truth objects:

    TID (per GT object) = # frames from GT first appearance to first match

Reported values:

* **tid_mean_frames** — mean delay in frames (macro-averaged over GT objects).
* **tid_mean_sec** — mean delay in seconds (``tid_mean_frames / fps``).
* **tid_median_frames** — median delay in frames.
* **tid_ratio_immediate** — fraction of GT objects matched on their very
  first frame (delay = 0), i.e. "instantaneous detection rate".
* **tid_ratio_within_1s** — fraction matched within 1 second.
* **num_gt_objects** — total GT objects considered.
* **num_never_matched** — GT objects that are *never* matched in any frame.

A GT object that is never matched contributes ``num_visible_frames`` as
its delay (worst case: the tracker never initiated a track for it).

Uses CLEAR-style Hungarian matching (IoU threshold + continuity bonus),
consistent with all other metrics in the pipeline.

Lower TID is better; higher immediate/within-1s ratios are better.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment


class RealTimeKPI:
    """Track Initiation Delay metric (TrackEval-compatible interface)."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.threshold: float = float(config.get("THRESHOLD", 0.5))
        self.fps: float = float(config.get("FPS", 30.0))

    def eval_sequence(self, data: dict) -> dict:
        """Compute TID for one sequence."""
        num_gt_ids: int = data["num_gt_ids"]

        if data["num_gt_dets"] == 0:
            return self._empty_result()

        gt_first_appearance = np.full(num_gt_ids, -1, dtype=int)
        gt_first_match = np.full(num_gt_ids, -1, dtype=int)

        prev_timestep_tracker_id = np.full(num_gt_ids, np.nan)

        for t, (gt_ids_t, tracker_ids_t) in enumerate(
            zip(data["gt_ids"], data["tracker_ids"]),
        ):
            if len(gt_ids_t) == 0:
                continue

            for gid in gt_ids_t:
                if gt_first_appearance[gid] < 0:
                    gt_first_appearance[gid] = t

            if len(tracker_ids_t) == 0:
                prev_timestep_tracker_id[:] = np.nan
                continue

            similarity = data["similarity_scores"][t]

            score_mat = (
                tracker_ids_t[np.newaxis, :]
                == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]]
            )
            score_mat = 1000 * score_mat + similarity
            score_mat[
                similarity < self.threshold - np.finfo("float").eps
            ] = 0

            match_rows, match_cols = linear_sum_assignment(-score_mat)
            actually_matched = (
                score_mat[match_rows, match_cols]
                > 0 + np.finfo("float").eps
            )
            match_rows = match_rows[actually_matched]
            match_cols = match_cols[actually_matched]

            matched_gt_ids = gt_ids_t[match_rows]
            matched_tracker_ids = tracker_ids_t[match_cols]

            for gid in matched_gt_ids:
                if gt_first_match[gid] < 0:
                    gt_first_match[gid] = t

            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids

        return self._build_result(
            gt_first_appearance, gt_first_match,
            num_gt_ids, data["num_timesteps"],
        )

    def combine_sequences(self, all_res: Dict[str, dict]) -> dict:
        """Combine TID results across sequences (macro average)."""
        all_delays: list[int] = []
        total_never = 0
        total_gt = 0
        for r in all_res.values():
            all_delays.extend(r["delays"])
            total_never += r["num_never_matched"]
            total_gt += r["num_gt_objects"]

        return self._compute_stats(all_delays, total_never, total_gt)

    def _build_result(
        self,
        gt_first_appearance: np.ndarray,
        gt_first_match: np.ndarray,
        num_gt_ids: int,
        num_timesteps: int,
    ) -> dict:
        active = gt_first_appearance >= 0
        if not np.any(active):
            return self._empty_result()

        delays: list[int] = []
        num_never = 0

        for i in range(num_gt_ids):
            if not active[i]:
                continue
            if gt_first_match[i] < 0:
                visible_frames = num_timesteps - int(gt_first_appearance[i])
                delays.append(visible_frames)
                num_never += 1
            else:
                delays.append(int(gt_first_match[i] - gt_first_appearance[i]))

        return self._compute_stats(delays, num_never, len(delays))

    def _compute_stats(
        self,
        delays: list[int],
        num_never: int,
        num_gt: int,
    ) -> dict:
        if not delays:
            return self._empty_result()

        arr = np.array(delays, dtype=np.float64)
        fps_threshold = self.fps

        return {
            "tid_mean_frames": float(np.mean(arr)),
            "tid_mean_sec": float(np.mean(arr)) / self.fps if self.fps > 0 else 0.0,
            "tid_median_frames": float(np.median(arr)),
            "tid_ratio_immediate": float(np.mean(arr == 0)),
            "tid_ratio_within_1s": float(np.mean(arr <= fps_threshold)),
            "num_gt_objects": num_gt,
            "num_never_matched": num_never,
            "delays": [int(d) for d in delays],
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "tid_mean_frames": 0.0,
            "tid_mean_sec": 0.0,
            "tid_median_frames": 0.0,
            "tid_ratio_immediate": 0.0,
            "tid_ratio_within_1s": 0.0,
            "num_gt_objects": 0,
            "num_never_matched": 0,
            "delays": [],
        }
