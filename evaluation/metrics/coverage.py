"""Track Coverage metric.

Measures the fraction of visible ground-truth target frames that are
matched to a valid tracker output, using Hungarian IoU matching
(same matching as CLEAR/MOTA).

    Coverage (%) = 100 × matched_visible_frames / total_visible_frames

Two variants are reported:

* **coverage** — micro-average across all GT detections (equivalent to
  ``recall × 100``).
* **coverage_per_track** — macro-average: mean of per-GT-track coverage
  ratios, so every ground-truth track contributes equally regardless
  of its length.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackCoverage:
    """Track Coverage metric compatible with the TrackEval eval interface.

    Accepts the same ``data`` dict produced by
    :func:`~evaluation.adapter.build_trackeval_data_from_frame_results`
    and used by TrackEval's CLEAR / Identity / HOTA metrics.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.threshold: float = float(config.get("THRESHOLD", 0.5))

    # -- per-sequence evaluation -------------------------------------------

    def eval_sequence(self, data: dict) -> dict:
        """Compute track coverage for one sequence."""
        num_gt_ids: int = data["num_gt_ids"]

        if data["num_gt_dets"] == 0:
            return self._empty_result(num_gt_ids)

        gt_id_count = np.zeros(num_gt_ids)
        gt_matched_count = np.zeros(num_gt_ids)

        if data["num_tracker_dets"] == 0:
            for gt_ids_t in data["gt_ids"]:
                if len(gt_ids_t) > 0:
                    gt_id_count[gt_ids_t] += 1
            return self._build_result(gt_id_count, gt_matched_count, num_gt_ids)

        prev_timestep_tracker_id = np.full(num_gt_ids, np.nan)

        for t, (gt_ids_t, tracker_ids_t) in enumerate(
            zip(data["gt_ids"], data["tracker_ids"]),
        ):
            if len(gt_ids_t) == 0:
                continue

            gt_id_count[gt_ids_t] += 1

            if len(tracker_ids_t) == 0:
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

            gt_matched_count[matched_gt_ids] += 1

            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids

        return self._build_result(gt_id_count, gt_matched_count, num_gt_ids)

    # -- cross-sequence combination ----------------------------------------

    def combine_sequences(self, all_res: Dict[str, dict]) -> dict:
        """Combine coverage results across sequences."""
        total_visible = sum(r["num_visible_frames"] for r in all_res.values())
        total_covered = sum(r["num_covered_frames"] for r in all_res.values())
        total_tracks = sum(r["num_gt_tracks"] for r in all_res.values())

        all_ratios: list[float] = []
        for r in all_res.values():
            gc = r["gt_id_count"]
            gm = r["gt_matched_count"]
            active = gc > 0
            if np.any(active):
                all_ratios.extend((gm[active] / gc[active]).tolist())

        coverage = (
            100.0 * total_covered / total_visible
            if total_visible > 0
            else 0.0
        )
        mean_per_track = (
            float(np.mean(all_ratios)) * 100.0 if all_ratios else 0.0
        )

        return {
            "num_visible_frames": total_visible,
            "num_covered_frames": total_covered,
            "num_gt_tracks": total_tracks,
            "coverage": coverage,
            "coverage_per_track": mean_per_track,
            "gt_id_count": np.zeros(0),
            "gt_matched_count": np.zeros(0),
        }

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _empty_result(num_gt_ids: int) -> dict:
        return {
            "num_visible_frames": 0,
            "num_covered_frames": 0,
            "num_gt_tracks": num_gt_ids,
            "coverage": 0.0,
            "coverage_per_track": 0.0,
            "gt_id_count": np.zeros(num_gt_ids),
            "gt_matched_count": np.zeros(num_gt_ids),
        }

    @staticmethod
    def _build_result(
        gt_id_count: np.ndarray,
        gt_matched_count: np.ndarray,
        num_gt_ids: int,
    ) -> dict:
        total_visible = int(np.sum(gt_id_count))
        total_covered = int(np.sum(gt_matched_count))

        active_mask = gt_id_count > 0
        if np.any(active_mask):
            per_track_ratios = (
                gt_matched_count[active_mask] / gt_id_count[active_mask]
            )
            mean_per_track = float(np.mean(per_track_ratios)) * 100.0
        else:
            mean_per_track = 0.0

        coverage = (
            100.0 * total_covered / total_visible
            if total_visible > 0
            else 0.0
        )

        return {
            "num_visible_frames": total_visible,
            "num_covered_frames": total_covered,
            "num_gt_tracks": num_gt_ids,
            "coverage": coverage,
            "coverage_per_track": mean_per_track,
            "gt_id_count": gt_id_count,
            "gt_matched_count": gt_matched_count,
        }
