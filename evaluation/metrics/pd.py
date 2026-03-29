"""Probability of Detection (PD) metric.

Measures the fraction of ground-truth objects (tracks) that triggered
at least one tracker track.

    PD = # GT objects that triggered a track / # GT objects

A GT object "triggers" a tracker track when the tracker track is
**first** matched to that GT object.  If a tracker track later
switches to a different GT (ID switch), the new GT was tracked but
did **not** trigger that track — it is excluded from the numerator.

Uses CLEAR-style Hungarian matching (IoU threshold + continuity
bonus) to determine per-frame matches.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment


class ProbabilityOfDetection:
    """PD metric compatible with the TrackEval eval interface."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.threshold: float = float(config.get("THRESHOLD", 0.5))

    def eval_sequence(self, data: dict) -> dict:
        """Compute PD for one sequence."""
        num_gt_ids: int = data["num_gt_ids"]

        if data["num_gt_dets"] == 0:
            return self._empty_result(0)

        if data["num_tracker_dets"] == 0:
            return self._empty_result(num_gt_ids)

        tracker_first_gt: dict[int, int] = {}
        prev_timestep_tracker_id = np.full(num_gt_ids, np.nan)

        for t, (gt_ids_t, tracker_ids_t) in enumerate(
            zip(data["gt_ids"], data["tracker_ids"]),
        ):
            if len(gt_ids_t) == 0 or len(tracker_ids_t) == 0:
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

            for gi, ti in zip(matched_gt_ids, matched_tracker_ids):
                tid = int(ti)
                if tid not in tracker_first_gt:
                    tracker_first_gt[tid] = int(gi)

            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids

        triggered_gts = set(tracker_first_gt.values())
        num_detected = len(triggered_gts)
        pd_value = (
            100.0 * num_detected / num_gt_ids if num_gt_ids > 0 else 0.0
        )

        return {
            "pd": pd_value,
            "num_detected": num_detected,
            "num_gt_objects": num_gt_ids,
        }

    def combine_sequences(self, all_res: Dict[str, dict]) -> dict:
        """Combine PD results across sequences (micro aggregation)."""
        total_detected = sum(r["num_detected"] for r in all_res.values())
        total_gt = sum(r["num_gt_objects"] for r in all_res.values())
        pd_value = (
            100.0 * total_detected / total_gt if total_gt > 0 else 0.0
        )
        return {
            "pd": pd_value,
            "num_detected": total_detected,
            "num_gt_objects": total_gt,
        }

    @staticmethod
    def _empty_result(num_gt_objects: int) -> dict:
        return {
            "pd": 0.0,
            "num_detected": 0,
            "num_gt_objects": num_gt_objects,
        }
