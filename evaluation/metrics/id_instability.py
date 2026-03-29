"""ID Instability Rate metric.

For each ground-truth object, computes:

    object_rate = # ID switches for that object / visible time (minutes)

Then averages across all GT objects (macro / per-object average), so
every object contributes equally regardless of how long it is visible.

Uses CLEAR-style Hungarian matching (IoU threshold + continuity
bonus) to detect ID switches: a switch occurs when a GT object was
previously matched to tracker ID X and is now matched to tracker
ID Y != X.

Lower is better.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment


class IDInstabilityRate:
    """ID Instability Rate metric compatible with the TrackEval interface."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.threshold: float = float(config.get("THRESHOLD", 0.5))
        self.fps: float = float(config.get("FPS", 30.0))

    def eval_sequence(self, data: dict) -> dict:
        """Compute per-object ID instability rate for one sequence."""
        num_gt_ids: int = data["num_gt_ids"]

        if data["num_gt_dets"] == 0:
            return self._empty_result()

        gt_idsw = np.zeros(num_gt_ids, dtype=int)
        gt_visible = np.zeros(num_gt_ids, dtype=int)

        prev_tracker_id = np.full(num_gt_ids, np.nan)
        prev_timestep_tracker_id = np.full(num_gt_ids, np.nan)

        for t, (gt_ids_t, tracker_ids_t) in enumerate(
            zip(data["gt_ids"], data["tracker_ids"]),
        ):
            if len(gt_ids_t) == 0:
                continue

            np.add.at(gt_visible, gt_ids_t, 1)

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

            prev_matched = prev_tracker_id[matched_gt_ids]
            is_idsw = (
                np.logical_not(np.isnan(prev_matched))
                & np.not_equal(matched_tracker_ids, prev_matched)
            )
            np.add.at(gt_idsw, matched_gt_ids[is_idsw], 1)

            prev_tracker_id[matched_gt_ids] = matched_tracker_ids
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids

        per_object_rates: list[float] = []
        for i in range(num_gt_ids):
            if gt_visible[i] > 0:
                visible_min = gt_visible[i] / (self.fps * 60.0)
                per_object_rates.append(float(gt_idsw[i]) / visible_min)

        num_objects = len(per_object_rates)
        sum_rates = sum(per_object_rates)
        rate = sum_rates / num_objects if num_objects > 0 else 0.0

        return {
            "id_instability_rate": rate,
            "sum_per_object_rates": sum_rates,
            "num_gt_objects": num_objects,
            "total_idsw": int(np.sum(gt_idsw)),
        }

    def combine_sequences(self, all_res: Dict[str, dict]) -> dict:
        """Combine per-object rates across sequences (macro average)."""
        total_sum = sum(r["sum_per_object_rates"] for r in all_res.values())
        total_objects = sum(r["num_gt_objects"] for r in all_res.values())
        rate = total_sum / total_objects if total_objects > 0 else 0.0
        return {
            "id_instability_rate": rate,
            "sum_per_object_rates": total_sum,
            "num_gt_objects": total_objects,
            "total_idsw": sum(r["total_idsw"] for r in all_res.values()),
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "id_instability_rate": 0.0,
            "sum_per_object_rates": 0.0,
            "num_gt_objects": 0,
            "total_idsw": 0,
        }
