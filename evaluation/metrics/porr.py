"""Post-Occlusion Recovery Rate (PORR).

Fraction of occlusion events where the tracker **re-identifies the same
track ID** on the first ground-truth frame after occlusion ends, among
events that satisfy pre-occlusion stability (see below).

**Occlusion**: GT visibility ≤ ``visibility_occluded_max`` (default 0.75).
If visibility is unknown anywhere on a track, that track is skipped for PORR.

**Size rows**: narrow side of the bbox (min of width/height, pixels) on the
last clearly visible frame before occlusion. Rows are the bins passed in
(typically from ``evaluation.size_bins`` via :func:`evaluation.filtering.size_bins_for_porr`).

**Time columns**: occlusion duration (seconds), upper edges
``TIME_COL_LABELS`` (0.25 … 2.0 s; longer → last bin).

**Stability**: the last ``min_pre_visible_frames`` frames before occlusion
must be clearly visible and matched to the **same** non-null tracker ID.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from evaluation.filtering import SizeBin, porr_metric_row_slug
from evaluation.schema import Detection, FrameResult

# Occlusion-duration column upper bounds (seconds); wider than last edge → last column.
TIME_COL_LABELS: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
N_TIME_BINS: int = len(TIME_COL_LABELS)


def _time_bin_index(duration_sec: float) -> int:
    """Map occlusion duration (seconds) to a column index."""
    for i, cap in enumerate(TIME_COL_LABELS):
        if duration_sec <= cap + 1e-9:
            return i
    return N_TIME_BINS - 1


def _size_bin_index(min_side_px: float, bins: List[SizeBin]) -> int:
    """Map narrow side (pixels) to a PORR size row."""
    if not bins:
        return 0
    for i, b in enumerate(bins):
        if b.contains(min_side_px):
            return i
    if min_side_px < bins[0].min_val:
        return 0
    return len(bins) - 1


def _gather_track_timelines(
    match_results: List[FrameResult],
) -> Dict[int, List[Tuple[int, float, int | None, float, bool]]]:
    """Per GT id: sorted (frame_id, vis, tracker_id|None, min_side, vis_known).

    *vis* is the raw MOT visibility in ``(0,1]`` when known; ``-1.0`` is unused
    because tracks with any unknown frame are skipped for PORR.
    """
    by_id: Dict[int, List[Tuple[int, float, int | None, float, bool]]] = {}
    for fr in match_results:
        for mp in fr.matched:
            g = mp.gt
            known = g.visibility is not None
            vis = float(g.visibility) if known else -1.0
            by_id.setdefault(g.object_id, []).append(
                (fr.frame_id, vis, int(mp.pred.object_id), float(g.min_side), known),
            )
        for g in fr.unmatched_gt:
            known = g.visibility is not None
            vis = float(g.visibility) if known else -1.0
            by_id.setdefault(g.object_id, []).append(
                (fr.frame_id, vis, None, float(g.min_side), known),
            )
    for oid in by_id:
        by_id[oid].sort(key=lambda x: x[0])
    return by_id


class PostOcclusionRecoveryRate:
    """PORR metric: occlusion/recovery statistics per size × time bin."""

    def __init__(
        self,
        config: dict | None = None,
        *,
        size_bins: List[SizeBin],
    ):
        c = config or {}
        self.visibility_occluded_max: float = float(
            c.get("VISIBILITY_OCCLUDED_MAX", c.get("visibility_occluded_max", 0.75)),
        )
        self.min_pre_visible_frames: int = int(
            c.get("MIN_PRE_VISIBLE_FRAMES", c.get("min_pre_visible_frames", 3)),
        )
        self._size_bins: List[SizeBin] = list(size_bins)
        self._row_slugs: Tuple[str, ...] = tuple(
            porr_metric_row_slug(b.name) for b in self._size_bins
        )
        self._n_rows: int = len(self._size_bins)

    def eval_from_match_results(
        self,
        match_results: List[FrameResult],
        *,
        fps: float,
    ) -> dict[str, Any]:
        """Compute PORR counts for one sequence (call on global ``all`` slice)."""
        if not match_results or fps <= 0:
            return self._empty_raw()

        success = np.zeros((self._n_rows, N_TIME_BINS), dtype=np.int64)
        total = np.zeros((self._n_rows, N_TIME_BINS), dtype=np.int64)

        timelines = _gather_track_timelines(match_results)
        any_vis = any(
            any(e[4] for e in ent) for ent in timelines.values()
        )
        for _oid, entries in timelines.items():
            self._process_track(
                entries, fps, success, total,
            )

        if not any_vis:
            out = self._empty_raw()
            out["skipped_no_gt_visibility"] = np.int64(1)
            return out

        out = {
            "porr_success": success,
            "porr_total": total,
            "skipped_no_gt_visibility": np.int64(0),
            "porr_row_slugs": self._row_slugs,
        }
        return out

    def _process_track(
        self,
        entries: List[Tuple[int, float, int | None, float, bool]],
        fps: float,
        success: np.ndarray,
        total: np.ndarray,
    ) -> None:
        occ_max = self.visibility_occluded_max
        k_req = self.min_pre_visible_frames
        if any(not e[4] for e in entries):
            return
        i = 0
        n = len(entries)
        while i < n:
            # Advance through a maximal clear prefix [i, j)
            j = i
            while j < n and entries[j][1] > occ_max:
                j += 1
            if j >= n:
                break
            # Occlusion starts at j; find end k
            k = j
            while k < n and entries[k][1] <= occ_max:
                k += 1
            clear_before = entries[i:j]
            occluded = entries[j:k]
            if not occluded:
                i = max(i + 1, k)
                continue
            # Need recovery frame?
            if len(clear_before) < k_req:
                i = k
                continue
            pre = clear_before[-k_req:]
            tids = [p[2] for p in pre]
            if any(t is None for t in tids):
                i = k
                continue
            anchor = tids[0]
            if not all(t == anchor for t in tids):
                i = k
                continue

            dur_sec = len(occluded) / fps
            sz = clear_before[-1][3]
            tb = _time_bin_index(dur_sec)
            sb = _size_bin_index(sz, self._size_bins)

            if k < n:
                rec_tid = entries[k][2]
                ok = rec_tid is not None and rec_tid == anchor
            else:
                # Track ends before recovery in annotated span
                i = k
                continue

            total[sb, tb] += 1
            if ok:
                success[sb, tb] += 1
            i = k

    def _empty_raw(self) -> dict[str, Any]:
        return {
            "porr_success": np.zeros((self._n_rows, N_TIME_BINS), dtype=np.int64),
            "porr_total": np.zeros((self._n_rows, N_TIME_BINS), dtype=np.int64),
            "skipped_no_gt_visibility": np.int64(0),
            "porr_row_slugs": self._row_slugs,
        }

    def combine_sequences(self, all_res: Dict[str, dict]) -> dict[str, Any]:
        """Sum PORR matrices across sequences."""
        acc_s = np.zeros((self._n_rows, N_TIME_BINS), dtype=np.int64)
        acc_t = np.zeros((self._n_rows, N_TIME_BINS), dtype=np.int64)
        skip = np.int64(0)
        slugs: Tuple[str, ...] | None = None
        for raw in all_res.values():
            acc_s = acc_s + raw["porr_success"]
            acc_t = acc_t + raw["porr_total"]
            skip += raw.get("skipped_no_gt_visibility", np.int64(0))
            if slugs is None:
                slugs = raw.get("porr_row_slugs", self._row_slugs)
        return {
            "porr_success": acc_s,
            "porr_total": acc_t,
            "skipped_no_gt_visibility": skip,
            "porr_row_slugs": slugs or self._row_slugs,
        }

    # TrackEval-compatible no-op (pipeline uses eval_from_match_results)
    def eval_sequence(self, data: dict) -> dict:
        return self._empty_raw()


def format_porr_metrics(raw: dict[str, Any]) -> Dict[str, float]:
    """Flatten PORR raw dict to JSON-serializable floats for :class:`BinResult`."""
    out: Dict[str, float] = {}
    success = raw["porr_success"]
    tot = raw["porr_total"]
    n = int(success.shape[0])
    row_slugs_in = raw.get("porr_row_slugs")
    if row_slugs_in:
        row_slugs = tuple(row_slugs_in)
    else:
        row_slugs = tuple(f"r{i}" for i in range(n))
    if len(row_slugs) < n:
        row_slugs = tuple(
            list(row_slugs) + [f"r{i}" for i in range(len(row_slugs), n)],
        )
    elif len(row_slugs) > n:
        row_slugs = row_slugs[:n]

    total_events = float(np.sum(tot))
    sum_ok = float(np.sum(success))
    out["porr_mean"] = sum_ok / total_events if total_events > 0 else float("nan")
    out["porr_num_events"] = total_events
    out["porr_skipped_no_visibility"] = float(
        raw.get("skipped_no_gt_visibility", 0),
    )
    out["porr_n_size_bins"] = float(n)

    for si in range(n):
        sname = row_slugs[si]
        for ti, tsec in enumerate(TIME_COL_LABELS):
            key = f"porr_{sname}_{tsec:g}"
            t = int(tot[si, ti])
            if t > 0:
                out[key] = float(success[si, ti]) / float(t)
            else:
                out[key] = float("nan")
            out[f"porr_n_{sname}_{tsec:g}"] = float(t)

    return out


def porr_table_arrays(
    formatted: Dict[str, float],
    row_slugs: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (rates matrix, counts matrix) from :func:`format_porr_metrics` output."""
    n = len(row_slugs)
    rates = np.full((n, N_TIME_BINS), np.nan, dtype=np.float64)
    counts = np.zeros((n, N_TIME_BINS), dtype=np.float64)
    for si, sname in enumerate(row_slugs):
        for ti, tsec in enumerate(TIME_COL_LABELS):
            rates[si, ti] = formatted.get(f"porr_{sname}_{tsec:g}", float("nan"))
            counts[si, ti] = formatted.get(f"porr_n_{sname}_{tsec:g}", 0.0)
    return rates, counts
