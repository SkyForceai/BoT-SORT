"""Adapter converting parsed CSV data to TrackEval's expected format.

TrackEval metric classes (HOTA, CLEAR, Identity) expect a specific
data dict with contiguous 0-based integer IDs, per-timestep arrays,
and precomputed similarity scores.  This module handles that conversion
from our :class:`~evaluation.schema.SequenceData` types, including IoU
computation **without** the PASCAL VOC -1 pixel adjustment to match
TrackEval's convention exactly.
"""

from __future__ import annotations

from typing import List

import numpy as np

from scipy.optimize import linear_sum_assignment

from evaluation.schema import (
    AnnotationMask,
    FrameResult,
    MatchedPair,
    SequenceData,
)


def compute_ious_xyxy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of ``(x1, y1, x2, y2)`` boxes.

    No -1 pixel adjustment — matches TrackEval's standard IoU.

    Parameters
    ----------
    boxes_a : (N, 4) array
    boxes_b : (M, 4) array

    Returns
    -------
    (N, M) IoU matrix
    """
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / union, 0.0)


def build_trackeval_data(
    gt: SequenceData,
    preds: SequenceData,
    annotation_mask: AnnotationMask,
) -> dict:
    """Convert parsed sequence data to TrackEval's expected data dict.

    The returned dict has the format expected by
    ``HOTA.eval_sequence``, ``CLEAR.eval_sequence``,
    and ``Identity.eval_sequence``:

    - ``gt_ids``, ``tracker_ids``: lists of 1-D int arrays
      (contiguous 0-based IDs, used as matrix indices by TrackEval)
    - ``similarity_scores``: list of 2-D float arrays (IoU matrices)
    - ``num_timesteps``, ``num_gt_ids``, ``num_tracker_ids``
    - ``num_gt_dets``, ``num_tracker_dets``
    """
    annotated = sorted(annotation_mask.resolve_frame_ids(gt, preds))
    num_timesteps = len(annotated)

    gt_ids_list: List[np.ndarray] = []
    tracker_ids_list: List[np.ndarray] = []
    similarity_list: List[np.ndarray] = []

    unique_gt_ids: set = set()
    unique_tracker_ids: set = set()
    total_gt_dets = 0
    total_tracker_dets = 0

    for fid in annotated:
        gt_frame = gt.frames.get(fid)
        pred_frame = preds.frames.get(fid)

        gt_dets = gt_frame.detections if gt_frame else []
        pred_dets = pred_frame.detections if pred_frame else []

        gt_obj_ids = (
            np.array([d.object_id for d in gt_dets], dtype=int)
            if gt_dets
            else np.empty(0, dtype=int)
        )
        pred_obj_ids = (
            np.array([d.object_id for d in pred_dets], dtype=int)
            if pred_dets
            else np.empty(0, dtype=int)
        )

        unique_gt_ids.update(gt_obj_ids.tolist())
        unique_tracker_ids.update(pred_obj_ids.tolist())
        total_gt_dets += len(gt_dets)
        total_tracker_dets += len(pred_dets)

        if len(gt_dets) > 0 and len(pred_dets) > 0:
            gt_boxes = np.array(
                [d.bbox_xyxy for d in gt_dets], dtype=np.float64,
            )
            pred_boxes = np.array(
                [d.bbox_xyxy for d in pred_dets], dtype=np.float64,
            )
            sim = compute_ious_xyxy(gt_boxes, pred_boxes)
        else:
            sim = np.zeros(
                (len(gt_dets), len(pred_dets)), dtype=np.float64,
            )

        gt_ids_list.append(gt_obj_ids)
        tracker_ids_list.append(pred_obj_ids)
        similarity_list.append(sim)

    # Remap IDs to contiguous 0-based integers (TrackEval uses them
    # as array indices in potential_matches_count matrices).
    sorted_gt = sorted(unique_gt_ids)
    sorted_tr = sorted(unique_tracker_ids)
    gt_map = {orig: new for new, orig in enumerate(sorted_gt)}
    tr_map = {orig: new for new, orig in enumerate(sorted_tr)}

    for t in range(num_timesteps):
        if len(gt_ids_list[t]) > 0:
            gt_ids_list[t] = np.array(
                [gt_map[int(x)] for x in gt_ids_list[t]], dtype=int,
            )
        if len(tracker_ids_list[t]) > 0:
            tracker_ids_list[t] = np.array(
                [tr_map[int(x)] for x in tracker_ids_list[t]], dtype=int,
            )

    return {
        "num_timesteps": num_timesteps,
        "num_gt_ids": len(sorted_gt),
        "num_tracker_ids": len(sorted_tr),
        "num_gt_dets": total_gt_dets,
        "num_tracker_dets": total_tracker_dets,
        "gt_ids": gt_ids_list,
        "tracker_ids": tracker_ids_list,
        "similarity_scores": similarity_list,
    }


# ------------------------------------------------------------------
# Global matching  (match once, then slice by size bin)
# ------------------------------------------------------------------

def global_match_sequence(
    gt: SequenceData,
    preds: SequenceData,
    annotation_mask: AnnotationMask,
    iou_threshold: float = 0.5,
) -> List[FrameResult]:
    """Run CLEAR-style Hungarian matching once on all objects.

    Uses the same per-frame matching algorithm as TrackEval's CLEAR
    metric: Hungarian assignment maximising ``1000 * continuity + IoU``
    with an IoU threshold gate.

    Returns one :class:`FrameResult` per annotated frame, containing
    matched pairs, unmatched GT, and unmatched predictions.  These
    results are then sliced by size bin before being fed to TrackEval
    metrics for per-bin evaluation.
    """
    annotated = sorted(annotation_mask.resolve_frame_ids(gt, preds))

    all_gt_oids: set[int] = set()
    for fid in annotated:
        gt_frame = gt.frames.get(fid)
        if gt_frame:
            for d in gt_frame.detections:
                all_gt_oids.add(d.object_id)

    sorted_gt_oids = sorted(all_gt_oids)
    gt_oid_to_idx = {oid: idx for idx, oid in enumerate(sorted_gt_oids)}
    num_gt_ids = len(sorted_gt_oids)

    prev_timestep_tracker_id = np.full(num_gt_ids, np.nan)

    results: List[FrameResult] = []

    for fid in annotated:
        gt_frame = gt.frames.get(fid)
        pred_frame = preds.frames.get(fid)

        gt_dets = list(gt_frame.detections) if gt_frame else []
        pred_dets = list(pred_frame.detections) if pred_frame else []

        if not gt_dets or not pred_dets:
            results.append(FrameResult(
                frame_id=fid,
                gt_detections=gt_dets,
                pred_detections=pred_dets,
                matched=[],
                unmatched_gt=list(gt_dets),
                unmatched_pred=list(pred_dets),
            ))
            continue

        gt_boxes = np.array(
            [d.bbox_xyxy for d in gt_dets], dtype=np.float64,
        )
        pred_boxes = np.array(
            [d.bbox_xyxy for d in pred_dets], dtype=np.float64,
        )
        similarity = compute_ious_xyxy(gt_boxes, pred_boxes)

        gt_indices = np.array(
            [gt_oid_to_idx[d.object_id] for d in gt_dets],
        )
        pred_oid_arr = np.array(
            [d.object_id for d in pred_dets], dtype=float,
        )

        score_mat = (
            pred_oid_arr[np.newaxis, :]
            == prev_timestep_tracker_id[gt_indices[:, np.newaxis]]
        ).astype(float)
        score_mat = 1000.0 * score_mat + similarity
        score_mat[
            similarity < iou_threshold - np.finfo("float").eps
        ] = 0

        match_rows, match_cols = linear_sum_assignment(-score_mat)
        actually_matched = (
            score_mat[match_rows, match_cols] > np.finfo("float").eps
        )
        match_rows = match_rows[actually_matched]
        match_cols = match_cols[actually_matched]

        matched: List[MatchedPair] = []
        matched_gt_idx: set[int] = set()
        matched_pred_idx: set[int] = set()

        prev_timestep_tracker_id[:] = np.nan

        for gr, pc in zip(match_rows, match_cols):
            gt_det = gt_dets[gr]
            pred_det = pred_dets[pc]
            matched.append(MatchedPair(
                gt=gt_det, pred=pred_det,
                iou=float(similarity[gr, pc]),
            ))
            matched_gt_idx.add(int(gr))
            matched_pred_idx.add(int(pc))
            prev_timestep_tracker_id[
                gt_oid_to_idx[gt_det.object_id]
            ] = pred_det.object_id

        unmatched_gt = [
            gt_dets[i] for i in range(len(gt_dets))
            if i not in matched_gt_idx
        ]
        unmatched_pred = [
            pred_dets[i] for i in range(len(pred_dets))
            if i not in matched_pred_idx
        ]

        results.append(FrameResult(
            frame_id=fid,
            gt_detections=gt_dets,
            pred_detections=pred_dets,
            matched=matched,
            unmatched_gt=unmatched_gt,
            unmatched_pred=unmatched_pred,
        ))

    return results


def build_trackeval_data_from_frame_results(
    frame_results: List[FrameResult],
) -> dict:
    """Convert (sliced) :class:`FrameResult` list to a TrackEval data dict.

    Each frame contributes:

    * **GT detections** = matched-pair GTs + unmatched GTs
    * **Tracker detections** = matched-pair predictions + unmatched predictions
    * **Similarity scores** = recomputed IoU matrix between the above

    The returned dict has exactly the same schema as
    :func:`build_trackeval_data`.
    """
    gt_ids_list: List[np.ndarray] = []
    tracker_ids_list: List[np.ndarray] = []
    similarity_list: List[np.ndarray] = []

    unique_gt_ids: set[int] = set()
    unique_tracker_ids: set[int] = set()
    total_gt_dets = 0
    total_tracker_dets = 0

    for fr in frame_results:
        gt_dets = [mp.gt for mp in fr.matched] + list(fr.unmatched_gt)
        pred_dets = [mp.pred for mp in fr.matched] + list(fr.unmatched_pred)

        gt_obj_ids = (
            np.array([d.object_id for d in gt_dets], dtype=int)
            if gt_dets
            else np.empty(0, dtype=int)
        )
        pred_obj_ids = (
            np.array([d.object_id for d in pred_dets], dtype=int)
            if pred_dets
            else np.empty(0, dtype=int)
        )

        unique_gt_ids.update(gt_obj_ids.tolist())
        unique_tracker_ids.update(pred_obj_ids.tolist())
        total_gt_dets += len(gt_dets)
        total_tracker_dets += len(pred_dets)

        if gt_dets and pred_dets:
            gt_boxes = np.array(
                [d.bbox_xyxy for d in gt_dets], dtype=np.float64,
            )
            pred_boxes = np.array(
                [d.bbox_xyxy for d in pred_dets], dtype=np.float64,
            )
            sim = compute_ious_xyxy(gt_boxes, pred_boxes)
        else:
            sim = np.zeros(
                (len(gt_dets), len(pred_dets)), dtype=np.float64,
            )

        gt_ids_list.append(gt_obj_ids)
        tracker_ids_list.append(pred_obj_ids)
        similarity_list.append(sim)

    sorted_gt = sorted(unique_gt_ids)
    sorted_tr = sorted(unique_tracker_ids)
    gt_map = {orig: new for new, orig in enumerate(sorted_gt)}
    tr_map = {orig: new for new, orig in enumerate(sorted_tr)}

    num_timesteps = len(frame_results)
    for t in range(num_timesteps):
        if len(gt_ids_list[t]) > 0:
            gt_ids_list[t] = np.array(
                [gt_map[int(x)] for x in gt_ids_list[t]], dtype=int,
            )
        if len(tracker_ids_list[t]) > 0:
            tracker_ids_list[t] = np.array(
                [tr_map[int(x)] for x in tracker_ids_list[t]], dtype=int,
            )

    return {
        "num_timesteps": num_timesteps,
        "num_gt_ids": len(sorted_gt),
        "num_tracker_ids": len(sorted_tr),
        "num_gt_dets": total_gt_dets,
        "num_tracker_dets": total_tracker_dets,
        "gt_ids": gt_ids_list,
        "tracker_ids": tracker_ids_list,
        "similarity_scores": similarity_list,
    }
