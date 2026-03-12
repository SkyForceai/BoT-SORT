"""
Unified box operations for TCB.

Consolidated from:
- RT-DETR (https://github.com/lyuwenyu/RT-DETR) - Copyright(c) 2023 lyuwenyu
- DETR (https://github.com/facebookresearch/detr) - Copyright (c) Facebook, Inc.
"""

from typing import Tuple

import torch
import torchvision
from torch import Tensor
from torchvision.ops.boxes import box_area


# =============================================================================
# Box Format Conversions
# =============================================================================

def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    """
    Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).
    
    Args:
        x: Tensor of shape [..., 4] with (cx, cy, w, h) format
        
    Returns:
        Tensor of shape [..., 4] with (x1, y1, x2, y2) format
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w.clamp(min=0.0)),
        (y_c - 0.5 * h.clamp(min=0.0)),
        (x_c + 0.5 * w.clamp(min=0.0)),
        (y_c + 0.5 * h.clamp(min=0.0)),
    ]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    """
    Convert boxes from corner format (x1, y1, x2, y2) to center format (cx, cy, w, h).
    
    Args:
        x: Tensor of shape [..., 4] with (x1, y1, x2, y2) format
        
    Returns:
        Tensor of shape [..., 4] with (cx, cy, w, h) format
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# =============================================================================
# IoU Computations - Pairwise (N x M matrices)
# =============================================================================

def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute pairwise IoU between two sets of boxes.
    
    Modified from torchvision to also return the union.
    
    Args:
        boxes1: Tensor of shape [N, 4] with (x1, y1, x2, y2) format
        boxes2: Tensor of shape [M, 4] with (x1, y1, x2, y2) format
        
    Returns:
        iou: Tensor of shape [N, M] with pairwise IoU values
        union: Tensor of shape [N, M] with pairwise union areas
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute Generalized IoU (GIoU) between two sets of boxes.
    
    GIoU from https://giou.stanford.edu/
    
    Args:
        boxes1: Tensor of shape [N, 4] with (x1, y1, x2, y2) format
        boxes2: Tensor of shape [M, 4] with (x1, y1, x2, y2) format
        
    Returns:
        Tensor of shape [N, M] with pairwise GIoU values
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1: x2 must be >= x1 and y2 must be >= y1"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2: x2 must be >= x1 and y2 must be >= y1"
    
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


# =============================================================================
# IoU Computations - Elementwise (N vectors, paired boxes)
# =============================================================================

def elementwise_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute elementwise IoU between paired boxes.
    
    Args:
        boxes1: Tensor of shape [N, 4] with (x1, y1, x2, y2) format
        boxes2: Tensor of shape [N, 4] with (x1, y1, x2, y2) format (same N)
        
    Returns:
        iou: Tensor of shape [N] with IoU for each pair
        union: Tensor of shape [N] with union area for each pair
    """
    area1 = torchvision.ops.box_area(boxes1)  # [N]
    area2 = torchvision.ops.box_area(boxes2)  # [N]
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N, 2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N, 2]
    wh = (rb - lt).clamp(min=0)  # [N, 2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    union = area1 + area2 - inter
    iou = inter / union
    return iou, union


def elementwise_generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute elementwise Generalized IoU (GIoU) between paired boxes.
    
    Args:
        boxes1: Tensor of shape [N, 4] with (x1, y1, x2, y2) format
        boxes2: Tensor of shape [N, 4] with (x1, y1, x2, y2) format (same N)
        
    Returns:
        giou: Tensor of shape [N] with GIoU for each pair
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = elementwise_box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, :2], boxes2[:, :2])  # [N, 2]
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])  # [N, 2]
    wh = (rb - lt).clamp(min=0)  # [N, 2]
    area = wh[:, 0] * wh[:, 1]
    return iou - (area - union) / area


# =============================================================================
# Point-Box Operations
# =============================================================================

def check_point_inside_box(points: Tensor, boxes: Tensor, eps: float = 1e-9) -> Tensor:
    """
    Check if points are inside boxes.
    
    Args:
        points: Tensor of shape [K, 2] with (x, y) coordinates
        boxes: Tensor of shape [N, 4] with (x1, y1, x2, y2) format
        eps: Small epsilon for numerical stability
        
    Returns:
        Tensor of shape [K, N] (bool) indicating if point k is inside box n
    """
    x, y = [p.unsqueeze(-1) for p in points.unbind(-1)]
    x1, y1, x2, y2 = [b.unsqueeze(0) for b in boxes.unbind(-1)]

    l = x - x1
    t = y - y1
    r = x2 - x
    b = y2 - y

    ltrb = torch.stack([l, t, r, b], dim=-1)
    mask = ltrb.min(dim=-1).values > eps

    return mask


def point_box_distance(points: Tensor, boxes: Tensor) -> Tensor:
    """
    Compute distances from points to box edges.
    
    Args:
        points: Tensor of shape [N, 2] with (x, y) coordinates
        boxes: Tensor of shape [N, 4] with (x1, y1, x2, y2) format
        
    Returns:
        Tensor of shape [N, 4] with (left, top, right, bottom) distances
    """
    x1y1, x2y2 = torch.split(boxes, 2, dim=-1)
    lt = points - x1y1
    rb = x2y2 - points
    return torch.concat([lt, rb], dim=-1)


def point_distance_box(points: Tensor, distances: Tensor) -> Tensor:
    """
    Convert point + distances to box coordinates.
    
    Args:
        points: Tensor of shape [N, 2] with (x, y) coordinates
        distances: Tensor of shape [N, 4] with (left, top, right, bottom) distances
        
    Returns:
        Tensor of shape [N, 4] with (x1, y1, x2, y2) box coordinates
    """
    lt, rb = torch.split(distances, 2, dim=-1)
    x1y1 = -lt + points
    x2y2 = rb + points
    boxes = torch.concat([x1y1, x2y2], dim=-1)
    return boxes


# =============================================================================
# Mask Operations
# =============================================================================

def masks_to_boxes(masks: Tensor) -> Tensor:
    """
    Compute bounding boxes around the provided masks.
    
    Args:
        masks: Tensor of shape [N, H, W] where N is the number of masks
        
    Returns:
        Tensor of shape [N, 4] with boxes in (x1, y1, x2, y2) format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x, indexing="ij")

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
