"""Postprocessing: clip, NMS, merge, and filter detections.

Every function takes and returns a list of detection dicts so they compose
cleanly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.config import MERGE_IOU_THRESH, MIN_BOX_AREA_PX, NMS_IOU_THRESH

logger = logging.getLogger(__name__)


def clip_boxes(
    detections: List[Dict[str, Any]],
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    """Clip all bounding boxes to [0, width] x [0, height]."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        det["bbox"] = [
            max(0.0, min(float(x1), float(width))),
            max(0.0, min(float(y1), float(height))),
            max(0.0, min(float(x2), float(width))),
            max(0.0, min(float(y2), float(height))),
        ]
    return detections


def filter_small_boxes(
    detections: List[Dict[str, Any]],
    min_area: int = MIN_BOX_AREA_PX,
) -> List[Dict[str, Any]]:
    """Remove detections whose area is below min_area pixels."""
    kept: List[Dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if (x2 - x1) * (y2 - y1) >= min_area:
            kept.append(det)
    return kept


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlap_ratio(box_small: List[float], box_large: List[float]) -> float:
    """Compute fraction of box_small that is contained inside box_large."""
    xa = max(box_small[0], box_large[0])
    ya = max(box_small[1], box_large[1])
    xb = min(box_small[2], box_large[2])
    yb = min(box_small[3], box_large[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_small = (box_small[2] - box_small[0]) * (box_small[3] - box_small[1])
    return inter / area_small if area_small > 0 else 0.0


def nms(
    detections: List[Dict[str, Any]],
    iou_thresh: float = NMS_IOU_THRESH,
) -> List[Dict[str, Any]]:
    """Greedy NMS sorted by confidence."""
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep: List[Dict[str, Any]] = []
    suppressed = set()

    for i, d in enumerate(dets):
        if i in suppressed:
            continue
        keep.append(d)
        for j in range(i + 1, len(dets)):
            if j in suppressed:
                continue
            # If same-class or heavy overlap, suppress lower confidence
            if _iou(d["bbox"], dets[j]["bbox"]) >= iou_thresh:
                suppressed.add(j)
            # If a small box is > 70% contained in a filler box, suppress it
            elif d.get("label") == "filler" and _overlap_ratio(dets[j]["bbox"], d["bbox"]) > 0.70:
                suppressed.add(j)

    return keep


def merge_same_class(
    detections: List[Dict[str, Any]],
    iou_thresh: float = MERGE_IOU_THRESH,
) -> List[Dict[str, Any]]:
    """Merge pairs of same-class boxes with high IoU or adjacent overlap."""
    merged = True
    result = list(detections)
    while merged:
        merged = False
        new_result: List[Dict[str, Any]] = []
        used = set()
        for i in range(len(result)):
            if i in used:
                continue
            current = dict(result[i])
            current["bbox"] = list(current["bbox"])
            for j in range(i + 1, len(result)):
                if j in used:
                    continue
                if result[j].get("label") != current.get("label"):
                    continue
                if _iou(current["bbox"], result[j]["bbox"]) >= iou_thresh:
                    bj = result[j]["bbox"]
                    current["bbox"] = [
                        min(current["bbox"][0], bj[0]),
                        min(current["bbox"][1], bj[1]),
                        max(current["bbox"][2], bj[2]),
                        max(current["bbox"][3], bj[3]),
                    ]
                    current["confidence"] = max(
                        current["confidence"], result[j]["confidence"]
                    )
                    used.add(j)
                    merged = True
            new_result.append(current)
        result = new_result
    return result


def postprocess(
    detections: List[Dict[str, Any]],
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    """Full postprocessing pipeline."""
    dets = clip_boxes(detections, width, height)
    dets = filter_small_boxes(dets)
    dets = nms(dets)
    dets = merge_same_class(dets)
    dets = clip_boxes(dets, width, height)
    return dets
