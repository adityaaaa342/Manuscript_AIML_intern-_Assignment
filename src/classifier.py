"""Rule-based classifier that assigns one of the 5 layout labels.

Used when detections lack pre-assigned labels.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import cv2
import numpy as np

from src.config import (
    FOOTER_STRIP_RATIO,
    HEADER_STRIP_RATIO,
    SIDE_TEXT_EDGE_RATIO,
    SIDE_TEXT_WIDTH_RATIO,
)

logger = logging.getLogger(__name__)


def classify_region(
    bbox: List[float],
    page_w: int,
    page_h: int,
    image: np.ndarray,
) -> str:
    """Assign a layout label to a single region based on geometry and position."""
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    center_y = (y1 + y2) / 2.0
    center_x = (x1 + x2) / 2.0

    # Header: top strip of page
    if center_y < page_h * HEADER_STRIP_RATIO:
        return "header"

    # Footer: bottom strip of page
    if center_y > page_h * (1.0 - FOOTER_STRIP_RATIO):
        return "footer"

    # Side text: narrow margin column on left or right edge
    is_narrow = box_w < page_w * SIDE_TEXT_WIDTH_RATIO
    near_left = center_x < page_w * SIDE_TEXT_EDGE_RATIO
    near_right = center_x > page_w * (1.0 - SIDE_TEXT_EDGE_RATIO)
    if is_narrow and (near_left or near_right):
        return "side_text"

    # Main text: default for body content
    return "main_text"


def classify_regions(
    detections: List[Dict[str, Any]],
    page_w: int,
    page_h: int,
    image: np.ndarray,
    force: bool = False,
) -> List[Dict[str, Any]]:
    for det in detections:
        if force or det.get("label") is None:
            det["label"] = classify_region(
                det["bbox"], page_w, page_h, image
            )
    return detections
