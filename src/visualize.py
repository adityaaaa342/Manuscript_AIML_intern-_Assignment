"""Visualization: draw annotated bounding boxes on manuscript images.

Each class gets a distinct color (from :mod:`src.config`).  Text labels
include the class name and confidence, rendered with a filled background
for readability on both light and dark manuscripts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from src.config import CLASS_COLORS

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICKNESS = 1
_BOX_THICKNESS = 2


def draw_regions(
    image: np.ndarray,
    regions: List[Dict[str, Any]],
) -> np.ndarray:
    """Draw coloured bounding boxes with labels on a **copy** of *image*.

    Parameters
    ----------
    image : np.ndarray
        BGR image (not modified).
    regions : list[dict]
        Each dict has ``label``, ``bbox`` ``[x1,y1,x2,y2]``, ``confidence``.

    Returns
    -------
    np.ndarray
        Annotated BGR image.
    """
    canvas = image.copy()
    for region in regions:
        label = region["label"]
        conf = region["confidence"]
        x1, y1, x2, y2 = [int(v) for v in region["bbox"]]
        color = CLASS_COLORS.get(label, (200, 200, 200))

        # Draw bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, _BOX_THICKNESS)

        # Label text
        text = f"{label} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            text, _FONT, _FONT_SCALE, _FONT_THICKNESS
        )

        # Position label above the box; if no room, put it inside
        label_y = y1 - 6
        if label_y - th < 0:
            label_y = y1 + th + 6

        # Background rectangle for readability
        cv2.rectangle(
            canvas,
            (x1, label_y - th - 4),
            (x1 + tw + 4, label_y + 4),
            color,
            cv2.FILLED,
        )
        # Use white or black text depending on colour brightness
        brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        cv2.putText(
            canvas, text, (x1 + 2, label_y),
            _FONT, _FONT_SCALE, text_color, _FONT_THICKNESS, cv2.LINE_AA,
        )

    return canvas


def save_annotated(
    image: np.ndarray,
    regions: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Draw regions and save to *output_path*.

    Parameters
    ----------
    image : np.ndarray
        Original BGR image.
    regions : list[dict]
        Detection results.
    output_path : Path
        Destination ``.jpg`` file.
    """
    annotated = draw_regions(image, regions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    logger.info("Saved annotated image → %s", output_path)
