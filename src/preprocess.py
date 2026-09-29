"""Preprocessing pipeline for manuscript images.

Each function operates on a BGR (OpenCV-format) image and returns a new array.
Coordinates are always in the **original** image space; any geometric
transforms (e.g. deskew) produce an inverse mapping so downstream bounding
boxes can be projected back.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    DENOISE_H,
    ILLUMINATION_KERNEL_SIZE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLAHE contrast enhancement
# ---------------------------------------------------------------------------

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement for faded ink.

    Parameters
    ----------
    image : np.ndarray
        BGR input image.

    Returns
    -------
    np.ndarray
        Contrast-enhanced BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID
    )
    l_chan = clahe.apply(l_chan)
    merged = cv2.merge([l_chan, a_chan, b_chan])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Illumination / background normalization
# ---------------------------------------------------------------------------

def normalize_illumination(image: np.ndarray) -> np.ndarray:
    """Remove uneven lighting by dividing by a large-kernel Gaussian blur.

    Parameters
    ----------
    image : np.ndarray
        BGR input.

    Returns
    -------
    np.ndarray
        Illumination-normalised BGR image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ksize = ILLUMINATION_KERNEL_SIZE
    # Ensure kernel is odd
    if ksize % 2 == 0:
        ksize += 1
    bg = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    # Avoid division by zero
    bg = np.clip(bg, 1.0, None)
    normed = (gray / bg * 128).clip(0, 255).astype(np.uint8)
    # Convert back to 3-channel for consistency
    return cv2.cvtColor(normed, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

def denoise(image: np.ndarray) -> np.ndarray:
    """Light denoising with ``fastNlMeansDenoisingColored``.

    Parameters
    ----------
    image : np.ndarray
        BGR input.

    Returns
    -------
    np.ndarray
        Denoised BGR image.
    """
    return cv2.fastNlMeansDenoisingColored(
        image, None, DENOISE_H, DENOISE_H, 7, 21
    )


# ---------------------------------------------------------------------------
# Deskew
# ---------------------------------------------------------------------------

def estimate_skew_angle(image: np.ndarray) -> float:
    """Estimate page skew in degrees using minAreaRect on text contours.

    Parameters
    ----------
    image : np.ndarray
        BGR input.

    Returns
    -------
    float
        Estimated skew angle in degrees (positive = counter-clockwise).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4,
    )
    # dilate to connect text into blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0

    # Use the largest contour's minAreaRect
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[-1]

    # Normalise to [-45, 45]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    return angle


def deskew(
    image: np.ndarray,
    angle: Optional[float] = None,
) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """Deskew *image* by the given angle (or auto-estimated).

    Parameters
    ----------
    image : np.ndarray
        BGR input.
    angle : float | None
        Skew angle in degrees.  Auto-estimated if ``None``.

    Returns
    -------
    tuple[np.ndarray, float, np.ndarray | None]
        ``(deskewed_image, angle_used, inverse_matrix)``
        *inverse_matrix* maps points in the deskewed image back to the
        original coordinate system (``None`` if angle ≈ 0).
    """
    if angle is None:
        angle = estimate_skew_angle(image)

    if abs(angle) < 0.3:
        logger.debug("Skew angle %.2f° is negligible — skipping deskew.", angle)
        return image.copy(), angle, None

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rot_mat, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # Build inverse matrix for coordinate mapping
    inv_mat = cv2.invertAffineTransform(rot_mat)
    logger.info("Deskewed by %.2f°", angle)
    return rotated, angle, inv_mat


# ---------------------------------------------------------------------------
# Coordinate mapping helpers
# ---------------------------------------------------------------------------

def map_boxes_to_original(
    boxes: np.ndarray,
    inv_matrix: Optional[np.ndarray],
) -> np.ndarray:
    """Transform ``[x1, y1, x2, y2]`` boxes back to original coords.

    Parameters
    ----------
    boxes : np.ndarray
        Shape ``(N, 4)`` — ``[x_min, y_min, x_max, y_max]``.
    inv_matrix : np.ndarray | None
        Inverse affine matrix from :func:`deskew`.  If ``None`` the boxes
        are returned unchanged.

    Returns
    -------
    np.ndarray
        Transformed boxes, same shape.
    """
    if inv_matrix is None or len(boxes) == 0:
        return boxes.copy()

    out = boxes.copy().astype(np.float64)
    corners = np.array([
        [out[:, 0], out[:, 1]],  # top-left
        [out[:, 2], out[:, 1]],  # top-right
        [out[:, 2], out[:, 3]],  # bottom-right
        [out[:, 0], out[:, 3]],  # bottom-left
    ])  # shape (4, 2, N)

    transformed = []
    for c in range(4):
        pts = np.vstack([corners[c], np.ones((1, len(out)))])  # (3, N)
        mapped = inv_matrix @ pts  # (2, N)
        transformed.append(mapped)

    xs = np.array([t[0] for t in transformed])  # (4, N)
    ys = np.array([t[1] for t in transformed])  # (4, N)
    out[:, 0] = xs.min(axis=0)
    out[:, 1] = ys.min(axis=0)
    out[:, 2] = xs.max(axis=0)
    out[:, 3] = ys.max(axis=0)
    return out


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_image(
    image: np.ndarray,
    do_deskew: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    image : np.ndarray
        BGR input (unmodified).
    do_deskew : bool
        Whether to auto-deskew.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        ``(preprocessed_image, inverse_deskew_matrix)``
    """
    result = apply_clahe(image)
    result = normalize_illumination(result)
    result = denoise(result)

    inv_mat: Optional[np.ndarray] = None
    if do_deskew:
        result, _angle, inv_mat = deskew(result)

    return result, inv_mat
