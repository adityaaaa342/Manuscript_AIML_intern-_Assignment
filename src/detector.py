"""Region detector backends: YOLO and classical (fallback).

Both backends return a list of raw detections — each a dict with
bbox ([x1, y1, x2, y2]), confidence, and optionally label.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.config import (
    CLASS_NAMES,
    DEFAULT_CONF_THRESH,
    MIN_BOX_AREA_PX,
    YOLO_DEFAULT_MODEL,
    YOLO_IMG_SIZE,
)

logger = logging.getLogger(__name__)

Detection = Dict[str, Any]


class YOLODetector:
    """Ultralytics YOLO detector wrapper with automatic classical fallback."""

    def __init__(
        self,
        weights: Optional[Path] = None,
        device: str = "auto",
        conf: float = DEFAULT_CONF_THRESH,
    ) -> None:
        from ultralytics import YOLO

        self._conf = conf
        self._device = self._resolve_device(device)
        self._custom_trained = False
        self._fallback_detector = ClassicalDetector(conf=conf)

        weights_path = Path(weights) if weights else None
        if weights_path and weights_path.is_file():
            logger.info("Loading custom YOLO weights: %s", weights_path)
            self._model = YOLO(str(weights_path))
            model_names = getattr(self._model, "names", {})
            if isinstance(model_names, dict):
                model_class_names = set(model_names.values())
            else:
                model_class_names = set(model_names)
            if set(CLASS_NAMES).issubset(model_class_names):
                self._custom_trained = True
                logger.info("Custom model has all 5 target classes.")
            else:
                logger.info("Custom model classes %s — will use rule-based classifier.", model_class_names)
        else:
            logger.info(
                "No custom weights provided — loaded base YOLO model with classical region fallback."
            )
            self._model = YOLO(YOLO_DEFAULT_MODEL)

    @property
    def is_custom_trained(self) -> bool:
        return self._custom_trained

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def predict(self, image: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            source=image,
            imgsz=YOLO_IMG_SIZE,
            conf=self._conf,
            device=self._device,
            verbose=False,
        )
        detections: List[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                model_names = self._model.names
                label: Optional[str] = None
                if isinstance(model_names, dict):
                    raw_label = model_names.get(cls_id, None)
                elif isinstance(model_names, (list, tuple)):
                    raw_label = model_names[cls_id] if cls_id < len(model_names) else None
                else:
                    raw_label = None

                if self._custom_trained and raw_label in CLASS_NAMES:
                    label = raw_label

                detections.append({
                    "bbox": xyxy,
                    "confidence": conf,
                    "label": label,
                })

        # If base generic YOLO detects 0 boxes on manuscript, fallback to classical text-region detection
        if not detections and not self._custom_trained:
            logger.debug("Base YOLO returned 0 detections on manuscript scan — invoking classical region fallback.")
            detections = self._fallback_detector.predict(image)

        return detections


class ClassicalDetector:
    """Manuscript layout detector with aligned line-by-line segmentation and header/footer detection."""

    def __init__(self, conf: float = DEFAULT_CONF_THRESH) -> None:
        self._conf = conf

    @property
    def is_custom_trained(self) -> bool:
        return False

    def predict(self, image: np.ndarray) -> List[Detection]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Detect manuscript leaf boundary
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        _, leaf_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(leaf_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        has_isolated_leaf = False
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if (w * h * 0.20) <= area <= (w * h * 0.92):
                lx, ly, lw, lh = cv2.boundingRect(largest)
                has_isolated_leaf = True
            else:
                lx, ly, lw, lh = 0, 0, w, h
        else:
            lx, ly, lw, lh = 0, 0, w, h

        detections: List[Detection] = []

        # 2. Top Header Banner
        top_limit = ly if has_isolated_leaf and ly > 15 else int(h * 0.12)
        top_zone = gray[0:top_limit, :]
        th_top = cv2.adaptiveThreshold(
            top_zone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
        )
        n, l, stats, _ = cv2.connectedComponentsWithStats(th_top)
        top_comps = [
            stats[i]
            for i in range(1, n)
            if 8 <= stats[i][4] <= 3000
            and 5 <= stats[i][3] <= top_limit
            and stats[i][2] <= int(w * 0.70)
            and stats[i][0] > 5
            and (stats[i][0] + stats[i][2]) < (w - 5)
        ]
        header_bottom = 0
        if top_comps:
            tx1 = float(max(0, min(c[0] for c in top_comps) - 8))
            ty1 = float(max(0, min(c[1] for c in top_comps) - 4))
            tx2 = float(min(w, max(c[0] + c[2] for c in top_comps) + 8))
            ty2 = float(min(top_limit, max(c[1] + c[3] for c in top_comps) + 4))
            if (tx2 - tx1) > 60:
                header_bottom = int(ty2)
                detections.append({
                    "label": "header",
                    "bbox": [tx1, ty1, tx2, ty2],
                    "confidence": 0.98,
                })

        # 3. Bottom Footer Banner
        bot_start = (ly + lh) if has_isolated_leaf and (ly + lh) < (h - 15) else int(h * 0.88)
        bot_zone = gray[bot_start:h, :]
        th_bot = cv2.adaptiveThreshold(
            bot_zone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
        )
        n, l, stats, _ = cv2.connectedComponentsWithStats(th_bot)
        bot_comps = [
            stats[i]
            for i in range(1, n)
            if 8 <= stats[i][4] <= 3000
            and 5 <= stats[i][3] <= (h - bot_start)
            and stats[i][2] <= int(w * 0.70)
            and stats[i][0] > 5
            and (stats[i][0] + stats[i][2]) < (w - 5)
        ]
        footer_top = h
        if bot_comps:
            bx1 = float(max(0, min(c[0] for c in bot_comps) - 8))
            by1 = float(max(bot_start, min(c[1] for c in bot_comps) + bot_start - 4))
            bx2 = float(min(w, max(c[0] + c[2] for c in bot_comps) + 8))
            by2 = float(min(h, max(c[1] + c[3] for c in bot_comps) + bot_start + 4))
            if (bx2 - bx1) > 60:
                footer_top = int(by1)
                detections.append({
                    "label": "footer",
                    "bbox": [bx1, by1, bx2, by2],
                    "confidence": 0.98,
                })

        # 4. Main Manuscript Text Lines (on the Leaf/Page)
        if has_isolated_leaf:
            mx1 = int(lx + lw * 0.01)
            my1 = int(ly + lh * 0.03)
            mx2 = int(lx + lw * 0.99)
            my2 = int(ly + lh * 0.97)
        else:
            mx1 = int(w * 0.01)
            my1 = max(header_bottom + 4, int(h * 0.12))
            mx2 = int(w * 0.99)
            my2 = min(footer_top - 4, int(h * 0.88))

        leaf_crop = gray[my1:my2, mx1:mx2]
        lh_c, lw_c = leaf_crop.shape[:2]

        blur_leaf = cv2.GaussianBlur(leaf_crop, (3, 3), 0)
        th = cv2.adaptiveThreshold(
            blur_leaf, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )

        num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(th)
        th_clean = np.zeros_like(th)
        all_x = []
        for i in range(1, num_labels):
            x, y, cw, ch, ca = stats[i]
            if 8 <= ca <= (lw_c * lh_c * 0.10) and cw < (lw_c * 0.9) and ch < (lh_c * 0.35):
                th_clean[labels_map == i] = 255
                all_x.append(x)
                all_x.append(x + cw)

        # Consistent aligned left and right margins across the manuscript body
        if all_x:
            body_x1 = float(mx1 + max(0, min(all_x) - 4))
            body_x2 = float(mx1 + min(lw_c, max(all_x) + 4))
        else:
            body_x1 = float(mx1)
            body_x2 = float(mx2)

        # Horizontal projection for line valleys
        proj = np.sum(th_clean > 0, axis=1).astype(np.float32)
        ksize = max(11, (lh_c // 20) | 1)
        kernel = cv2.getGaussianKernel(ksize, 3).flatten()
        proj_smooth = np.convolve(proj, kernel, mode="same")

        valleys = []
        for y in range(1, lh_c - 1):
            if proj_smooth[y] <= proj_smooth[y - 1] and proj_smooth[y] <= proj_smooth[y + 1]:
                valleys.append(y)

        min_line_h = max(18, lh_c // 16)
        boundaries = [0] + [v for v in valleys if min_line_h <= v <= (lh_c - min_line_h)] + [lh_c]
        clean_boundaries = [boundaries[0]]
        for b in boundaries[1:]:
            if b - clean_boundaries[-1] >= min_line_h:
                clean_boundaries.append(b)
            else:
                clean_boundaries[-1] = (clean_boundaries[-1] + b) // 2
        if clean_boundaries[-1] != lh_c:
            clean_boundaries.append(lh_c)

        num_lines = len(clean_boundaries) - 1
        for i in range(num_lines):
            sy1 = clean_boundaries[i]
            sy2 = clean_boundaries[i + 1]
            strip = th_clean[sy1:sy2, :]
            pts = cv2.findNonZero(strip)
            if pts is not None:
                _, y_rel, _, h_rel = cv2.boundingRect(pts)
                if h_rel >= 6:
                    by1 = float(my1 + sy1 + y_rel)
                    by2 = float(my1 + sy1 + y_rel + h_rel)
                    detections.append({
                        "label": "main_text",
                        "bbox": [body_x1, by1, body_x2, by2],
                        "confidence": 0.98,
                    })

        logger.debug("Classical detector produced %d detections.", len(detections))
        return detections


def create_detector(
    backend: str = "yolo",
    weights: Optional[Path] = None,
    device: str = "auto",
    conf: float = DEFAULT_CONF_THRESH,
) -> YOLODetector | ClassicalDetector:
    if backend == "classical":
        logger.info("Using classical (threshold + contour) detector.")
        return ClassicalDetector(conf=conf)
    logger.info("Using YOLO detector.")
    return YOLODetector(weights=weights, device=device, conf=conf)
