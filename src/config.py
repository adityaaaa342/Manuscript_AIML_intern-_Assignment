"""Central configuration for class names, colors, thresholds, and paths.

All tunable parameters live here so the rest of the codebase stays free of
magic numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Class definitions (order matters — index is used in YOLO label files)
# ---------------------------------------------------------------------------
CLASS_NAMES: List[str] = [
    "header",
    "footer",
    "main_text",
    "side_text",
    "filler",
]

NUM_CLASSES: int = len(CLASS_NAMES)

# BGR colors for visualization (distinct, readable on light *and* dark BGs)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "header":    (0, 200, 255),   # orange
    "footer":    (255, 100, 0),   # blue
    "main_text": (0, 255, 100),   # green
    "side_text": (200, 0, 255),   # magenta
    "filler":    (128, 128, 255), # salmon / light-red
}

# ---------------------------------------------------------------------------
# Classification thresholds (fraction of page dimension)
# ---------------------------------------------------------------------------
HEADER_STRIP_RATIO: float = 0.12       # top 12 % of the page
FOOTER_STRIP_RATIO: float = 0.12       # bottom 12 %
SIDE_TEXT_WIDTH_RATIO: float = 0.20     # region narrower than 20 % of page W
SIDE_TEXT_EDGE_RATIO: float = 0.15      # center within 15 % of page edge
MIN_MAIN_TEXT_AREA_RATIO: float = 0.05  # at least 5 % of page area

# Filler heuristics
FILLER_MAX_TEXT_DENSITY: float = 0.15   # low text-stroke density
FILLER_MAX_AREA_RATIO: float = 0.08    # small decorative blobs

# ---------------------------------------------------------------------------
# Detection & post-processing
# ---------------------------------------------------------------------------
DEFAULT_CONF_THRESH: float = 0.25
NMS_IOU_THRESH: float = 0.45
MERGE_IOU_THRESH: float = 0.60         # merge same-class boxes above this
MIN_BOX_AREA_PX: int = 400             # remove tiny boxes (< 20×20)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID: Tuple[int, int] = (8, 8)
DENOISE_H: int = 10                     # fastNlMeans strength
ILLUMINATION_KERNEL_SIZE: int = 51      # Gaussian blur for bg normalization

# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp",
)

# ---------------------------------------------------------------------------
# Default paths (all relative – no absolute paths)
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR: Path = Path("results")

# ---------------------------------------------------------------------------
# YOLO / training defaults
# ---------------------------------------------------------------------------
YOLO_DEFAULT_MODEL: str = "yolov8n.pt"  # pretrained nano model (fallback)
YOLO_IMG_SIZE: int = 1024
TRAIN_EPOCHS: int = 100
TRAIN_BATCH: int = 16

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
