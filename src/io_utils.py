"""I/O utilities: image discovery, safe reading, JSON writing.

All file-system interaction is centralised here so the rest of the pipeline
never deals with raw ``os`` calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from src.config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def find_images(input_path: Path) -> List[Path]:
    """Return a sorted list of image paths from *input_path*.

    Parameters
    ----------
    input_path : Path
        A single image file **or** a directory containing images.

    Returns
    -------
    list[Path]
        Sorted list of discovered image paths.
    """
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [input_path]
        logger.warning("File %s has unsupported extension.", input_path)
        return []

    if input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        logger.info("Found %d image(s) in %s", len(images), input_path)
        return images

    logger.error("Input path does not exist: %s", input_path)
    return []


# ---------------------------------------------------------------------------
# Safe image reading
# ---------------------------------------------------------------------------

def safe_read_image(path: Path) -> Optional[np.ndarray]:
    """Read an image via OpenCV; return ``None`` on failure.

    Parameters
    ----------
    path : Path
        Path to the image file.

    Returns
    -------
    np.ndarray | None
        BGR image array, or ``None`` if the file is corrupt / unreadable.
    """
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2.imread returned None for {path}")
        return img
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# File integrity
# ---------------------------------------------------------------------------

def file_md5(path: Path) -> str:
    """Return the hex MD5 digest of *path* (used in tests)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def build_image_result(
    filename: str,
    width: int,
    height: int,
    regions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the per-image JSON result dictionary.

    Parameters
    ----------
    filename : str
        Original image file name (basename).
    width, height : int
        Original image dimensions.
    regions : list[dict]
        List of region dicts, each with *label*, *bbox*, *confidence*.

    Returns
    -------
    dict
        Conformant result dictionary.
    """
    return {
        "image": filename,
        "width": width,
        "height": height,
        "regions": regions,
    }


def write_json(data: Any, path: Path) -> None:
    """Serialise *data* as indented JSON to *path*.

    Parent directories are created if they don't exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote JSON → %s", path)


def write_summary(
    all_results: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write a combined ``summary.json`` for the entire batch run.

    Parameters
    ----------
    all_results : sequence[dict]
        Per-image result dicts.
    output_dir : Path
        Target folder.
    """
    summary = {
        "total_images": len(all_results),
        "results": list(all_results),
    }
    write_json(summary, output_dir / "summary.json")
