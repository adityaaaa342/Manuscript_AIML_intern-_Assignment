"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixture: temporary directory with test images
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    """Create a small synthetic image and return its path."""
    img = np.full((600, 400, 3), 200, dtype=np.uint8)
    # Draw some "text" lines
    for y in range(50, 550, 15):
        cv2.line(img, (40, y), (360, y), (40, 40, 40), 1)
    path = tmp_path / "sample.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def sample_image_dir(tmp_path: Path) -> Path:
    """Create a directory with 3 valid images + 1 corrupt file."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    rng = np.random.RandomState(0)
    for i in range(3):
        img = rng.randint(100, 220, (400, 300, 3), dtype=np.uint8)
        cv2.line(img, (30, 50), (270, 50), (30, 30, 30), 2)
        cv2.line(img, (30, 350), (270, 350), (30, 30, 30), 2)
        cv2.imwrite(str(img_dir / f"page_{i:02d}.jpg"), img)

    # Corrupt file
    (img_dir / "corrupt.jpg").write_bytes(b"GARBAGE")

    return img_dir


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Return a clean output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out
