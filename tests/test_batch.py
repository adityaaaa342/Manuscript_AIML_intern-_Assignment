"""Tests for batch inference (folder of images)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestBatchRun:
    """Run inference on a folder and verify outputs exist."""

    def test_batch_produces_outputs(self, sample_image_dir: Path, output_dir: Path) -> None:
        """Running on a folder creates annotated images + JSON for valid files."""
        from inference import main

        main([
            "--input", str(sample_image_dir),
            "--output", str(output_dir),
            "--backend", "classical",
            "--no-preprocess",
        ])

        # Should have summary.json
        summary_path = output_dir / "summary.json"
        assert summary_path.exists(), "summary.json not found"

        with open(summary_path) as f:
            summary = json.load(f)

        # 3 valid images (corrupt.jpg should fail gracefully)
        assert summary["total_images"] >= 2  # at least 2 of 3 should succeed

        # Each processed image should have _annotated.jpg and .json
        for result in summary["results"]:
            stem = Path(result["image"]).stem
            assert (output_dir / f"{stem}_annotated.jpg").exists()
            assert (output_dir / f"{stem}.json").exists()

    def test_single_image_works(self, sample_image: Path, output_dir: Path) -> None:
        """Running on a single file produces output."""
        from inference import main

        main([
            "--input", str(sample_image),
            "--output", str(output_dir),
            "--backend", "classical",
            "--no-preprocess",
        ])

        summary = json.loads((output_dir / "summary.json").read_text())
        assert summary["total_images"] == 1
