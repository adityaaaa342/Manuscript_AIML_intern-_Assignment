"""Tests for corrupt-image handling and input-file integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.io_utils import file_md5, safe_read_image


class TestCorruptImage:
    """Corrupt images must not crash the batch."""

    def test_safe_read_returns_none(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "bad.jpg"
        corrupt.write_bytes(b"THIS IS NOT AN IMAGE")
        assert safe_read_image(corrupt) is None

    def test_batch_survives_corrupt(self, sample_image_dir: Path, output_dir: Path) -> None:
        """Batch with a corrupt file still produces a summary without crashing."""
        from inference import main

        # Should NOT raise
        main([
            "--input", str(sample_image_dir),
            "--output", str(output_dir),
            "--backend", "classical",
            "--no-preprocess",
        ])

        summary = json.loads((output_dir / "summary.json").read_text())
        # At least some images processed
        assert summary["total_images"] >= 1


class TestInputIntegrity:
    """Original input files must never be modified."""

    def test_input_files_unchanged(self, sample_image_dir: Path, output_dir: Path) -> None:
        # Compute hashes BEFORE
        hashes_before = {}
        for p in sorted(sample_image_dir.iterdir()):
            hashes_before[p.name] = file_md5(p)

        from inference import main
        main([
            "--input", str(sample_image_dir),
            "--output", str(output_dir),
            "--backend", "classical",
            "--no-preprocess",
        ])

        # Compute hashes AFTER
        for p in sorted(sample_image_dir.iterdir()):
            assert file_md5(p) == hashes_before[p.name], (
                f"Input file {p.name} was modified!"
            )
