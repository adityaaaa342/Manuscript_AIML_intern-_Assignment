"""Tests for JSON schema validity."""

from __future__ import annotations

import json

import pytest

from src.config import CLASS_NAMES
from src.io_utils import build_image_result


class TestJsonSchema:
    """Ensure the per-image JSON output conforms to the required schema."""

    def _make_result(self) -> dict:
        return build_image_result(
            filename="test.jpg",
            width=800,
            height=1200,
            regions=[
                {"label": "header", "bbox": [10.0, 5.0, 790.0, 100.0], "confidence": 0.92},
                {"label": "main_text", "bbox": [50.0, 120.0, 750.0, 1050.0], "confidence": 0.88},
                {"label": "footer", "bbox": [100.0, 1100.0, 700.0, 1190.0], "confidence": 0.75},
            ],
        )

    def test_top_level_keys(self) -> None:
        result = self._make_result()
        assert "image" in result
        assert "width" in result
        assert "height" in result
        assert "regions" in result

    def test_dimensions_are_ints(self) -> None:
        result = self._make_result()
        assert isinstance(result["width"], int)
        assert isinstance(result["height"], int)

    def test_regions_is_list(self) -> None:
        result = self._make_result()
        assert isinstance(result["regions"], list)

    def test_region_keys(self) -> None:
        result = self._make_result()
        for region in result["regions"]:
            assert "label" in region
            assert "bbox" in region
            assert "confidence" in region

    def test_label_is_valid_class(self) -> None:
        result = self._make_result()
        for region in result["regions"]:
            assert region["label"] in CLASS_NAMES

    def test_bbox_format(self) -> None:
        result = self._make_result()
        for region in result["regions"]:
            bbox = region["bbox"]
            assert isinstance(bbox, list)
            assert len(bbox) == 4
            assert all(isinstance(v, (int, float)) for v in bbox)
            assert bbox[0] <= bbox[2]  # x_min <= x_max
            assert bbox[1] <= bbox[3]  # y_min <= y_max

    def test_confidence_range(self) -> None:
        result = self._make_result()
        for region in result["regions"]:
            assert 0.0 <= region["confidence"] <= 1.0

    def test_json_serialisable(self) -> None:
        result = self._make_result()
        # Must not raise
        serialised = json.dumps(result)
        reloaded = json.loads(serialised)
        assert reloaded == result
