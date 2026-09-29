"""Tests for bounding-box clipping and postprocessing."""

from __future__ import annotations

import pytest

from src.postprocess import clip_boxes, filter_small_boxes, nms, postprocess, _iou


class TestClipBoxes:
    """Verify bbox clipping to image boundaries."""

    def test_boxes_inside_image_unchanged(self) -> None:
        dets = [{"bbox": [10.0, 20.0, 100.0, 200.0], "confidence": 0.9, "label": "main_text"}]
        result = clip_boxes(dets, 400, 600)
        assert result[0]["bbox"] == [10.0, 20.0, 100.0, 200.0]

    def test_negative_coords_clipped_to_zero(self) -> None:
        dets = [{"bbox": [-50.0, -30.0, 100.0, 200.0], "confidence": 0.8, "label": "header"}]
        result = clip_boxes(dets, 400, 600)
        assert result[0]["bbox"][0] == 0.0
        assert result[0]["bbox"][1] == 0.0

    def test_coords_beyond_image_clipped(self) -> None:
        dets = [{"bbox": [10.0, 20.0, 500.0, 700.0], "confidence": 0.7, "label": "footer"}]
        result = clip_boxes(dets, 400, 600)
        assert result[0]["bbox"][2] == 400.0
        assert result[0]["bbox"][3] == 600.0

    def test_all_negative_becomes_zero_area(self) -> None:
        dets = [{"bbox": [-100.0, -100.0, -50.0, -50.0], "confidence": 0.5, "label": "filler"}]
        result = clip_boxes(dets, 400, 600)
        assert all(v == 0.0 for v in result[0]["bbox"])


class TestFilterSmall:
    def test_tiny_boxes_removed(self) -> None:
        dets = [
            {"bbox": [0, 0, 5, 5], "confidence": 0.9, "label": "a"},      # 25 px
            {"bbox": [0, 0, 100, 100], "confidence": 0.9, "label": "b"},  # 10000 px
        ]
        result = filter_small_boxes(dets, min_area=100)
        assert len(result) == 1
        assert result[0]["label"] == "b"


class TestNMS:
    def test_no_overlap_keeps_all(self) -> None:
        dets = [
            {"bbox": [0, 0, 50, 50], "confidence": 0.9, "label": "a"},
            {"bbox": [200, 200, 250, 250], "confidence": 0.8, "label": "b"},
        ]
        assert len(nms(dets, iou_thresh=0.5)) == 2

    def test_high_overlap_suppresses_lower(self) -> None:
        dets = [
            {"bbox": [0, 0, 100, 100], "confidence": 0.9, "label": "a"},
            {"bbox": [5, 5, 105, 105], "confidence": 0.7, "label": "b"},
        ]
        result = nms(dets, iou_thresh=0.5)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9


class TestIoU:
    def test_identical_boxes(self) -> None:
        assert _iou([0, 0, 100, 100], [0, 0, 100, 100]) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert _iou([0, 0, 50, 50], [100, 100, 200, 200]) == 0.0

    def test_partial_overlap(self) -> None:
        iou = _iou([0, 0, 100, 100], [50, 50, 150, 150])
        assert 0.0 < iou < 1.0
