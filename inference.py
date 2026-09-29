#!/usr/bin/env python3
"""CLI entry point for Manuscript Layout Region Detection.

Usage examples::

    python inference.py --input ./data/test_images --output ./results
    python inference.py --input page.jpg --backend classical --save-debug
    python inference.py --input ./scans --weights best.pt --device cuda

"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from src.classifier import classify_regions
from src.config import CLASS_NAMES, DEFAULT_CONF_THRESH, DEFAULT_OUTPUT_DIR, RANDOM_SEED
from src.detector import create_detector
from src.io_utils import (
    build_image_result,
    find_images,
    safe_read_image,
    write_json,
    write_summary,
)
from src.postprocess import postprocess
from src.preprocess import map_boxes_to_original, preprocess_image
from src.visualize import save_annotated

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Manuscript Layout Region Detection — inference pipeline.",
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to a single image or a folder of images.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output folder for annotated images and JSON (default: ./results).",
    )
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="Path to custom YOLO model weights (.pt).",
    )
    parser.add_argument(
        "--backend", choices=["yolo", "classical"], default="yolo",
        help="Detector backend (default: yolo).",
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF_THRESH,
        help=f"Confidence threshold (default: {DEFAULT_CONF_THRESH}).",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "auto"], default="auto",
        help="Compute device (default: auto).",
    )
    parser.add_argument(
        "--no-preprocess", action="store_true",
        help="Skip the preprocessing pipeline.",
    )
    parser.add_argument(
        "--save-debug", action="store_true",
        help="Save preprocessed images to the output folder.",
    )
    return parser


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_single_image(
    image_path: Path,
    detector: Any,
    output_dir: Path,
    *,
    skip_preprocess: bool = False,
    save_debug: bool = False,
) -> Dict[str, Any] | None:
    """Run the full pipeline on one image.

    Returns the per-image result dict, or ``None`` on failure.
    """
    logger.info("Processing: %s", image_path.name)

    # 1. Read
    image = safe_read_image(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]

    # 2. Preprocess
    inv_mat = None
    if skip_preprocess:
        proc_image = image
    else:
        proc_image, inv_mat = preprocess_image(image)
        if save_debug:
            debug_path = output_dir / f"{image_path.stem}_preprocessed.jpg"
            import cv2
            cv2.imwrite(str(debug_path), proc_image)
            logger.info("Debug → %s", debug_path)

    # 3. Detect
    detections = detector.predict(proc_image)

    # 4. Map boxes back to original coords (if deskewed)
    if inv_mat is not None and detections:
        boxes = np.array([d["bbox"] for d in detections])
        boxes = map_boxes_to_original(boxes, inv_mat)
        for det, box in zip(detections, boxes):
            det["bbox"] = box.tolist()

    # 5. Classify unlabelled detections
    need_classifier = not detector.is_custom_trained or any(
        d.get("label") is None for d in detections
    )
    if need_classifier:
        detections = classify_regions(detections, w, h, image)

    # 6. Post-process
    detections = postprocess(detections, w, h)

    # 7. Build region list
    regions: List[Dict[str, Any]] = []
    for det in detections:
        regions.append({
            "label": det["label"],
            "bbox": [round(v, 1) for v in det["bbox"]],
            "confidence": round(det["confidence"], 4),
        })

    # 8. Save annotated image
    ann_path = output_dir / f"{image_path.stem}_annotated.jpg"
    save_annotated(image, regions, ann_path)

    # 9. Save per-image JSON
    result = build_image_result(image_path.name, w, h, regions)
    json_path = output_dir / f"{image_path.stem}.json"
    write_json(result, json_path)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Discover images
    images = find_images(args.input)
    if not images:
        logger.error("No valid images found at %s", args.input)
        sys.exit(1)

    # Create output dir
    args.output.mkdir(parents=True, exist_ok=True)

    # Load detector ONCE
    detector = create_detector(
        backend=args.backend,
        weights=args.weights,
        device=args.device,
        conf=args.conf,
    )

    all_results: List[Dict[str, Any]] = []
    failed = 0
    times: List[float] = []

    for img_path in tqdm(images, desc="Inference", unit="img"):
        t0 = time.perf_counter()
        try:
            result = process_single_image(
                img_path,
                detector,
                args.output,
                skip_preprocess=args.no_preprocess,
                save_debug=args.save_debug,
            )
        except Exception as exc:
            logger.error("Unhandled error on %s: %s", img_path.name, exc)
            result = None

        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if result is None:
            failed += 1
        else:
            all_results.append(result)

    # Write combined summary
    write_summary(all_results, args.output)

    # Final report
    processed = len(all_results)
    total = processed + failed
    avg_time = sum(times) / len(times) if times else 0.0
    logger.info("=" * 60)
    logger.info("Done.  Processed: %d / %d  |  Failed: %d", processed, total, failed)
    logger.info("Avg time per image: %.2f s", avg_time)
    logger.info("Results written to: %s", args.output.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
