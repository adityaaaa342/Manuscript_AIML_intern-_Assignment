#!/usr/bin/env python3
"""Fine-tuning script for YOLO on a 5-class manuscript layout dataset.

Usage::

    python train.py --data data.yaml --epochs 100 --batch 16 --device auto

This script wraps the Ultralytics YOLO training API with sensible defaults
and strong augmentation for manuscript images.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    RANDOM_SEED,
    TRAIN_BATCH,
    TRAIN_EPOCHS,
    YOLO_DEFAULT_MODEL,
    YOLO_IMG_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# data.yaml template
# ---------------------------------------------------------------------------

DATA_YAML_TEMPLATE = """\
# YOLO dataset configuration
# Place this file at the root of your dataset folder.
#
# Folder structure:
#   dataset/
#     images/
#       train/   ← training images
#       val/     ← validation images
#     labels/
#       train/   ← YOLO-format .txt label files
#       val/
#
# Each .txt label file has one line per object:
#   <class_id> <x_center> <y_center> <width> <height>
# All values are normalised to [0, 1].

path: ./dataset          # dataset root dir (edit this)
train: images/train
val: images/val

nc: {nc}
names: {names}
"""


def write_data_yaml_template(output: Path) -> None:
    """Write a template ``data.yaml`` if it does not already exist."""
    if output.exists():
        logger.info("data.yaml already exists at %s — skipping.", output)
        return
    content = DATA_YAML_TEMPLATE.format(
        nc=NUM_CLASSES,
        names=CLASS_NAMES,
    )
    output.write_text(content, encoding="utf-8")
    logger.info("Wrote data.yaml template → %s", output)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Run YOLO training / fine-tuning."""
    from ultralytics import YOLO

    data_path = Path(args.data)
    if not data_path.exists():
        logger.warning(
            "data.yaml not found at %s — writing template.", data_path
        )
        write_data_yaml_template(data_path)
        logger.info("Edit %s with your dataset paths, then re-run.", data_path)
        return

    device = args.device
    if device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(args.model)
    logger.info("Starting training: %d epochs, batch %d, device %s",
                args.epochs, args.batch, device)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=YOLO_IMG_SIZE,
        device=device,
        seed=RANDOM_SEED,
        # --- strong augmentation for manuscripts ---
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=5.0,          # slight rotation
        translate=0.1,
        scale=0.3,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,           # manuscripts shouldn't be flipped vertically
        fliplr=0.0,           # or horizontally
        mosaic=0.5,
        mixup=0.1,
        erasing=0.1,
        blur=0.1,
        # output
        project="runs/train",
        name="manuscript_layout",
        exist_ok=True,
        verbose=True,
    )
    logger.info("Training complete.  Weights saved to runs/train/manuscript_layout/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO for manuscript layout detection.",
    )
    parser.add_argument(
        "--data", type=str, default="data.yaml",
        help="Path to YOLO data.yaml.",
    )
    parser.add_argument(
        "--model", type=str, default=YOLO_DEFAULT_MODEL,
        help=f"Base YOLO model to fine-tune (default: {YOLO_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=TRAIN_EPOCHS,
        help=f"Number of training epochs (default: {TRAIN_EPOCHS}).",
    )
    parser.add_argument(
        "--batch", type=int, default=TRAIN_BATCH,
        help=f"Batch size (default: {TRAIN_BATCH}).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: cpu | 0 | auto (default: auto).",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
