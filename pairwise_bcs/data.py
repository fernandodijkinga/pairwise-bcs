"""Input files, image loading and augmentation.

Expected inputs (CSV, UTF-8, with header):
  comparisons.csv : winner_id,loser_id        one row per pairwise decision
  ordinal.csv     : image_id,bcs              ordinal BCS (1-5, 0.25 steps), optional
  images/         : one file per image, named <image_id>.jpg / .png / .jpeg

Images are expected to be already cropped to the animal (in the study, the
posterior region was cropped with a YOLO detector; see Section 2.1). They are
resized to 224 x 224, kept in memory as uint8, and normalised with the
ImageNet mean and standard deviation.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LEVELS = np.round(np.arange(1.0, 5.0001, 0.25), 2)     # 17 ordinal BCS levels


def load_comparisons(path) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["winner_id"], r["loser_id"]) for r in csv.DictReader(f)]


def load_ordinal(path) -> dict[str, float]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["image_id"]: float(r["bcs"]) for r in csv.DictReader(f)}


def level_index(bcs: float) -> int:
    """Index (0..16) of the nearest ordinal level."""
    return int(np.argmin(np.abs(LEVELS - bcs)))


class ImageStore:
    """All images resized to 224 x 224 and kept in memory (uint8)."""

    def __init__(self, image_dir, image_ids):
        image_dir = Path(image_dir)
        self.index, arrays = {}, []
        for i in image_ids:
            path = next((image_dir / f"{i}{ext}" for ext in (".jpg", ".png", ".jpeg")
                         if (image_dir / f"{i}{ext}").exists()), None)
            if path is None:
                raise FileNotFoundError(f"image for id {i} not found in {image_dir}")
            img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            self.index[i] = len(arrays)
            arrays.append(np.asarray(img, dtype=np.uint8))
        self.images = np.stack(arrays)

    def batch(self, ids, device, augment: bool = False, rng: np.random.Generator | None = None):
        x = self.images[[self.index[i] for i in ids]].astype(np.float32) / 255.0
        if augment:
            rng = rng or np.random.default_rng()
            for k in range(len(x)):
                if rng.random() < 0.5:                       # horizontal flip
                    x[k] = x[k, :, ::-1, :]
                gain = 1.0 + rng.uniform(-0.2, 0.2)          # brightness jitter
                bias = rng.uniform(-0.02, 0.02)
                x[k] = np.clip(x[k] * gain + bias, 0.0, 1.0)
        x = (x - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(device)
