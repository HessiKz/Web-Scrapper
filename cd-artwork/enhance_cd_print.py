#!/usr/bin/env python3
"""Enhance CD disc artwork for high-DPI printable output."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_bgr(path: Path) -> np.ndarray:
    rgb = np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_rgb(path: Path, bgr: np.ndarray, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, dpi=(dpi, dpi), compress_level=1)


def to_square_white_canvas(bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def convert_reflective_to_white(bgr: np.ndarray) -> np.ndarray:
    """Map silver / reflective / near-white metallic tones to pure white."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan = lab[..., 0]

    # Red disc body mask: preserve saturated warm reds.
    red_mask = (
        (h < 20) | (h > 160)
    ) & (s > 45) & (v > 35)

    # Black ink / logo mask.
    black_mask = l_chan < 58

    # Reflective silver / highlight mask.
    reflective = (
        ((s < 55) & (v > 95))
        | ((l_chan > 168) & (s < 80))
        | ((v > 175) & (s < 70))
    )
    reflective &= ~red_mask
    reflective &= ~black_mask

    out = bgr.copy()
    out[reflective] = (255, 255, 255)
    return out


def enhance_text_clarity(bgr: np.ndarray) -> np.ndarray:
    """Boost local contrast and edge definition without shifting hues."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    merged = cv2.merge([l2, a, b])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Gentle unsharp mask for typography and line art.
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
    sharp = cv2.addWeighted(enhanced, 1.55, blur, -0.55, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def preserve_core_colors(bgr: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Re-lock saturated red and deep black from source to avoid color drift."""
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    red = ((h < 18) | (h > 165)) & (s > 70) & (v > 40)
    black = v < 55

    out = bgr.copy()
    out[red] = original[red]
    out[black] = original[black]
    return out


def upscale_lanczos(bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LANCZOS4)


def final_print_sharpen(bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), 0.85)
    sharp = cv2.addWeighted(bgr, 1.35, blur, -0.35, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def process(input_path: Path, output_path: Path, size: int, dpi: int) -> None:
    src = load_bgr(input_path)
    sq = to_square_white_canvas(src, size)
    white_mapped = convert_reflective_to_white(sq)
    clearer = enhance_text_clarity(white_mapped)
    locked = preserve_core_colors(clearer, sq)
    final = final_print_sharpen(locked)
    save_rgb(output_path, final, dpi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--size", type=int, default=6000)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    process(Path(args.input), Path(args.output), args.size, args.dpi)


if __name__ == "__main__":
    main()
