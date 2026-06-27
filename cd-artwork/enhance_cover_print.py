#!/usr/bin/env python3
"""Upscale and sharpen album cover / booklet art for printable 600 DPI output."""

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


def fit_square_canvas(bgr: np.ndarray, size: int, bg: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((size, size, 3), bg, dtype=np.uint8)
    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def convert_reflective_to_white(bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan = lab[..., 0]

    # Preserve saturated colors (red disc, art colors).
    saturated = s > 55
    dark = l_chan < 62

    reflective = (
        ((s < 48) & (v > 92))
        | ((l_chan > 175) & (s < 75))
        | ((v > 178) & (s < 65))
    )
    reflective &= ~saturated
    reflective &= ~dark

    out = bgr.copy()
    if strength >= 1.0:
        out[reflective] = (255, 255, 255)
    else:
        blend = out[reflective].astype(np.float32)
        blend = blend * (1 - strength) + 255.0 * strength
        out[reflective] = np.clip(blend, 0, 255).astype(np.uint8)
    return out


def reduce_scan_bleed(bgr: np.ndarray) -> np.ndarray:
    """Reduce faint reverse-page ghosting on white booklet scans."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Lift very light background while keeping strong text/ink.
    mask_ink = l < 200
    l_bg = cv2.GaussianBlur(l, (0, 0), 2.2)
    l2 = l.copy()
    l2[~mask_ink] = np.clip(l_bg[~mask_ink] * 0.55 + 255 * 0.45, 0, 255).astype(np.uint8)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def enhance_clarity(bgr: np.ndarray, text_boost: bool = True) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clip = 2.2 if text_boost else 1.6
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    smooth = cv2.bilateralFilter(enhanced, 5, 20, 20)
    blur = cv2.GaussianBlur(smooth, (0, 0), 0.9)
    sharp = cv2.addWeighted(smooth, 1.45, blur, -0.45, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def preserve_ink_and_colors(bgr: np.ndarray, original: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    l = lab[..., 0]

    dark_ink = l < 70
    strong_color = (s > 65) & (v > 35)

    out = bgr.copy()
    out[dark_ink] = original[dark_ink]
    out[strong_color] = original[strong_color]
    return out


def upscale_to_square(bgr: np.ndarray, size: int) -> np.ndarray:
    sq = fit_square_canvas(bgr, size)
    # Two-pass upscale for max sharpness when source is much smaller.
    h, w = bgr.shape[:2]
    if max(h, w) < size * 0.5:
        mid = size // 2
        sq_mid = fit_square_canvas(bgr, mid)
        return cv2.resize(sq_mid, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return sq


def final_sharpen(bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), 0.75)
    sharp = cv2.addWeighted(bgr, 1.3, blur, -0.3, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def process_cover(
    input_path: Path,
    output_path: Path,
    size: int = 6000,
    dpi: int = 600,
    *,
    white_reflective: bool = True,
    text_boost: bool = True,
    bleed_reduce: bool = False,
) -> None:
    src = load_bgr(input_path)
    sq = upscale_to_square(src, size)
    if white_reflective:
        sq = convert_reflective_to_white(sq)
    if bleed_reduce:
        sq = reduce_scan_bleed(sq)
    clearer = enhance_clarity(sq, text_boost=text_boost)
    locked = preserve_ink_and_colors(clearer, sq)
    final = final_sharpen(locked)
    save_rgb(output_path, final, dpi)
    tiff_path = output_path.with_suffix(".tiff")
    rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(tiff_path, compression="tiff_lzw", dpi=(dpi, dpi))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--size", type=int, default=6000)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--no-white-reflective", action="store_true")
    p.add_argument("--no-text-boost", action="store_true")
    p.add_argument("--bleed-reduce", action="store_true")
    args = p.parse_args()
    process_cover(
        Path(args.input),
        Path(args.output),
        args.size,
        args.dpi,
        white_reflective=not args.no_white_reflective,
        text_boost=not args.no_text_boost,
        bleed_reduce=args.bleed_reduce,
    )


if __name__ == "__main__":
    main()
