#!/usr/bin/env python3
"""Compute fusion quality metrics (mutual information, entropy, mean difference, spatial frequency).

The script expects three directories containing matched visible, infrared, and fused images.
Matching is controlled by filename-derived keys, similar to tools/psnr_eval.py.
"""

import argparse
import math
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute fusion metrics for matched visible/infrared/fused image triplets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("visible", type=Path, help="Directory with visible-light images.")
    parser.add_argument("infrared", type=Path, help="Directory with infrared images.")
    parser.add_argument("fused", type=Path, help="Directory with fused output images.")

    parser.add_argument(
        "--extensions",
        nargs="*",
        default=[".png", ".jpg", ".jpeg", ".bmp"],
        help="File extensions (case-insensitive) to consider.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories recursively for matching files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print aggregate metrics.",
    )
    parser.add_argument(
        "--match-delimiter",
        type=str,
        default="_",
        help="Delimiter used when trimming filename suffix segments for matching.",
    )
    parser.add_argument(
        "--match-drop-suffix-visible",
        type=int,
        default=0,
        help="Trailing segments to drop from visible filenames when matching.",
    )
    parser.add_argument(
        "--match-drop-suffix-infrared",
        type=int,
        default=0,
        help="Trailing segments to drop from infrared filenames when matching.",
    )
    parser.add_argument(
        "--match-drop-suffix-fused",
        type=int,
        default=0,
        help="Trailing segments to drop from fused filenames when matching.",
    )
    parser.add_argument(
        "--match-ignore-case",
        action="store_true",
        help="Ignore case when comparing derived filename keys.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to write per-image metrics as CSV.",
    )
    return parser.parse_args()


def normalize_exts(exts: List[str]) -> List[str]:
    normalized = []
    for ext in exts:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith('.'):
            ext = f".{ext}"
        normalized.append(ext)
    return normalized or ['.png']


def trim_suffix_segments(stem: str, delimiter: str, count: int) -> str:
    if count <= 0 or not delimiter:
        return stem
    segments = stem.split(delimiter)
    if len(segments) <= count:
        return stem
    trimmed = delimiter.join(segments[:-count])
    return trimmed if trimmed else stem


def make_key_builder(drop_suffix: int, delimiter: str, ignore_case: bool) -> Callable[[Path], str]:
    def builder(path: Path) -> str:
        stem = path.stem
        trimmed = trim_suffix_segments(stem, delimiter, drop_suffix)
        key = f"{trimmed}{path.suffix.lower()}"
        if ignore_case:
            key = key.lower()
        return key

    return builder


def collect_images(
    root: Path,
    exts: List[str],
    recursive: bool,
    key_builder: Callable[[Path], str],
) -> Tuple[Dict[str, Path], List[Path]]:
    files: Dict[str, Path] = {}
    duplicates: List[Path] = []
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        key = key_builder(path)
        if key in files:
            duplicates.append(path)
            continue
        files[key] = path
    return files, duplicates


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        if image.shape[2] == 1:
            return image[:, :, 0]
        if image.shape[2] >= 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def compute_histogram(image: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel()
    total = np.sum(hist)
    if total <= 0:
        raise ValueError("Histogram sum is zero; check input image.")
    return hist / total


def compute_entropy(image: np.ndarray) -> float:
    probs = compute_histogram(image)
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = np.log2(probs)
    logs[~np.isfinite(logs)] = 0.0
    entropy = -float(np.sum(probs * logs))
    return entropy


def compute_mutual_information(image_a: np.ndarray, image_b: np.ndarray) -> float:
    joint_hist, _, _ = np.histogram2d(
        image_a.ravel(),
        image_b.ravel(),
        bins=256,
        range=[[0, 255], [0, 255]],
    )
    joint_prob = joint_hist / np.sum(joint_hist)
    prob_a = np.sum(joint_prob, axis=1, keepdims=True)
    prob_b = np.sum(joint_prob, axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_term = np.log2(joint_prob) - np.log2(prob_a) - np.log2(prob_b)
    log_term[~np.isfinite(log_term)] = 0.0
    mi = float(np.sum(joint_prob * log_term))
    return mi


def compute_std_difference(vis: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    vis_f = vis.astype(np.float32)
    ir_f = ir.astype(np.float32)
    fused_f = fused.astype(np.float32)
    average_source = 0.5 * (vis_f + ir_f)
    diff = fused_f - average_source
    return float(np.std(diff))


def compute_spatial_frequency(image: np.ndarray) -> float:
    img = image.astype(np.float32)
    if img.size == 0:
        return 0.0
    row_diff = np.diff(img, axis=0)
    col_diff = np.diff(img, axis=1)
    rf = math.sqrt(float(np.mean(row_diff * row_diff))) if row_diff.size else 0.0
    cf = math.sqrt(float(np.mean(col_diff * col_diff))) if col_diff.size else 0.0
    return math.sqrt(rf * rf + cf * cf)


def write_csv(path: Path, rows: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    for directory in (args.visible, args.infrared, args.fused):
        if not directory.is_dir():
            print(f"Directory not found: {directory}")
            return 1

    exts = normalize_exts(args.extensions)
    vis_key_builder = make_key_builder(
        args.match_drop_suffix_visible,
        args.match_delimiter,
        args.match_ignore_case,
    )
    ir_key_builder = make_key_builder(
        args.match_drop_suffix_infrared,
        args.match_delimiter,
        args.match_ignore_case,
    )
    fused_key_builder = make_key_builder(
        args.match_drop_suffix_fused,
        args.match_delimiter,
        args.match_ignore_case,
    )

    vis_images, vis_dupes = collect_images(args.visible, exts, args.recursive, vis_key_builder)
    ir_images, ir_dupes = collect_images(args.infrared, exts, args.recursive, ir_key_builder)
    fused_images, fused_dupes = collect_images(args.fused, exts, args.recursive, fused_key_builder)

    if vis_dupes:
        print(f"Warning: skipped {len(vis_dupes)} visible files due to duplicate match keys.")
    if ir_dupes:
        print(f"Warning: skipped {len(ir_dupes)} infrared files due to duplicate match keys.")
    if fused_dupes:
        print(f"Warning: skipped {len(fused_dupes)} fused files due to duplicate match keys.")

    common_keys = sorted(set(vis_images) & set(ir_images) & set(fused_images))
    if not common_keys:
        print("No matching triplets found across the provided directories.")
        return 1

    header = "Key,MI_visible,MI_infrared,MI_total,Entropy,StdDifference,SpatialFrequency"
    csv_rows: List[str] = [header]
    if not args.quiet:
        print(header)

    aggregates = {
        "mi_visible": [],
        "mi_infrared": [],
        "entropy": [],
        "std_difference": [],
        "spatial_frequency": [],
    }
    processed_keys: List[str] = []

    for key in common_keys:
        vis_path = vis_images[key]
        ir_path = ir_images[key]
        fused_path = fused_images[key]

        vis_img = load_image(vis_path)
        ir_img = load_image(ir_path)
        fused_img = load_image(fused_path)

        if vis_img.shape[:2] != ir_img.shape[:2] or vis_img.shape[:2] != fused_img.shape[:2]:
            print(
                f"Skip {vis_path.name}, {ir_path.name}, {fused_path.name}:"
                f" spatial mismatch {vis_img.shape} / {ir_img.shape} / {fused_img.shape}"
            )
            continue

        vis_gray = ensure_grayscale(vis_img)
        ir_gray = ensure_grayscale(ir_img)
        fused_gray = ensure_grayscale(fused_img)

        try:
            entropy = compute_entropy(fused_gray)
            mi_vis = compute_mutual_information(fused_gray, vis_gray)
            mi_ir = compute_mutual_information(fused_gray, ir_gray)
            std_diff = compute_std_difference(vis_gray, ir_gray, fused_gray)
            spatial_freq = compute_spatial_frequency(fused_gray)
        except ValueError as exc:
            print(f"Skip {fused_path.name}: {exc}")
            continue

        aggregates["mi_visible"].append(mi_vis)
        aggregates["mi_infrared"].append(mi_ir)
        aggregates["entropy"].append(entropy)
        aggregates["std_difference"].append(std_diff)
        aggregates["spatial_frequency"].append(spatial_freq)
        processed_keys.append(key)

        row = (
            f"{key},{mi_vis:.6f},{mi_ir:.6f},{mi_vis + mi_ir:.6f},"
            f"{entropy:.6f},{std_diff:.6f},{spatial_freq:.6f}"
        )
        csv_rows.append(row)
        if not args.quiet:
            print(row)

    count = len(processed_keys)
    if count == 0:
        print("No valid image triplets after filtering mismatches.")
        return 1

    avg_mi_vis = float(sum(aggregates["mi_visible"]) / count)
    avg_mi_ir = float(sum(aggregates["mi_infrared"]) / count)
    avg_entropy = float(sum(aggregates["entropy"]) / count)
    avg_std_diff = float(sum(aggregates["std_difference"]) / count)
    avg_spatial_freq = float(sum(aggregates["spatial_frequency"]) / count)

    summary = (
        "Averages: "
        f"MI_visible={avg_mi_vis:.6f}, "
        f"MI_infrared={avg_mi_ir:.6f}, "
        f"MI_total={avg_mi_vis + avg_mi_ir:.6f}, "
        f"Entropy={avg_entropy:.6f}, "
        f"StdDifference={avg_std_diff:.6f}, "
        f"SpatialFrequency={avg_spatial_freq:.6f}"
    )
    print(summary)

    if args.output_csv:
        csv_rows.append(
            f"Averages,{avg_mi_vis:.6f},{avg_mi_ir:.6f},{avg_mi_vis + avg_mi_ir:.6f},"
            f"{avg_entropy:.6f},{avg_std_diff:.6f},{avg_spatial_freq:.6f}"
        )
        write_csv(args.output_csv, csv_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
