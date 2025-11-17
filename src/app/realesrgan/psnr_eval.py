#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PSNR for matching image pairs in two directories."
    )
    parser.add_argument("reference", type=Path, help="Directory with ground-truth images.")
    parser.add_argument("target", type=Path, help="Directory with super-resolved images.")
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=[".png", ".jpg", ".jpeg", ".bmp"],
        help="File extensions (case-insensitive) to consider. Default: %(default)s",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories recursively for matching files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the aggregate PSNR value.",
    )
    parser.add_argument(
        "--match-delimiter",
        type=str,
        default="_",
        help="Delimiter used to split filename stems when dropping suffix segments. Use empty string to disable.",
    )
    parser.add_argument(
        "--match-drop-suffix-ref",
        type=int,
        default=0,
        help="Number of trailing segments (after splitting by delimiter) to drop from reference filenames when matching.",
    )
    parser.add_argument(
        "--match-drop-suffix-target",
        type=int,
        default=None,
        help="Number of trailing segments to drop from target filenames when matching. Defaults to the reference setting if omitted.",
    )
    parser.add_argument(
        "--match-ignore-case",
        action="store_true",
        help="Ignore case when comparing derived filename keys.",
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


def collect_images(
    root: Path,
    exts: List[str],
    recursive: bool,
    key_builder: Callable[[Path], str],
) -> Tuple[Dict[str, Path], List[Path]]:
    files: Dict[str, Path] = {}
    duplicates: List[Path] = []
    if recursive:
        iterator = root.rglob("*")
    else:
        iterator = root.glob("*")
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


def compute_psnr(reference: np.ndarray, target: np.ndarray) -> float:
    diff = reference.astype(np.float32) - target.astype(np.float32)
    mse = np.mean(diff * diff)
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def main() -> int:
    args = parse_args()
    ref_dir = args.reference
    tgt_dir = args.target

    if not ref_dir.is_dir():
        print(f"Reference directory not found: {ref_dir}")
        return 1
    if not tgt_dir.is_dir():
        print(f"Target directory not found: {tgt_dir}")
        return 1

    exts = normalize_exts(args.extensions)
    drop_target = (
        args.match_drop_suffix_target
        if args.match_drop_suffix_target is not None
        else args.match_drop_suffix_ref
    )
    ref_key_builder = make_key_builder(
        args.match_drop_suffix_ref,
        args.match_delimiter,
        args.match_ignore_case,
    )
    tgt_key_builder = make_key_builder(
        drop_target,
        args.match_delimiter,
        args.match_ignore_case,
    )

    ref_images, ref_duplicates = collect_images(ref_dir, exts, args.recursive, ref_key_builder)
    tgt_images, tgt_duplicates = collect_images(tgt_dir, exts, args.recursive, tgt_key_builder)

    if ref_duplicates:
        print(f"Warning: skipped {len(ref_duplicates)} reference files due to duplicate match keys.")
    if tgt_duplicates:
        print(f"Warning: skipped {len(tgt_duplicates)} target files due to duplicate match keys.")

    common_keys = sorted(set(ref_images.keys()) & set(tgt_images.keys()))
    if not common_keys:
        print("No matching image pairs found.")
        return 1

    scores: List[Tuple[str, float]] = []
    for key in common_keys:
        ref_path = ref_images[key]
        tgt_path = tgt_images[key]
        ref_img = load_image(ref_path)
        tgt_img = load_image(tgt_path)
        if ref_img.shape != tgt_img.shape:
            print(
                f"Skip {ref_path.name} <-> {tgt_path.name}:"
                f" shape mismatch {ref_img.shape} vs {tgt_img.shape}"
            )
            continue
        score = compute_psnr(ref_img, tgt_img)
        scores.append((f"{ref_path.name} <-> {tgt_path.name}", score))

    if not scores:
        print("No valid image pairs after filtering mismatches.")
        return 1

    average = sum(score for _, score in scores) / len(scores)

    if not args.quiet:
        print("Reference/Target,PSNR(dB)")
        for name, score in scores:
            if math.isinf(score):
                print(f"{name},inf")
            else:
                print(f"{name},{score:.4f}")
    print(f"Average PSNR: {average:.4f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
