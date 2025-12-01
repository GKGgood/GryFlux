#!/usr/bin/env python3
"""Compare ONNX detections against TensorRT engine outputs (FP16/INT8)."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

from detection_utils import (
    COCO_CLASSES,
    bbox_iou_xyxy,
    ensure_json_structure,
    flatten_detection_json,
    normalize_image_name,
    read_json,
)

BBox = List[float]
Detection = Dict[str, float | int | str | BBox]
DetectionMap = Dict[str, List[Detection]]

READ_FILE_RE = re.compile(r"Reading file\s+(.+)$")
LETTERBOX_RE = re.compile(r"Letterbox processing:\s*id=(\d+)")
DET_RE = re.compile(
    r"x1:\s*(-?[0-9.]+),\s*y1:\s*(-?[0-9.]+),\s*x2:\s*(-?[0-9.]+),\s*y2:\s*(-?[0-9.]+),\s*id:\s*(\d+),\s*obj_conf:\s*([0-9.]+)"
)
FRAME_DONE_RE = re.compile(r"Frame\s+(\d+)\s+processed")


def load_detections(source: Path) -> DetectionMap:
    if source.is_dir():
        json_path = source / "detections.json"
        if not json_path.exists():
            raise SystemExit(f"Directory {source} does not contain detections.json")
        payload = read_json(json_path)
        return flatten_detection_json(ensure_json_structure(payload))

    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = read_json(source)
        return flatten_detection_json(ensure_json_structure(payload))
    if suffix == ".log":
        return parse_streaming_log(source)

    raise SystemExit(f"Unsupported detection source: {source}")


def parse_streaming_log(log_path: Path) -> DetectionMap:
    """Parse GryFlux streaming logs to rebuild detection results."""
    id_to_image: Dict[int, str] = {}
    detections: Dict[str, List[Detection]] = defaultdict(list)
    current_id: int | None = None
    next_image_id = 0

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            read_match = READ_FILE_RE.search(line)
            if read_match:
                image_path = normalize_image_name(read_match.group(1))
                id_to_image[next_image_id] = image_path
                next_image_id += 1
                continue

            letter_match = LETTERBOX_RE.search(line)
            if letter_match:
                current_id = int(letter_match.group(1))
                continue

            frame_match = FRAME_DONE_RE.search(line)
            if frame_match:
                current_id = None
                continue

            det_match = DET_RE.search(line)
            if det_match and current_id is not None:
                x1, y1, x2, y2 = (float(det_match.group(i)) for i in range(1, 5))
                class_id = int(det_match.group(5))
                score = float(det_match.group(6))
                image_id = id_to_image.get(current_id, f"frame_{current_id}")
                detections[image_id].append(
                    {
                        "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                        "score": score,
                        "class_id": class_id,
                        "class_name": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id),
                    }
                )
    for image_name in id_to_image.values():
        detections.setdefault(image_name, [])
    return detections


def filter_detections(items: Iterable[Detection], score_threshold: float) -> List[Detection]:
    filtered: List[Detection] = []
    for det in items:
        bbox = det.get("bbox")
        score = float(det.get("score", 0.0))
        class_id = int(det.get("class_id", -1))
        if score < score_threshold or bbox is None or len(bbox) != 4 or class_id < 0:
            continue
        class_name = det.get("class_name")
        if not class_name:
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)
        filtered.append(
            {
                "bbox": [float(coord) for coord in bbox],
                "score": score,
                "class_id": class_id,
                "class_name": class_name,
            }
        )
    return filtered


def match_detections(
    reference: Sequence[Detection],
    candidate: Sequence[Detection],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    matches: List[Tuple[int, int, float]] = []
    used_candidate = set()
    used_reference = set()

    sorted_refs = sorted(enumerate(reference), key=lambda item: item[1]["score"], reverse=True)
    for ref_idx, ref_det in sorted_refs:
        best_idx = None
        best_iou = 0.0
        for cand_idx, cand_det in enumerate(candidate):
            if cand_idx in used_candidate:
                continue
            if cand_det["class_id"] != ref_det["class_id"]:
                continue
            iou = bbox_iou_xyxy(ref_det["bbox"], cand_det["bbox"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_idx = cand_idx
        if best_idx is not None:
            used_candidate.add(best_idx)
            used_reference.add(ref_idx)
            matches.append((ref_idx, best_idx, best_iou))

    unmatched_ref = [idx for idx in range(len(reference)) if idx not in used_reference]
    unmatched_cand = [idx for idx in range(len(candidate)) if idx not in used_candidate]
    return matches, unmatched_ref, unmatched_cand


def evaluate_candidate(
    name: str,
    reference_map: DetectionMap,
    candidate_map: DetectionMap,
    iou_threshold: float,
    score_threshold: float,
) -> Dict:
    image_ids = sorted(set(reference_map.keys()) | set(candidate_map.keys()))
    totals = {
        "name": name,
        "images": len(image_ids),
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "ref_detections": 0,
        "cand_detections": 0,
        "iou_sum": 0.0,
        "score_gap_sum": 0.0,
        "bbox_l1_sum": 0.0,
        "match_count": 0,
    }
    per_image: List[Dict] = []
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "match_count": 0})

    for image_id in image_ids:
        ref = filter_detections(reference_map.get(image_id, []), score_threshold)
        cand = filter_detections(candidate_map.get(image_id, []), score_threshold)
        matches, unmatched_ref_idx, unmatched_cand_idx = match_detections(ref, cand, iou_threshold)
        tp = len(matches)
        fp = len(unmatched_cand_idx)
        fn = len(unmatched_ref_idx)

        totals["ref_detections"] += len(ref)
        totals["cand_detections"] += len(cand)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

        image_iou = []
        for ref_idx, cand_idx, iou in matches:
            ref_det = ref[ref_idx]
            cand_det = cand[cand_idx]
            totals["iou_sum"] += iou
            totals["score_gap_sum"] += abs(ref_det["score"] - cand_det["score"])
            bbox_gap = sum(abs(a - b) for a, b in zip(ref_det["bbox"], cand_det["bbox"]))
            totals["bbox_l1_sum"] += bbox_gap
            totals["match_count"] += 1
            per_class_stats = per_class[ref_det["class_id"]]
            per_class_stats["tp"] += 1
            per_class_stats["iou_sum"] += iou
            per_class_stats["match_count"] += 1
            image_iou.append(iou)

        for idx in unmatched_cand_idx:
            per_class[cand[idx]["class_id"]]["fp"] += 1
        for idx in unmatched_ref_idx:
            per_class[ref[idx]["class_id"]]["fn"] += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_image.append(
            {
                "image_id": image_id,
                "reference": len(ref),
                "candidate": len(cand),
                "matched": tp,
                "precision": precision,
                "recall": recall,
                "mean_iou": mean(image_iou) if image_iou else 0.0,
            }
        )

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) > 0 else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    totals.update(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": totals["iou_sum"] / totals["match_count"] if totals["match_count"] else 0.0,
            "mean_score_gap": totals["score_gap_sum"] / totals["match_count"] if totals["match_count"] else 0.0,
            "mean_bbox_l1": totals["bbox_l1_sum"] / totals["match_count"] if totals["match_count"] else 0.0,
            "per_image": per_image,
            "per_class": build_per_class_report(per_class),
        }
    )
    return totals


def build_per_class_report(per_class_stats: Dict[int, Dict]) -> List[Dict]:
    report = []
    for class_id, stats in per_class_stats.items():
        tp = stats["tp"]
        fp = stats.get("fp", 0)
        fn = stats.get("fn", 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        report.append(
            {
                "class_id": class_id,
                "class_name": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "mean_iou": stats["iou_sum"] / stats["match_count"] if stats["match_count"] else 0.0,
            }
        )
    return sorted(report, key=lambda item: item["recall"], reverse=True)


def print_summary(stats: Dict) -> None:
    print(f"\n[{stats['name']}] vs reference")
    print(f"  Images evaluated : {stats['images']}")
    print(f"  Reference dets   : {stats['ref_detections']}")
    print(f"  Candidate dets   : {stats['cand_detections']}")
    print(f"  Precision        : {stats['precision']:.4f}")
    print(f"  Recall           : {stats['recall']:.4f}")
    print(f"  F1-score         : {stats['f1']:.4f}")
    print(f"  Mean IoU         : {stats['mean_iou']:.4f}")
    print(f"  |Score gap|      : {stats['mean_score_gap']:.4f}")
    print(f"  |BBox L1|        : {stats['mean_bbox_l1']:.2f}")

    top_worst = sorted(stats["per_class"], key=lambda item: item["recall"])[:5]
    if top_worst:
        print("  Lowest recall classes (top-5):")
        for entry in top_worst:
            print(
                f"    - {entry['class_name']} (id {entry['class_id']}):"
                f" precision={entry['precision']:.3f}, recall={entry['recall']:.3f}, IoU={entry['mean_iou']:.3f}"
            )


def save_json_report(path: Path, fp16_stats: Dict, int8_stats: Dict) -> None:
    payload = {
        "fp16": fp16_stats,
        "int8": int8_stats,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_per_image_csv(path: Path, fp16_stats: Dict, int8_stats: Dict) -> None:
    import csv

    header = [
        "image_id",
        "model",
        "reference",
        "candidate",
        "matched",
        "precision",
        "recall",
        "mean_iou",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for stats in (fp16_stats, int8_stats):
            for row in stats["per_image"]:
                writer.writerow(
                    [
                        row["image_id"],
                        stats["name"],
                        row["reference"],
                        row["candidate"],
                        row["matched"],
                        f"{row['precision']:.4f}",
                        f"{row['recall']:.4f}",
                        f"{row['mean_iou']:.4f}",
                    ]
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ONNX detections with FP16/INT8 engine outputs.")
    parser.add_argument("--reference", type=Path, required=True, help="Reference detection JSON (ONNX script output)")
    parser.add_argument("--fp16", type=Path, required=True, help="FP16 engine detection JSON or log file")
    parser.add_argument("--int8", type=Path, required=True, help="INT8 engine detection JSON or log file")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching detections")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Confidence threshold before matching")
    parser.add_argument("--report-json", type=Path, help="Optional path to store a JSON summary")
    parser.add_argument("--per-image-csv", type=Path, help="Optional path to store a per-image CSV report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_map = load_detections(args.reference)
    fp16_map = load_detections(args.fp16)
    int8_map = load_detections(args.int8)

    fp16_stats = evaluate_candidate(
        name="TensorRT FP16",
        reference_map=reference_map,
        candidate_map=fp16_map,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )
    int8_stats = evaluate_candidate(
        name="TensorRT INT8",
        reference_map=reference_map,
        candidate_map=int8_map,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )

    print_summary(fp16_stats)
    print_summary(int8_stats)

    if args.report_json:
        save_json_report(args.report_json, fp16_stats, int8_stats)
        print(f"Saved aggregate JSON report to {args.report_json}")
    if args.per_image_csv:
        save_per_image_csv(args.per_image_csv, fp16_stats, int8_stats)
        print(f"Saved per-image CSV report to {args.per_image_csv}")


if __name__ == "__main__":
    main()
