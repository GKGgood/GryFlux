"""Shared helpers for detection scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bbox_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def normalize_image_name(path_str: str) -> str:
    """Convert absolute paths logged by C++ into dataset-relative file names."""
    path = Path(path_str.strip().strip())
    return path.name


def ensure_json_structure(payload: Dict) -> Dict:
    """Small helper that enforces consistent keys for detection JSON blobs."""
    required = {"images"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Detection JSON is missing required keys: {missing}")
    return payload


def flatten_detection_json(payload: Dict) -> Dict[str, List[Dict]]:
    """Return a mapping from image_id to detections."""
    result: Dict[str, List[Dict]] = {}
    for image_entry in payload.get("images", []):
        image_id = image_entry.get("image_id") or Path(image_entry.get("image_path", "")).name
        result[image_id] = image_entry.get("detections", [])
    return result


def is_json_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".json"


def read_json(path: Path) -> Dict:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
