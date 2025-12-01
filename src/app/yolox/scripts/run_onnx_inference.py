#!/usr/bin/env python3
"""Run YOLOX ONNX inference over a dataset and export visual + JSON results."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "onnxruntime is required. Install it via `pip install onnxruntime` before running this script."
    ) from exc

from detection_utils import COCO_CLASSES, bbox_iou_xyxy, clamp

SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def letterbox_image(image: np.ndarray, target_size: int, pad_color: int = 114) -> Tuple[np.ndarray, float, int, int]:
    """Resize with unchanged aspect ratio using padding (YOLOX letterbox)."""
    height, width = image.shape[:2]
    scale = min(target_size / width, target_size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), pad_color, dtype=resized.dtype)
    x_pad = (target_size - new_w) // 2
    y_pad = (target_size - new_h) // 2
    canvas[y_pad : y_pad + new_h, x_pad : x_pad + new_w] = resized
    return canvas, scale, x_pad, y_pad


def preprocess(image: np.ndarray, input_size: int) -> Tuple[np.ndarray, Dict[str, float]]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    letterboxed, scale, x_pad, y_pad = letterbox_image(rgb, input_size)
    # C++ reference does not normalize by 255.0, it uses 0-255 float values.
    normalized = letterboxed.astype(np.float32)
    chw = np.transpose(normalized, (2, 0, 1))
    batched = np.expand_dims(chw, axis=0)
    meta = {
        "scale": scale,
        "x_pad": x_pad,
        "y_pad": y_pad,
        "model_w": input_size,
        "model_h": input_size,
    }
    return batched, meta


def decode_outputs(
    raw_outputs: Sequence[np.ndarray],
    meta: Dict[str, float],
    score_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> List[Dict[str, float]]:
    boxes: List[List[float]] = []
    scores: List[float] = []
    class_ids: List[int] = []
    input_w = meta["model_w"]

    for output in raw_outputs:
        _, channels, grid_h, grid_w = output.shape
        assert channels == 85, "Expected YOLOX output with 85 channels (cx, cy, w, h, obj, 80 classes)."
        stride = input_w // grid_w
        flattened = output.reshape(channels, grid_h * grid_w)
        grid_len = grid_h * grid_w

        for idx in range(grid_len):
            # C++ reference treats these as probabilities directly, implying the model output is already sigmoid-ed.
            obj_conf = float(flattened[4, idx])
            if obj_conf < score_threshold:
                continue

            cls_scores = flattened[5:, idx]
            class_id = int(np.argmax(cls_scores))
            class_conf = float(cls_scores[class_id])
            
            score = obj_conf * class_conf
            if score < score_threshold:
                continue

            cx = float(flattened[0, idx])
            cy = float(flattened[1, idx])
            width = float(flattened[2, idx])
            height = float(flattened[3, idx])

            grid_y, grid_x = divmod(idx, grid_w)
            cx = (cx + grid_x) * stride
            cy = (cy + grid_y) * stride
            width = math.exp(width) * stride
            height = math.exp(height) * stride

            x1 = cx - width / 2.0
            y1 = cy - height / 2.0
            x2 = x1 + width
            y2 = y1 + height

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(class_id)

    if not boxes:
        return []

    order = np.argsort(scores)[::-1]
    keep_flags = [True] * len(boxes)
    filtered_indices: List[int] = []

    for order_pos, idx in enumerate(order):
        if not keep_flags[idx]:
            continue
        filtered_indices.append(idx)
        if len(filtered_indices) >= max_detections:
            break
        for next_idx in order[order_pos + 1 :]:
            if not keep_flags[next_idx]:
                continue
            if class_ids[idx] != class_ids[next_idx]:
                continue
            if bbox_iou_xyxy(boxes[idx], boxes[next_idx]) > nms_threshold:
                keep_flags[next_idx] = False

    detections: List[Dict[str, float]] = []
    for idx in filtered_indices:
        det = {
            "bbox": adjust_bbox_to_original(boxes[idx], meta),
            "score": round(scores[idx], 6),
            "class_id": class_ids[idx],
            "class_name": COCO_CLASSES[class_ids[idx]] if class_ids[idx] < len(COCO_CLASSES) else str(class_ids[idx]),
        }
        detections.append(det)
    return detections



def adjust_bbox_to_original(box_xyxy: Sequence[float], meta: Dict[str, float]) -> List[int]:
    x1, y1, x2, y2 = box_xyxy
    x1 = clamp(x1 - meta["x_pad"], 0, meta["model_w"]) / meta["scale"]
    y1 = clamp(y1 - meta["y_pad"], 0, meta["model_h"]) / meta["scale"]
    x2 = clamp(x2 - meta["x_pad"], 0, meta["model_w"]) / meta["scale"]
    y2 = clamp(y2 - meta["y_pad"], 0, meta["model_h"]) / meta["scale"]
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def draw_detections(image: np.ndarray, detections: Sequence[Dict[str, float]]) -> np.ndarray:
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['score']:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - text_h - 4)), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 2, max(12, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    return vis


def collect_images(dataset_dir: Path) -> List[Path]:
    images = [p for p in sorted(dataset_dir.iterdir()) if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    if not images:
        raise SystemExit(f"No input images found under {dataset_dir}")
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOX ONNX inference over a dataset.")
    parser.add_argument("--model", type=Path, default=Path("data/model/yolox_s.onnx"), help="Path to the ONNX model file")
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"), help="Directory containing input images")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_onnx"), help="Directory to store results")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Objectness/class probability threshold")
    parser.add_argument("--nms-threshold", type=float, default=0.5, help="IoU threshold used during per-class NMS")
    parser.add_argument("--max-detections", type=int, default=100, help="Maximum detections kept per image after NMS")
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Optional custom ONNX Runtime providers (defaults to CUDA if available, otherwise CPU).",
    )
    parser.add_argument(
        "--no-save-vis",
        dest="save_vis",
        action="store_false",
        help="Skip writing visualized JPEGs (only JSON results will be produced).",
    )
    parser.set_defaults(save_vis=True)
    return parser.parse_args()


def resolve_providers(explicit: Sequence[str] | None) -> List[str]:
    if explicit:
        return list(explicit)
    preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    resolved = [p for p in preferred if p in available]
    if not resolved:
        raise SystemExit("No compatible ONNX Runtime execution provider is available.")
    return resolved


def build_session(model_path: Path, providers: Sequence[str] | None) -> Tuple[ort.InferenceSession, str, int]:
    sess_opts = ort.SessionOptions()
    sess_opts.enable_mem_pattern = False
    sess_opts.enable_cpu_mem_arena = True
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    num_threads = min(8, max(1, os.cpu_count() or 1))
    sess_opts.intra_op_num_threads = num_threads
    sess_opts.inter_op_num_threads = max(1, num_threads // 2)

    resolved_providers = resolve_providers(providers)
    print(f"[INFO] Using ONNX Runtime providers: {resolved_providers}")
    session = ort.InferenceSession(model_path.as_posix(), sess_opts, providers=resolved_providers)

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    _, _, height, width = input_meta.shape
    if height != width:
        raise SystemExit("Only square inputs are supported at the moment.")
    return session, input_name, int(width)


def main() -> None:
    args = parse_args()
    session, input_name, input_size = build_session(args.model, args.providers)
    images = collect_images(args.dataset)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "visualizations"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": args.model.as_posix(),
        "dataset": args.dataset.as_posix(),
        "input_size": input_size,
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "images": [],
    }

    for image_path in images:
        image = cv2.imread(image_path.as_posix())
        if image is None:
            print(f"[WARN] Failed to read {image_path}, skipping.", file=sys.stderr)
            continue

        blob, meta = preprocess(image, input_size)
        raw_outputs = session.run(None, {input_name: blob})
        detections = decode_outputs(raw_outputs, meta, args.score_threshold, args.nms_threshold, args.max_detections)

        results["images"].append(
            {
                "image_id": image_path.name,
                "image_path": image_path.as_posix(),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "detections": detections,
            }
        )

        if args.save_vis:
            vis = draw_detections(image, detections)
            out_path = vis_dir / f"{image_path.stem}.jpg"
            cv2.imwrite(out_path.as_posix(), vis)

        print(f"Processed {image_path.name}: {len(detections)} detections")

    json_path = args.output_dir / "detections.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved detection summary to {json_path}")
    if args.save_vis:
        print(f"Annotated images stored under {vis_dir}")


if __name__ == "__main__":
    main()
