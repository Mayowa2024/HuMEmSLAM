#!/usr/bin/env python3
"""Compare rectangular and mask-isolated OCR crops without score filtering."""

import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=["traffic_sign"])
    parser.add_argument("--yolo-confidence", type=float, default=0.5)
    parser.add_argument("--margin", type=int, default=10)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--device", default="gpu:0")
    return parser.parse_args()


def result_dictionary(result):
    if isinstance(result, dict):
        value = result
    elif hasattr(result, "json"):
        value = result.json() if callable(result.json) else result.json
    elif hasattr(result, "to_dict"):
        value = result.to_dict() if callable(result.to_dict) else result.to_dict
    else:
        return {}
    if isinstance(value, dict) and isinstance(value.get("res"), dict):
        value = value["res"]
    return value if isinstance(value, dict) else {}


def serialisable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialisable(v) for v in value]
    return value


def enhance(image, scale):
    enlarged = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )
    lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(str(args.yolo), task="segment")
    report = []
    pending = []
    for image_path in args.images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        results = yolo(image, conf=args.yolo_confidence, verbose=False)
        sign_number = 0
        for result in results:
            if result.boxes is None:
                continue
            for index, box in enumerate(result.boxes):
                class_name = yolo.names[int(box.cls[0])]
                if class_name not in args.classes:
                    continue
                sign_number += 1
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1 = max(0, x1 - args.margin), max(0, y1 - args.margin)
                x2, y2 = min(width, x2 + args.margin), min(height, y2 + args.margin)
                box_crop = image[y1:y2, x1:x2].copy()
                mask_crop = box_crop.copy()
                if result.masks is not None:
                    mask = result.masks.data[index].cpu().numpy()
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    mask_crop[mask[y1:y2, x1:x2] <= 0.5] = 255
                variants = {
                    "box": box_crop,
                    "mask": mask_crop,
                    "box_enhanced": enhance(box_crop, args.scale),
                    "mask_enhanced": enhance(mask_crop, args.scale),
                }
                for variant, crop in variants.items():
                    stem = f"{image_path.stem}_sign{sign_number}_{variant}"
                    crop_path = args.output / f"{stem}.png"
                    cv2.imwrite(str(crop_path), crop)
                    pending.append((image_path, class_name, confidence,
                                    [x1, y1, x2, y2], variant, crop_path))




    del yolo
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    ocr = PaddleOCR(
        lang="en", device=args.device,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, enable_mkldnn=False,
        text_det_limit_side_len=640, text_det_limit_type="max",
    )
    for image_path, class_name, confidence, box, variant, crop_path in pending:
        crop = cv2.imread(str(crop_path))
        raw = [result_dictionary(item) for item in ocr.predict(crop)]
        texts = []
        for item in raw:
            for text, score in zip(
                item.get("rec_texts", []), item.get("rec_scores", [])
            ):
                texts.append({"text": str(text), "confidence": float(score)})
        report.append({
            "image": str(image_path), "class": class_name,
            "yolo_confidence": confidence, "box": box,
            "variant": variant, "crop": str(crop_path),
            "ocr": texts, "raw_result": serialisable(raw),
        })
    (args.output / "ocr_crop_diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(args.output / "ocr_crop_diagnostics.json")


if __name__ == "__main__":
    main()
