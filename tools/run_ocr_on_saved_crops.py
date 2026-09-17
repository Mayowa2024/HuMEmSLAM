#!/usr/bin/env python3
"""Run unfiltered PaddleOCR diagnostics on crops already saved to disk."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def dictionary(result):
    if isinstance(result, dict):
        value = result
    elif hasattr(result, "json"):
        value = result.json() if callable(result.json) else result.json
    elif hasattr(result, "to_dict"):
        value = result.to_dict() if callable(result.to_dict) else result.to_dict
    else:
        return {}
    if isinstance(value, dict) and isinstance(value.get("res"), dict):
        return value["res"]
    return value if isinstance(value, dict) else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("crop_dir", type=Path)
    parser.add_argument("--device", default="gpu:0")
    args = parser.parse_args()
    ocr = PaddleOCR(
        lang="en", device=args.device,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, enable_mkldnn=False,
        text_det_limit_side_len=640, text_det_limit_type="max",
    )
    report = []
    for path in sorted(args.crop_dir.glob("*.png")):
        image = cv2.imread(str(path))
        raw = [dictionary(item) for item in ocr.predict(image)]
        texts = []
        for item in raw:
            for text, score in zip(item.get("rec_texts", []),
                                   item.get("rec_scores", [])):
                texts.append({"text": str(text), "confidence": float(score)})
        report.append({"crop": str(path), "ocr": texts,
                       "raw_result": plain(raw)})
        print(path.name, texts)
    output = args.crop_dir / "ocr_crop_diagnostics.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
