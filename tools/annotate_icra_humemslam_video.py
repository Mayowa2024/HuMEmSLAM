#!/usr/bin/env python3
"""Render an anonymous HuMemSLAM perception overlay for the ICRA video."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

YELLOW = (9, 201, 249)
CYAN = (235, 205, 70)
WHITE = (245, 245, 245)
TEXT_CLASSES = {
    "advertisement_sign", "store_sign", "information_sign", "traffic_sign",
    "building", "wall", "fence",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path,
        default=ROOT / "weights/humanSLAM_YOLO_seg.pt",
    )
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--detector-interval", type=int, default=2)
    parser.add_argument("--ocr-interval", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--target-mb", type=float, default=18.8)
    return parser.parse_args()


def initialise_ocr():
    try:
        from slam.ocr_onnx_runtime import OnnxPPOCRv5
        yaml_path = (
            Path.home() / ".paddlex/official_models/"
            "en_PP-OCRv5_mobile_rec/inference.yml"
        )
        return OnnxPPOCRv5(
            ROOT / "weights/ocr_onnx/PP-OCRv5_mobile_det.onnx",
            ROOT / "weights/ocr_onnx/en_PP-OCRv5_mobile_rec.onnx",
            yaml_path,
            confidence_threshold=0.60,
            detector_side=640,
            providers=["CPUExecutionProvider"],
        )
    except Exception as error:
        print(f"OCR unavailable: {error}", file=sys.stderr)
        return None


def mask_overlay(frame, result):
    overlay = frame.copy()
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        for mask in masks:
            resized = cv2.resize(mask, (frame.shape[1], frame.shape[0])) > 0.5
            overlay[resized] = (
                0.65 * overlay[resized] + 0.35 * np.asarray(YELLOW)
            ).astype(np.uint8)
    return cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)


def detection_rows(result):
    rows = []
    if result.boxes is None:
        return rows
    xyxy = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    for box, class_id, confidence in zip(xyxy, classes, confidences):
        rows.append({
            "box": tuple(int(value) for value in box),
            "class": result.names[int(class_id)],
            "confidence": float(confidence),
            "ocr": [],
        })
    return rows


def run_ocr(frame, detections, ocr):
    if ocr is None:
        return
    candidates = [row for row in detections if row["class"] in TEXT_CLASSES]
    candidates.sort(
        key=lambda row: (row["box"][2] - row["box"][0])
        * (row["box"][3] - row["box"][1]),
        reverse=True,
    )
    for row in candidates[:3]:
        x0, y0, x1, y1 = row["box"]
        margin = 8
        crop = frame[max(0, y0 - margin):min(frame.shape[0], y1 + margin),
                     max(0, x0 - margin):min(frame.shape[1], x1 + margin)]
        if crop.size:
            row["ocr"] = [anchor.text for anchor in ocr.predict_anchors(crop)]


def draw_box_label(frame, row):
    x0, y0, x1, y1 = row["box"]
    cv2.rectangle(frame, (x0, y0), (x1, y1), YELLOW, 2, cv2.LINE_AA)
    label = f"{row['class'].replace('_', ' ')}  {row['confidence']:.2f}"
    if row["ocr"]:
        label += f"  |  text: {' / '.join(row['ocr'])}"
    (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    top = max(height + 6, y0)
    cv2.rectangle(frame, (x0, top - height - 6),
                  (min(frame.shape[1] - 1, x0 + width + 8), top), (20, 24, 27), -1)
    cv2.putText(frame, label, (x0 + 4, top - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, WHITE, 1, cv2.LINE_AA)


def draw_orb(frame, orb):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = sorted(orb.detect(gray, None), key=lambda keypoint: keypoint.response,
                    reverse=True)[:220]
    for point in points:
        x, y = (int(round(value)) for value in point.pt)
        cv2.circle(frame, (x, y), 1, CYAN, -1, cv2.LINE_AA)
    return len(points)


def draw_legend(frame, object_count, orb_count):
    x0, y0, width, height = frame.shape[1] - 294, 18, 276, 72
    region = frame[y0:y0 + height, x0:x0 + width]
    backing = np.full_like(region, (18, 22, 25))
    frame[y0:y0 + height, x0:x0 + width] = cv2.addWeighted(
        region, 0.28, backing, 0.72, 0
    )
    cv2.circle(frame, (x0 + 14, y0 + 21), 3, CYAN, -1, cv2.LINE_AA)
    cv2.putText(frame, f"ORB keypoints  {orb_count}", (x0 + 27, y0 + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0 + 10, y0 + 44), (x0 + 18, y0 + 52), YELLOW, 2)
    cv2.putText(frame, f"Stable landmarks  {object_count}", (x0 + 27, y0 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, WHITE, 1, cv2.LINE_AA)


def render(args, intermediate):
    capture = cv2.VideoCapture(str(args.input))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_seconds > 0:
        total = min(total, int(round(args.max_seconds * fps)))
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "14", "-pix_fmt", "yuv420p", str(intermediate),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    model = YOLO(str(args.model))
    ocr = initialise_ocr()
    orb = cv2.ORB_create(nfeatures=700, fastThreshold=20)
    detections = []
    for index in range(total):
        ok, frame = capture.read()
        if not ok:
            break
        clean = frame.copy()
        if index % max(1, args.detector_interval) == 0:
            result = model.predict(
                frame, imgsz=640, conf=args.confidence, device="cpu",
                verbose=False,
            )[0]
            detections = detection_rows(result)
            if index % max(1, args.ocr_interval) == 0:
                run_ocr(clean, detections, ocr)
            frame = mask_overlay(frame, result)
        for row in detections:
            draw_box_label(frame, row)
        orb_count = draw_orb(frame, orb)
        draw_legend(frame, len(detections), orb_count)
        encoder.stdin.write(frame.tobytes())
        if index % 200 == 0:
            print(f"Rendered {index}/{total}", flush=True)
    capture.release()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("Intermediate video encoding failed")
    return fps


def encode_upload(intermediate, output, fps, target_mb):
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(intermediate),
    ], text=True)
    duration = float(probe.strip())
    bitrate = int(target_mb * 8_000_000 / duration)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="icra_passlog_") as temp_name:
        passlog = str(Path(temp_name) / "pass")
        common = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(intermediate),
            "-vf", "scale=854:480:flags=lanczos", "-an", "-c:v", "libx264",
            "-preset", "slow", "-b:v", str(bitrate), "-maxrate", str(bitrate),
            "-bufsize", str(bitrate * 2), "-r", str(fps), "-pix_fmt", "yuv420p",
        ]
        subprocess.run(common + ["-pass", "1", "-passlogfile", passlog,
                                  "-f", "mp4", "/dev/null"], check=True)
        subprocess.run(common + ["-pass", "2", "-passlogfile", passlog,
                                  "-movflags", "+faststart", str(output)], check=True)


def main():
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="icra_humem_overlay_") as temp_name:
        intermediate = Path(temp_name) / "annotated.mp4"
        fps = render(args, intermediate)
        encode_upload(intermediate, args.output, fps, args.target_mb)
    print(args.output)


if __name__ == "__main__":
    main()
