#!/usr/bin/env python3
"""Convert selected Mapillary Vistas v2.0 polygons to YOLO segmentation."""

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = [
    "building",
    "bridge",
    "advertisement_sign",
    "store_sign",
    "information_sign",
    "traffic_sign",
    "traffic_light",
    "wall",
    "fence",
    "guard_rail",
    "tunnel",
    "street_light",
    "pole",
    "bench",
    "bike_rack",
    "cctv_camera",
    "fire_hydrant",
    "junction_box",
    "mailbox",
    "parking_meter",
    "phone_booth",
]

CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

MAPILLARY_TO_CLASS = {
    "construction--structure--building": "building",
    "construction--structure--garage": "building",
    "construction--structure--bridge": "bridge",
    "object--banner": "advertisement_sign",
    "object--sign--advertisement": "advertisement_sign",
    "object--sign--store": "store_sign",
    "object--sign--information": "information_sign",
    "object--traffic-sign--direction-front": "traffic_sign",
    "object--traffic-sign--front": "traffic_sign",
    "object--traffic-sign--information-parking": "traffic_sign",
    "object--traffic-light--general-single": "traffic_light",
    "object--traffic-light--pedestrians": "traffic_light",
    "object--traffic-light--general-upright": "traffic_light",
    "object--traffic-light--general-horizontal": "traffic_light",
    "object--traffic-light--cyclists": "traffic_light",
    "object--traffic-light--other": "traffic_light",
    "construction--barrier--wall": "wall",
    "construction--barrier--fence": "fence",
    "construction--barrier--guard-rail": "guard_rail",
    "construction--structure--tunnel": "tunnel",
    "object--street-light": "street_light",
    "object--support--pole": "pole",
    "object--support--pole-group": "pole",
    "object--support--utility-pole": "pole",
    "object--bench": "bench",
    "object--bike-rack": "bike_rack",
    "object--cctv-camera": "cctv_camera",
    "object--fire-hydrant": "fire_hydrant",
    "object--junction-box": "junction_box",
    "object--mailbox": "mailbox",
    "object--parking-meter": "parking_meter",
    "object--phone-booth": "phone_booth",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapillary-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--splits", nargs="+", choices=("training", "validation"),
        default=("training", "validation"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=1e-7,
        help="Discard polygons below this fraction of image area.",
    )
    parser.add_argument(
        "--simplify-epsilon",
        type=float,
        default=0.0,
        help="Optional polygon approximation as a fraction of image diagonal.",
    )
    parser.add_argument("--preview-count", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2.0


def clean_polygon(raw_points, width, height, simplify_epsilon=0.0):
    if width <= 0 or height <= 0 or len(raw_points) < 3:
        return None
    points = np.asarray(raw_points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(points).all():
        return None
    points[:, 0] = np.clip(points[:, 0], 0.0, width - 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, height - 1.0)
    keep = np.ones(len(points), dtype=bool)
    if len(points) > 1:
        keep[1:] = np.any(np.abs(np.diff(points, axis=0)) > 1e-9, axis=1)
    points = points[keep]
    if len(points) >= 2 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if simplify_epsilon > 0.0 and len(points) >= 4:
        epsilon = simplify_epsilon * float(np.hypot(width, height))
        points = cv2.approxPolyDP(
            points.astype(np.float32), epsilon, True
        ).reshape(-1, 2).astype(np.float64)
    if len(points) < 3 or polygon_area(points) <= 0.0:
        return None
    points[:, 0] /= float(width)
    points[:, 1] /= float(height)
    return np.clip(points, 0.0, 1.0)


def image_for_stem(image_dir, stem):
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def prepare_output(output, root, overwrite):
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)
    (output / "labels" / "train").mkdir(parents=True)
    (output / "labels" / "val").mkdir(parents=True)
    (output / "previews").mkdir(parents=True)
    (output / "reports").mkdir(parents=True)
    (output / "images").mkdir(parents=True)
    for split, short_name in (("training", "train"), ("validation", "val")):
        os.symlink(
            (root / split / "images").resolve(),
            output / "images" / short_name,
            target_is_directory=True,
        )


def write_yaml(output):
    lines = [
        f"path: {output.resolve()}",
        "train: train.txt",
        "val: val.txt",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (output / "mapillary_humanslam.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def colour_for_class(class_id):
    rng = np.random.default_rng(class_id + 2026)
    return tuple(int(value) for value in rng.integers(40, 256, size=3))


def make_preview(image_path, annotations, output_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False
    overlay = image.copy()
    height, width = image.shape[:2]
    for class_id, points in annotations:
        pixels = np.column_stack(
            (points[:, 0] * width, points[:, 1] * height)
        ).round().astype(np.int32)
        colour = colour_for_class(class_id)
        cv2.fillPoly(overlay, [pixels], colour)
        anchor = tuple(pixels[0])
        cv2.putText(
            image,
            CLASS_NAMES[class_id],
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.45, min(width, height) / 1600.0),
            colour,
            2,
            cv2.LINE_AA,
        )
    preview = cv2.addWeighted(overlay, 0.38, image, 0.62, 0.0)
    max_width = 1600
    if preview.shape[1] > max_width:
        scale = max_width / preview.shape[1]
        preview = cv2.resize(preview, None, fx=scale, fy=scale)
    return cv2.imwrite(str(output_path), preview)


def convert_split(args, split, preview_budget):
    short_name = "train" if split == "training" else "val"
    polygon_dir = args.mapillary_root / split / "v2.0" / "polygons"
    image_dir = args.mapillary_root / split / "images"
    files = sorted(polygon_dir.glob("*.json"))
    if args.limit > 0:
        random.Random(args.seed + (0 if split == "training" else 1)).shuffle(files)
        files = sorted(files[: args.limit])

    class_objects = Counter()
    class_images = Counter()
    skipped = Counter()
    image_rows = []
    preview_candidates = []

    for annotation_path in files:
        try:
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped["invalid_json"] += 1
            continue
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        image_path = image_for_stem(image_dir, annotation_path.stem)
        if image_path is None:
            skipped["missing_image"] += 1
            continue

        converted = []
        present = set()
        for obj in data.get("objects", []):
            target_name = MAPILLARY_TO_CLASS.get(obj.get("label"))
            if target_name is None:
                continue
            points = clean_polygon(
                obj.get("polygon", []),
                width,
                height,
                args.simplify_epsilon,
            )
            if points is None:
                skipped["invalid_polygon"] += 1
                continue
            if polygon_area(points) < args.min_area_ratio:
                skipped["tiny_polygon"] += 1
                continue
            class_id = CLASS_TO_ID[target_name]
            converted.append((class_id, points))
            class_objects[target_name] += 1
            present.add(target_name)

        for name in present:
            class_images[name] += 1
        label_path = args.output / "labels" / short_name / f"{annotation_path.stem}.txt"
        with label_path.open("w", encoding="utf-8") as stream:
            for class_id, points in converted:
                coordinates = " ".join(
                    f"{value:.6f}" for value in points.reshape(-1)
                )
                stream.write(f"{class_id} {coordinates}\n")
        image_rows.append(
            {
                "stem": annotation_path.stem,
                "objects": len(converted),
                "classes": ";".join(sorted(present)),
            }
        )
        if converted:
            preview_candidates.append((image_path, annotation_path.stem, converted))

    rng = random.Random(args.seed + (10 if split == "training" else 11))
    rng.shuffle(preview_candidates)
    for image_path, stem, converted in preview_candidates[:preview_budget]:
        make_preview(
            image_path,
            converted,
            args.output / "previews" / f"{short_name}_{stem}.jpg",
        )

    with (args.output / "reports" / f"{short_name}_images.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=("stem", "objects", "classes"))
        writer.writeheader()
        writer.writerows(image_rows)
    (args.output / f"{short_name}.txt").write_text(
        "".join(



            f"{args.output / 'images' / short_name / (row['stem'] + image_for_stem(image_dir, row['stem']).suffix)}\n"
            for row in image_rows
        ),
        encoding="utf-8",
    )

    return {
        "source_annotations": len(files),
        "written_labels": len(image_rows),
        "nonempty_labels": sum(row["objects"] > 0 for row in image_rows),
        "empty_labels": sum(row["objects"] == 0 for row in image_rows),
        "class_objects": dict(class_objects),
        "class_images": dict(class_images),
        "skipped": dict(skipped),
    }


def main():
    args = parse_args()
    args.mapillary_root = args.mapillary_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.min_area_ratio < 0.0 or args.simplify_epsilon < 0.0:
        raise ValueError("Area and simplification thresholds must be non-negative")
    prepare_output(args.output, args.mapillary_root, args.overwrite)
    write_yaml(args.output)
    report = {
        "mapillary_root": str(args.mapillary_root),
        "output": str(args.output),
        "seed": args.seed,
        "limit_per_split": args.limit,
        "min_area_ratio": args.min_area_ratio,
        "simplify_epsilon": args.simplify_epsilon,
        "class_names": CLASS_NAMES,
        "mapillary_mapping": MAPILLARY_TO_CLASS,
        "splits": {},
    }
    per_split_previews = max(1, args.preview_count // max(1, len(args.splits)))
    for split in args.splits:
        report["splits"][split] = convert_split(args, split, per_split_previews)
    (args.output / "reports" / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
