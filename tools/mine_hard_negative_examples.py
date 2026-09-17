#!/usr/bin/env python3
"""Mine and render auditable hard-negative examples from local datasets."""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from slam.global_place_descriptor import DescriptorSpec, TensorRTGlobalDescriptor


def load_kitti_positions(path):
    positions = []
    for line in path.read_text().splitlines():
        values = [float(value) for value in line.split()]
        transform = np.asarray(values, dtype=np.float64).reshape(3, 4)
        positions.append(transform[:, 3])
    return np.asarray(positions)


def mine_existing_kitti(csv_path, pose_path, count, min_distance, min_frames):
    positions = load_kitti_positions(pose_path)
    candidates = []
    with csv_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            query = int(row["query_frame_id"])
            candidate = int(row["candidate_source_frame_id"])
            if max(query, candidate) >= len(positions):
                continue
            distance = float(np.linalg.norm(positions[query] - positions[candidate]))
            if distance < min_distance or abs(query - candidate) < min_frames:
                continue
            candidates.append({
                "source": "KITTI 06 old HuMemSLAM proposal",
                "query_path": row["query_image_path"],
                "candidate_path": row["candidate_image_path"],
                "query_id": query,
                "candidate_id": candidate,
                "similarity": float(row["unified_score"]),
                "distance_m": distance,
            })
    candidates.sort(key=lambda row: row["similarity"], reverse=True)
    selected, used_queries, used_candidates = [], set(), set()
    for row in candidates:
        if row["query_id"] in used_queries or row["candidate_id"] in used_candidates:
            continue
        if not Path(row["query_path"]).is_file() or not Path(row["candidate_path"]).is_file():
            continue
        selected.append(row)
        used_queries.add(row["query_id"])
        used_candidates.add(row["candidate_id"])
        if len(selected) == count:
            break
    return selected


def evenly_sample(directory, count):
    files = sorted(directory.glob("*.png"))
    indices = np.linspace(0, len(files) - 1, min(count, len(files)), dtype=int)
    return [files[index] for index in indices]


def describe_paths(runtime, paths):
    descriptors = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        descriptors.append(runtime.describe(image))
    return np.stack(descriptors)


def load_4seasons_positions(path):
    rows = np.loadtxt(path, delimiter=",", comments="#")
    return rows[:, 0].astype(np.int64), rows[:, 1:4]


def mine_4seasons_route(images, poses_path, spec_path, samples, count,
                       min_distance):
    first_paths = evenly_sample(images, samples)
    runtime = TensorRTGlobalDescriptor(DescriptorSpec.from_json(spec_path))
    first_descriptors = describe_paths(runtime, first_paths)
    similarities = first_descriptors @ first_descriptors.T
    pose_times, pose_positions = load_4seasons_positions(poses_path)
    image_positions = []
    for image_path in first_paths:
        timestamp = int(image_path.stem)
        pose_index = int(np.argmin(np.abs(pose_times - timestamp)))
        image_positions.append(pose_positions[pose_index])
    image_positions = np.asarray(image_positions)
    distances = np.linalg.norm(
        image_positions[:, None, :] - image_positions[None, :, :], axis=2
    )
    similarities[distances < min_distance] = -np.inf
    np.fill_diagonal(similarities, -np.inf)
    flattened = np.argsort(-similarities, axis=None)
    selected, used_images = [], set()
    for flat_index in flattened:
        first_index, second_index = np.unravel_index(flat_index, similarities.shape)
        if not np.isfinite(similarities[first_index, second_index]):
            break
        if first_index in used_images or second_index in used_images:
            continue
        selected.append({
            "source": "4Seasons neighbourhood 1 same-route pose-labelled",
            "query_path": str(first_paths[first_index]),
            "candidate_path": str(first_paths[second_index]),
            "query_id": first_paths[first_index].stem,
            "candidate_id": first_paths[second_index].stem,
            "similarity": float(similarities[first_index, second_index]),
            "distance_m": float(distances[first_index, second_index]),
        })
        used_images.update((first_index, second_index))
        if len(selected) == count:
            break
    return selected


def fit_image(image, width=640, height=360):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def render(rows, output_path, title):
    width, image_height, text_height = 640, 360, 76
    canvas = np.full(
        (58 + len(rows) * (image_height + text_height), width * 2, 3),
        245, dtype=np.uint8,
    )
    cv2.putText(canvas, title, (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (20, 20, 20), 2, cv2.LINE_AA)
    for index, row in enumerate(rows):
        y = 58 + index * (image_height + text_height)
        query = fit_image(cv2.imread(row["query_path"]))
        candidate = fit_image(cv2.imread(row["candidate_path"]))
        canvas[y:y + image_height, :width] = query
        canvas[y:y + image_height, width:] = candidate
        distance = row["distance_m"]
        if isinstance(distance, float):
            distance = f"{distance:.1f} m apart"
        labels = [
            f"QUERY frame {row['query_id']}",
            f"HARD-NEGATIVE frame {row['candidate_id']}",
            f"score={row['similarity']:.3f} | {distance}",
        ]
        text_y = y + image_height + 27
        cv2.putText(canvas, labels[0], (10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, labels[1], (width + 10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, labels[2], (10, text_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 60, 60), 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def write_manifest(rows, path):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-candidates", required=True, type=Path)
    parser.add_argument("--kitti-poses", required=True, type=Path)
    parser.add_argument("--neighbourhood-1", required=True, type=Path)
    parser.add_argument("--neighbourhood-1-poses", required=True, type=Path)
    parser.add_argument("--engine-spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--min-distance", type=float, default=20.0)
    parser.add_argument("--min-frames", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    kitti = mine_existing_kitti(
        args.kitti_candidates, args.kitti_poses, args.count,
        args.min_distance, args.min_frames,
    )
    seasons = mine_4seasons_route(
        args.neighbourhood_1, args.neighbourhood_1_poses, args.engine_spec,
        args.samples, args.count, args.min_distance,
    )
    if not kitti or not seasons:
        raise RuntimeError("Hard-negative mining produced an empty result")
    write_manifest(kitti, args.output / "kitti06_examples.csv")
    write_manifest(seasons, args.output / "4seasons_pose_labelled_examples.csv")
    render(kitti, args.output / "kitti06_hard_negatives.png",
           "KITTI 06: high-scoring HuMemSLAM proposals proven >20 m away")
    render(seasons, args.output / "4seasons_pose_labelled_hard_negatives.png",
           "4Seasons neighbourhood 1: high MixVPR similarity but >20 m apart")
    print(f"Wrote hard-negative examples to {args.output}")


if __name__ == "__main__":
    main()
