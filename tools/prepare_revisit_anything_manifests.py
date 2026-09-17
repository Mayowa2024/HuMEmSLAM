#!/usr/bin/env python3
"""Create lightweight fixed-manifest dataset views for Revisit Anything."""

import argparse
from pathlib import Path

from run_external_vpr_comparison import dataset_definition

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "external/vpr_comparison/Revisit-Anything/workdir_data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-queries", type=int, default=100)
    args = parser.parse_args()
    data = dataset_definition(args.dataset, args.max_queries)
    base = WORK / args.dataset
    for split, frames, paths in (
        ("ref", data["database_frames"], data["database_paths"]),
        ("query", data["query_frames"], data["query_paths"]),
    ):
        directory = base / split
        directory.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            target = Path(paths[frame]).resolve()
            link = directory / f"{frame:010d}{target.suffix.lower()}"
            if not link.exists():
                link.symlink_to(target)
    print(base)


if __name__ == "__main__":
    main()
