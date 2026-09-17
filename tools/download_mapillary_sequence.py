#!/usr/bin/env python3
"""Download one Mapillary sequence with ordered metadata.

The access token is read from MAPILLARY_ACCESS_TOKEN and is never written to
disk. Images and metadata are intended as local experiment inputs.
"""

import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


GRAPH_URL = "https://graph.mapillary.com"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--quality", choices=("1024", "2048", "original"), default="2048")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum images to download; zero downloads all")
    return parser.parse_args()


def get_json(url, token):
    separator = "&" if "?" in url else "?"
    request = urllib.request.Request(
        f"{url}{separator}{urllib.parse.urlencode({'access_token': token})}",
        headers={"User-Agent": "HuMemSLAM-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def sequence_image_ids(sequence_id, token):
    query = urllib.parse.urlencode({"sequence_id": sequence_id, "limit": 1000})
    url = f"{GRAPH_URL}/image_ids?{query}"
    identifiers = []
    while url:
        payload = get_json(url, token)
        identifiers.extend(str(item["id"]) for item in payload.get("data", []))
        url = payload.get("paging", {}).get("next")
    return identifiers


def image_metadata(image_id, quality, token):
    fields = ",".join((
        "id", "captured_at", "compass_angle", "geometry", "sequence",
        f"thumb_{quality}_url",
    ))
    query = urllib.parse.urlencode({"fields": fields})
    return get_json(f"{GRAPH_URL}/{image_id}?{query}", token)


def download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "HuMemSLAM-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def main():
    args = arguments()
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "MAPILLARY_ACCESS_TOKEN is not set. Export it in this terminal, "
            "then rerun the command."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)

    identifiers = sequence_image_ids(args.sequence_id, token)
    if args.limit:
        identifiers = identifiers[:args.limit]
    if not identifiers:
        raise SystemExit(f"No images returned for sequence {args.sequence_id}")

    records = []
    url_field = f"thumb_{args.quality}_url"
    for index, image_id in enumerate(identifiers):
        metadata = image_metadata(image_id, args.quality, token)
        coordinates = metadata.get("geometry", {}).get("coordinates", [None, None])
        filename = f"{index:06d}_{image_id}.jpg"
        path = image_dir / filename
        if not path.exists():
            download(metadata[url_field], path)
        records.append({
            "sequence_index": index,
            "image_id": image_id,
            "captured_at_ms": metadata.get("captured_at"),
            "longitude": coordinates[0],
            "latitude": coordinates[1],
            "compass_angle_deg": metadata.get("compass_angle"),
            "filename": filename,
            "mapillary_url": f"https://www.mapillary.com/app/?pKey={image_id}",
        })
        print(f"[{index + 1}/{len(identifiers)}] {filename}", flush=True)
        time.sleep(0.03)

    fields = list(records[0])
    with (args.output / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (args.output / "download_manifest.json").write_text(
        json.dumps({
            "sequence_id": args.sequence_id,
            "quality": args.quality,
            "image_count": len(records),
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
