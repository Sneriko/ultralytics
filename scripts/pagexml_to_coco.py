#!/usr/bin/env python3
"""Convert PAGE-XML files to a COCO instance-segmentation dataset.

The converter recursively scans a root directory for PAGE-XML files located in
folders named "page" (case-insensitive), builds COCO image/annotation records
from region polygons, and writes a single COCO JSON file.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="Root directory to scan")
    parser.add_argument("output_json", type=Path, help="Output COCO annotations JSON path")
    parser.add_argument(
        "--exclude-prefix",
        type=str,
        default="",
        help="Exclude folders whose basename starts with this prefix",
    )
    parser.add_argument(
        "--page-folder-name",
        type=str,
        default="page",
        help="Only parse XML files under folders with this name (default: page)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum polygon points required to keep an annotation",
    )
    return parser.parse_args()


def should_exclude(path: Path, prefix: str) -> bool:
    return bool(prefix) and any(part.startswith(prefix) for part in path.parts)


def iter_pagexml_files(root: Path, page_folder_name: str, exclude_prefix: str):
    target = page_folder_name.lower()
    for xml_file in root.rglob("*.xml"):
        if should_exclude(xml_file, exclude_prefix):
            continue
        if target not in {p.lower() for p in xml_file.parts}:
            continue
        yield xml_file


def parse_points(points_raw: str):
    pts = []
    for pair in points_raw.split():
        x_str, y_str = pair.split(",")
        pts.append((float(x_str), float(y_str)))
    return pts


def polygon_area_and_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    area2 = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area2 += x1 * y2 - x2 * y1
    area = abs(area2) * 0.5
    return area, [xmin, ymin, xmax - xmin, ymax - ymin]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def main() -> None:
    args = parse_args()

    categories = {}
    images = []
    annotations = []
    per_image_ann_count = defaultdict(int)

    image_id = 1
    ann_id = 1

    for xml_path in sorted(iter_pagexml_files(args.input_root, args.page_folder_name, args.exclude_prefix)):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        page_elem = next((e for e in root.iter() if local_name(e.tag) == "Page"), None)
        if page_elem is None:
            continue

        width = int(page_elem.attrib.get("imageWidth", 0))
        height = int(page_elem.attrib.get("imageHeight", 0))
        file_name = page_elem.attrib.get("imageFilename", xml_path.with_suffix(".jpg").name)

        rel_dir = xml_path.parent.relative_to(args.input_root)
        image_rel_path = str((rel_dir / file_name).as_posix())

        current_image_id = image_id
        image_id += 1

        images.append(
            {
                "id": current_image_id,
                "file_name": image_rel_path,
                "width": width,
                "height": height,
            }
        )

        for elem in root.iter():
            tag_name = local_name(elem.tag)
            if not (tag_name.endswith("Region") or tag_name == "TextLine"):
                continue

            coords = next((c for c in elem if local_name(c.tag) == "Coords"), None)
            if coords is None:
                continue

            points_raw = coords.attrib.get("points", "").strip()
            if not points_raw:
                continue

            points = parse_points(points_raw)
            if len(points) < args.min_points:
                continue

            category_name = tag_name
            if category_name not in categories:
                categories[category_name] = len(categories) + 1

            area, bbox = polygon_area_and_bbox(points)
            if area <= 0:
                continue

            segmentation = [coord for point in points for coord in point]

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": current_image_id,
                    "category_id": categories[category_name],
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            per_image_ann_count[current_image_id] += 1

    categories_list = [
        {"id": cid, "name": cname, "supercategory": "page_region"}
        for cname, cid in sorted(categories.items(), key=lambda x: x[1])
    ]

    coco = {
        "info": {"description": "PAGE-XML converted COCO dataset"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories_list,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    kept_images = sum(1 for img in images if per_image_ann_count[img["id"]] > 0)
    print(f"Converted {len(images)} PAGE-XML files")
    print(f"Created {len(annotations)} annotations across {kept_images} labeled images")
    print(f"Saved COCO JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
