#!/usr/bin/env python3
"""Convert PAGE-XML files to a YOLO segmentation dataset with train/val splits."""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="Root directory to scan")
    parser.add_argument("output_root", type=Path, help="Output dataset root path")
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Exclude folders whose basename starts with this prefix (repeatable)",
    )
    parser.add_argument(
        "--page-folder-name", type=str, default="page", help="Only parse XML files under folders with this name"
    )
    parser.add_argument(
        "--min-points", type=int, default=3, help="Minimum polygon points required to keep an annotation"
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--copy-images", action="store_true", help="Copy images into output_root/images/{train,val}")
    return parser.parse_args()


def should_exclude(path: Path, prefixes: list[str]) -> bool:
    return any(part.startswith(prefix) for prefix in prefixes if prefix for part in path.parts)


def iter_pagexml_files(root: Path, page_folder_name: str, exclude_prefixes: list[str]):
    target = page_folder_name.lower()
    for xml_file in root.rglob("*.xml"):
        if should_exclude(xml_file, exclude_prefixes):
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


def polygon_area(points):
    area2 = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def resolve_image_path(xml_path: Path, image_filename: str, search_root: Path | None = None) -> Path:
    """Resolve image path referenced by PAGE-XML.

    PAGE-XML files are often stored under a `page/` directory while images live one level above it, so check both
    locations before falling back to a repository-wide filename search.
    """
    candidate = Path(image_filename)
    if candidate.is_absolute():
        return candidate

    # Most specific first: same dir as XML, then parent dir (e.g. ../image.jpg).
    local = xml_path.parent / candidate
    if local.exists():
        return local

    parent_local = xml_path.parent.parent / candidate
    if parent_local.exists():
        return parent_local

    # Final fallback: find unique matching filename under input root.
    if search_root is not None and search_root.exists():
        matches = list(search_root.rglob(candidate.name))
        if len(matches) == 1:
            return matches[0]

    return local


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    categories: dict[str, int] = {}
    records = []
    malformed_xml_paths = []

    for xml_path in sorted(iter_pagexml_files(args.input_root, args.page_folder_name, args.exclude_prefix)):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            malformed_xml_paths.append(str(xml_path))
            continue
        root = tree.getroot()
        page_elem = next((e for e in root.iter() if local_name(e.tag) == "Page"), None)
        if page_elem is None:
            continue

        width = int(page_elem.attrib.get("imageWidth", 0))
        height = int(page_elem.attrib.get("imageHeight", 0))
        file_name = page_elem.attrib.get("imageFilename", xml_path.with_suffix(".jpg").name)
        image_path = resolve_image_path(xml_path, file_name, args.input_root)

        lines = []
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
            if polygon_area(points) <= 0:
                continue

            if tag_name not in categories:
                categories[tag_name] = len(categories)
            cls = categories[tag_name]

            norm = []
            for x, y in points:
                norm.extend([x / max(width, 1), y / max(height, 1)])
            lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in norm))

        if lines:
            records.append({"xml": xml_path, "image": image_path, "label_lines": lines})

    rng.shuffle(records)
    n_val = int(len(records) * args.val_ratio)
    val_set = set(id(r) for r in records[:n_val])

    out = args.output_root
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_counts = defaultdict(int)
    missing_images = 0
    for r in records:
        split = "val" if id(r) in val_set else "train"
        stem = r["image"].stem
        label_out = out / "labels" / split / f"{stem}.txt"
        label_out.write_text("\n".join(r["label_lines"]) + "\n", encoding="utf-8")

        image_out = out / "images" / split / r["image"].name
        if args.copy_images:
            if r["image"].exists():
                shutil.copy2(r["image"], image_out)
            else:
                missing_images += 1
        else:
            if image_out.exists() or image_out.is_symlink():
                image_out.unlink()
            if r["image"].exists():
                image_out.symlink_to(r["image"].resolve())
            else:
                missing_images += 1

        split_counts[split] += 1

    names = {idx: name for name, idx in sorted(categories.items(), key=lambda x: x[1])}
    yaml_out = out / "dataset_seg.yaml"
    yaml_out.write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": names,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    print(f"Created YOLO dataset at: {out}")
    print(f"Train images: {split_counts['train']}, Val images: {split_counts['val']}")
    print(f"Image mode: {'copy' if args.copy_images else 'symlink'}")
    print(f"Classes: {len(names)}")
    if missing_images:
        print(f"Missing source images: {missing_images}")
    if malformed_xml_paths:
        print(f"Skipped {len(malformed_xml_paths)} malformed PAGE-XML files")


if __name__ == "__main__":
    main()
