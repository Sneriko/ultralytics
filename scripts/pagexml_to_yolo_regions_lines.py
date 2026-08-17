#!/usr/bin/env python3
"""Create page-region and cropped text-line YOLO segmentation datasets from PAGE-XML files."""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create two YOLO segmentation datasets: full-page region polygons and TextLine polygons on region crops."
        )
    )
    parser.add_argument("input_root", type=Path, help="Root directory containing PAGE-XML files and page images")
    parser.add_argument("output_root", type=Path, help="Output root (creates regions/ and lines/ below it)")
    parser.add_argument("--page-folder-name", default="page", help="Only read XML files below folders with this name")
    parser.add_argument(
        "--exclude-prefix", action="append", default=[], help="Exclude paths with a component starting with this prefix"
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of pages assigned to validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for the page-level split")
    parser.add_argument("--min-points", type=int, default=3, help="Minimum polygon points per annotation")
    parser.add_argument("--crop-padding", type=int, default=0, help="Extra pixels around each region crop")
    parser.add_argument("--copy-images", action="store_true", help="Copy page images instead of creating symlinks")
    parser.add_argument(
        "--include-empty-crops", action="store_true", help="Keep region crops that contain no valid TextLine polygons"
    )
    return parser.parse_args()


def local_name(tag: str) -> str:
    """Return an XML tag without its namespace."""
    return tag.rsplit("}", 1)[-1]


def parse_points(raw: str) -> list[tuple[float, float]]:
    """Parse a PAGE points attribute, returning an empty list when it is invalid."""
    try:
        return [tuple(map(float, pair.split(","))) for pair in raw.split()]  # type: ignore[misc]
    except (TypeError, ValueError):
        return []


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Calculate the unsigned area of a polygon."""
    return (
        abs(
            sum(
                points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1]
                for i in range(len(points))
            )
        )
        / 2
    )


def element_points(element: ET.Element, min_points: int) -> list[tuple[float, float]] | None:
    """Read and validate the direct Coords child of a PAGE element."""
    coords = next((child for child in element if local_name(child.tag) == "Coords"), None)
    points = parse_points(coords.attrib.get("points", "")) if coords is not None else []
    return points if len(points) >= min_points and polygon_area(points) > 0 else None


def region_class(tag: str) -> str | None:
    """Map PAGE region elements to class names, matching pagexml_to_coco.py behavior."""
    if not tag.endswith("Region") or tag == "ImageRegion":
        return None
    return "TextRegion" if tag == "TableRegion" else tag


def excluded(path: Path, prefixes: list[str]) -> bool:
    """Return whether a path contains an excluded component."""
    return any(prefix and part.startswith(prefix) for prefix in prefixes for part in path.parts)


def pagexml_files(root: Path, folder_name: str, prefixes: list[str]):
    """Yield eligible PAGE-XML paths in deterministic order."""
    folder_name = folder_name.lower()
    for path in sorted(root.rglob("*.xml")):
        if not excluded(path.relative_to(root), prefixes) and folder_name in {p.lower() for p in path.parts}:
            yield path


def resolve_image(xml_path: Path, filename: str, root: Path) -> Path:
    """Resolve an image beside the XML, above its page folder, or uniquely below the input root."""
    supplied = Path(filename)
    candidates = (
        [supplied] if supplied.is_absolute() else [xml_path.parent / supplied, xml_path.parent.parent / supplied]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(root.rglob(supplied.name))
    return matches[0] if len(matches) == 1 else candidates[0]


def safe_stem(xml_path: Path, root: Path) -> str:
    """Build a readable, collision-resistant output stem for a page."""
    relative = xml_path.relative_to(root).with_suffix("").as_posix()
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", relative).strip("_")[-100:] or "page"
    return f"{readable}_{hashlib.sha1(relative.encode()).hexdigest()[:8]}"


def yolo_line(class_id: int, points: list[tuple[float, float]], width: int, height: int) -> str:
    """Serialize a polygon as a normalized YOLO segmentation label."""
    values = [value for x, y in points for value in (min(max(x / width, 0), 1), min(max(y / height, 0), 1))]
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in values)


def write_yaml(dataset_root: Path, names: dict[int, str]) -> None:
    """Write a YOLO dataset descriptor."""
    (dataset_root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"path": str(dataset_root.resolve()), "train": "images/train", "val": "images/val", "names": names},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def prepare_dataset(root: Path) -> None:
    """Create the standard YOLO image and label directories."""
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Convert PAGE-XML annotations into the two datasets."""
    args = parse_args()
    if not 0 <= args.val_ratio <= 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.crop_padding < 0:
        raise ValueError("--crop-padding cannot be negative")

    pages = []
    malformed = 0
    for xml_path in pagexml_files(args.input_root, args.page_folder_name, args.exclude_prefix):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            malformed += 1
            continue
        page = next((element for element in root.iter() if local_name(element.tag) == "Page"), None)
        if page is None:
            continue
        image_path = resolve_image(
            xml_path, page.attrib.get("imageFilename", xml_path.with_suffix(".jpg").name), args.input_root
        )
        if image_path.is_file():
            pages.append((xml_path, page, image_path))

    random.Random(args.seed).shuffle(pages)
    validation = {path for path, _, _ in pages[: int(len(pages) * args.val_ratio)]}
    region_names = sorted(
        {
            name
            for _, page, _ in pages
            for element in page.iter()
            if (name := region_class(local_name(element.tag))) is not None
            and element_points(element, args.min_points) is not None
        }
    )
    region_ids = {name: index for index, name in enumerate(region_names)}

    region_root, line_root = args.output_root / "regions", args.output_root / "lines"
    prepare_dataset(region_root)
    prepare_dataset(line_root)
    counts = Counter()

    for xml_path, page, image_path in pages:
        split = "val" if xml_path in validation else "train"
        stem = safe_stem(xml_path, args.input_root)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        region_labels = []

        for region_index, region in enumerate(
            element for element in page.iter() if region_class(local_name(element.tag))
        ):
            points = element_points(region, args.min_points)
            if points is None:
                continue
            class_name = region_class(local_name(region.tag))
            assert class_name is not None
            region_labels.append(yolo_line(region_ids[class_name], points, width, height))

            left = max(0, int(min(x for x, _ in points)) - args.crop_padding)
            top = max(0, int(min(y for _, y in points)) - args.crop_padding)
            right = min(width, int(max(x for x, _ in points) + 0.999) + args.crop_padding)
            bottom = min(height, int(max(y for _, y in points) + 0.999) + args.crop_padding)
            if right <= left or bottom <= top:
                continue
            line_labels = []
            for line in (element for element in region.iter() if local_name(element.tag) == "TextLine"):
                line_points = element_points(line, args.min_points)
                if line_points is not None:
                    local = [(x - left, y - top) for x, y in line_points]
                    line_labels.append(yolo_line(0, local, right - left, bottom - top))
            if not line_labels and not args.include_empty_crops:
                continue
            crop_stem = f"{stem}_region{region_index:04d}"
            image.crop((left, top, right, bottom)).save(line_root / "images" / split / f"{crop_stem}.jpg", quality=95)
            (line_root / "labels" / split / f"{crop_stem}.txt").write_text(
                "\n".join(line_labels) + ("\n" if line_labels else ""), encoding="utf-8"
            )
            counts[f"line_{split}"] += 1

        if not region_labels:
            continue
        image_destination = region_root / "images" / split / f"{stem}{image_path.suffix.lower()}"
        if args.copy_images:
            shutil.copy2(image_path, image_destination)
        else:
            image_destination.unlink(missing_ok=True)
            image_destination.symlink_to(image_path.resolve())
        (region_root / "labels" / split / f"{stem}.txt").write_text("\n".join(region_labels) + "\n", encoding="utf-8")
        counts[f"region_{split}"] += 1

    write_yaml(region_root, dict(enumerate(region_names)))
    write_yaml(line_root, {0: "TextLine"})
    print(f"Region dataset: {region_root} ({counts['region_train']} train, {counts['region_val']} val)")
    print(f"Line dataset: {line_root} ({counts['line_train']} train, {counts['line_val']} val crops)")
    if malformed:
        print(f"Skipped {malformed} malformed PAGE-XML file(s)")


if __name__ == "__main__":
    main()
