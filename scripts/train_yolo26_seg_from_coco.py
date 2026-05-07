#!/usr/bin/env python3
"""Convert COCO JSON annotations to YOLO-seg format and train a YOLO26-seg model.

This helper script:
1) Converts COCO JSON files from annotations/ into YOLO TXT segmentation labels.
2) Ensures labels are copied to dataset_root/labels/{train,val,test}.
3) Builds a dataset YAML from COCO categories.
4) Launches Ultralytics YOLO segment training with explicit (default) tunable parameters.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from ultralytics import YOLO, settings
from ultralytics.data.converter import convert_coco


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True, help="Dataset root for generated labels/yaml and optional default paths")
    p.add_argument(
        "--annotations-dir",
        type=Path,
        default=None,
        help="Directory containing COCO JSON files (default: <dataset-root>/annotations)",
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing train/ val/ test image folders (default: <dataset-root>/images)",
    )
    p.add_argument(
        "--train-json",
        type=str,
        default="instances_train.json",
        help="Train JSON filename located in annotations-dir",
    )
    p.add_argument(
        "--val-json",
        type=str,
        default="instances_val.json",
        help="Val JSON filename located in annotations-dir",
    )
    p.add_argument("--test-json", type=str, default="", help="Optional test JSON filename in annotations-dir")
    p.add_argument("--yaml-out", type=Path, default=None, help="Output dataset yaml path (default: <dataset-root>/dataset_seg.yaml)")
    p.add_argument(
        "--convert-save-dir",
        type=Path,
        default=None,
        help="Where convert_coco writes intermediate output (default: <dataset-root>/converted)",
    )

    # Model and train params (set to Ultralytics defaults)
    p.add_argument("--model", type=str, default="yolo26n-seg.pt", help="Model checkpoint or model YAML")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default=None, help="cuda device id(s), 'cpu', or None for auto")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--optimizer", type=str, default="auto")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--hsv-h", type=float, default=0.015)
    p.add_argument("--hsv-s", type=float, default=0.7)
    p.add_argument("--hsv-v", type=float, default=0.4)
    p.add_argument("--degrees", type=float, default=0.0)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--shear", type=float, default=0.0)
    p.add_argument("--perspective", type=float, default=0.0)
    p.add_argument("--flipud", type=float, default=0.0)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--copy-paste", type=float, default=0.0)

    p.add_argument("--project", type=str, default="runs/segment")
    p.add_argument("--name", type=str, default="train")
    p.add_argument("--exist-ok", action="store_true")

    # MLflow tracking controls
    p.add_argument("--mlflow", action="store_true", help="Enable MLflow logging for this run")
    p.add_argument("--mlflow-tracking-uri", type=str, default="", help="Remote MLflow tracking URI")
    p.add_argument("--mlflow-experiment-name", type=str, default="", help="MLflow experiment name")
    p.add_argument("--mlflow-run-name", type=str, default="", help="MLflow run name override")
    p.add_argument("--mlflow-username", type=str, default="", help="MLflow basic-auth username")
    p.add_argument("--mlflow-password", type=str, default="", help="MLflow basic-auth password")
    p.add_argument("--mlflow-token", type=str, default="", help="MLflow bearer token")
    p.add_argument(
        "--mlflow-keep-run-active",
        action="store_true",
        help="Keep MLflow run open after training (otherwise run is ended automatically)",
    )
    return p.parse_args()


def load_categories(coco_json: Path) -> dict[int, str]:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    categories = sorted(data["categories"], key=lambda x: x["id"])
    return {i: c["name"] for i, c in enumerate(categories)}


def ensure_split_labels(converted_labels_dir: Path, dataset_root: Path, split: str) -> None:
    src = converted_labels_dir / split
    dst = dataset_root / "labels" / split
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for label_file in src.glob("*.txt"):
        shutil.copy2(label_file, dst / label_file.name)


def write_dataset_yaml(yaml_path: Path, images_dir: Path, names: dict[int, str], has_test: bool) -> None:
    images_dir = images_dir.resolve()
    payload = {
        "train": str(images_dir / "train"),
        "val": str(images_dir / "val"),
        "names": names,
    }
    if has_test:
        payload["test"] = str(images_dir / "test")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    annotations_dir = (args.annotations_dir or (dataset_root / "annotations")).resolve()
    images_dir = (args.images_dir or (dataset_root / "images")).resolve()
    yaml_out = args.yaml_out or (dataset_root / "dataset_seg.yaml")
    convert_save_dir = (args.convert_save_dir or (dataset_root / "converted")).resolve()

    train_json_path = annotations_dir / args.train_json
    val_json_path = annotations_dir / args.val_json
    test_json_path = annotations_dir / args.test_json if args.test_json else None

    if not train_json_path.exists() or not val_json_path.exists():
        raise FileNotFoundError(f"Expected train/val JSON at {train_json_path} and {val_json_path}")

    convert_coco(
        labels_dir=str(annotations_dir),
        save_dir=str(convert_save_dir),
        use_segments=True,
        cls91to80=False,
    )

    converted_labels = convert_save_dir / "labels"
    ensure_split_labels(converted_labels, dataset_root, "train")
    ensure_split_labels(converted_labels, dataset_root, "val")
    if test_json_path and test_json_path.exists():
        ensure_split_labels(converted_labels, dataset_root, "test")

    names = load_categories(train_json_path)
    write_dataset_yaml(yaml_out, images_dir, names, has_test=bool(test_json_path and test_json_path.exists()))

    if args.mlflow:
        import os

        settings.update({"mlflow": True})
        if args.mlflow_tracking_uri:
            os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
        if args.mlflow_experiment_name:
            os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment_name
        if args.mlflow_run_name:
            os.environ["MLFLOW_RUN"] = args.mlflow_run_name
        if args.mlflow_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = args.mlflow_username
        if args.mlflow_password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = args.mlflow_password
        if args.mlflow_token:
            os.environ["MLFLOW_TRACKING_TOKEN"] = args.mlflow_token
        if args.mlflow_keep_run_active:
            os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "True"

    model = YOLO(args.model)
    model.train(
        data=str(yaml_out),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
