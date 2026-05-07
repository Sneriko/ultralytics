#!/usr/bin/env python3
"""Train a YOLO26 segmentation model directly from an existing YOLO dataset YAML."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO, settings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="Path to YOLO dataset YAML (e.g., dataset_seg.yaml)")

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
    p.add_argument("--mlflow-keep-run-active", action="store_true", help="Keep MLflow run open after training")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = args.data.resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    if args.mlflow:
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
        data=str(data_yaml),
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
