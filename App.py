import os

import torch
from ultralytics import YOLO


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


device = get_device()
print(f"[INFO] Using device: {device}")

weights_path = "./weights/yolov8x-seg.pt"
data_yaml_path = "data.yaml"
output_project = "runs/segment"
run_name = "test"

train_config = {
    "epochs": 3,
    "imgsz": 640,
    "batch": 8,
    "optimizer": "adam",
    "device": device,
    "project": output_project,
    "name": run_name,
    "verbose": True,
    "patience": 50,
    "save": True,
    "save_period": -1,
    "val": True,
    "workers": 4,
    "cos_lr": False,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "dropout": 0.0,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "auto_augment": None,
    "rect": False,
    "resume": False,
    "nosave": False,
    "noval": False,
    "noautoanchor": False,
    "sync_bn": False,
    "workers": 8,
}

print("[INFO] Loading YOLOv8 model...")
model = YOLO(weights_path)

print("[INFO] Starting training...")
model.train(data=data_yaml_path, **train_config)

print("[INFO] Training completed.")
