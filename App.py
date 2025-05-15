from ultralytics import YOLO
import yaml
import os

def train_model():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset_path = config.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        print("Ошибка: путь к датасету не указан или не существует")
        return

    model = YOLO("weights/yolov8x-seg.pt")

    train_config = {
        "data": os.path.join(dataset_path, "data.yaml"),
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "device": 0,
        "project": "runs/segment",
        "name": "train",
        "exist_ok": True
    }

    model.train(**train_config)

    best_weights_path = os.path.join(train_config["project"], train_config["name"], "weights", "best.pt")
    if os.path.exists(best_weights_path):
        config["weights_path"] = best_weights_path
        with open("config.yaml", "w") as f:
            yaml.safe_dump(config, f)
        print(f"Обучение завершено. Лучшие веса сохранены в: {best_weights_path}")


if __name__ == "__main__":
    train_model()
