import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from ultralytics import YOLO


def load_model(weights_path):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Модель не найдена: {weights_path}")
    model = YOLO(weights_path)
    print(f"Модель успешно загружена из {weights_path}")
    return model


def validate_model(model, data_config):
    print("Запуск валидации...")
    results = model.val(data=data_config)
    print("Результаты валидации:")
    print(results)
    return results


def run_inference(model, image_path, save_results=True, conf_threshold=0.5):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    print(f"Выполняется инференс на изображении: {image_path}")
    results = model.predict(
        source=image_path,
        save=save_results,
        conf=conf_threshold,
        show=False,
        verbose=True,
    )

    return results


def visualize_prediction(image_path, results):
    img = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Исходное изображение")

    if results and hasattr(results[0], "plot"):
        result_img = results[0].plot()
        plt.figure(figsize=(8, 8))
        plt.imshow(result_img)
        plt.axis("off")
        plt.title("Предсказание модели")
    else:
        print("Предсказание не содержит визуализируемых данных.")

    plt.show()


if __name__ == "__main__":

    weights_path = "runs/segment/test3/weights/best.pt"
    data_yaml_path = "data.yaml"
    test_image_path = "Dataset_split/images/val/image3.jpg"

    model = load_model(weights_path)

    validation_results = validate_model(model, data_yaml_path)

    inference_results = run_inference(
        model, test_image_path, save_results=True, conf_threshold=0.5
    )

    visualize_prediction(test_image_path, inference_results)
