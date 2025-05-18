import os
import sys

import cv2
import yaml
from ultralytics import YOLO


def process_media(media_path, model):
    output_dir = "runs/segment"
    os.makedirs(output_dir, exist_ok=True)

    if media_path.lower().endswith((".jpg", ".png", ".jpeg")):
        results = model(media_path)
        annotated_image = results[0].plot()
        output_path = os.path.join(output_dir, "output_" + os.path.basename(media_path))
        cv2.imwrite(output_path, annotated_image)
        return output_path
    elif media_path.lower().endswith((".mp4", ".avi")):
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            print("Ошибка открытия видео")
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_path = os.path.join(output_dir, "output_" + os.path.basename(media_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
        cap.release()
        out.release()
        return output_path
    else:
        print("Неподдерживаемый тип файла")
        return None


def run_inference():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    weights_path = config.get("weights_path", "best.pt")
    if not os.path.exists(weights_path):
        print(f"Ошибка: веса модели не найдены по пути {weights_path}")
        return

    model = YOLO(weights_path)

    if "--camera" in sys.argv:
        camera_index = config.get("camera_index", 0)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть камеру {camera_index}")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow("Детекция сосулек", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        if len(sys.argv) < 2:
            print("Укажите путь к файлу: python app_test.py <media_path>")
            sys.exit(1)
        media_path = sys.argv[1]
        output_path = process_media(media_path, model)
        if output_path:
            print(output_path)


if __name__ == "__main__":
    run_inference()
