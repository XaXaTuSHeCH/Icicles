from ultralytics import YOLO
import yaml
import cv2
import os
import sys

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
        media_path = config.get("media_path")
        if not media_path or not os.path.exists(media_path):
            print("Ошибка: путь к изображению/видео не указан или не существует")
            return

        results = model(media_path)
        for result in results:
            annotated_frame = result.plot()
            output_path = os.path.join("runs/segment", "output_" + os.path.basename(media_path))
            cv2.imwrite(output_path, annotated_frame)
            print(f"Результат сохранён: {output_path}")


if __name__ == "__main__":
    run_inference()
