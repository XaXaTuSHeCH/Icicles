import tkinter as tk
from tkinter import filedialog, ttk
import subprocess
import yaml
import cv2
from camera_utils import get_available_cameras


def get_camera_list():
    cameras = get_available_cameras()
    return [f"Камера {i}" for i in cameras]


class IcicleDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Детектор сосулек")
        self.root.geometry("500x400")

        # Загрузка конфигурации
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        # Элементы интерфейса
        self.create_widgets()

    def create_widgets(self):
        # Кнопка для загрузки датасета
        btn_load_dataset = tk.Button(self.root, text="Загрузить датасет", command=self.load_dataset)
        btn_load_dataset.pack(pady=10)

        # Кнопка для загрузки изображения/видео
        btn_load_media = tk.Button(self.root, text="Загрузить изображение/видео", command=self.load_media)
        btn_load_media.pack(pady=10)

        # Выбор камеры
        tk.Label(self.root, text="Выберите камеру:").pack(pady=5)
        self.camera_combobox = ttk.Combobox(self.root, values=get_camera_list())
        self.camera_combobox.pack(pady=5)
        btn_start_camera = tk.Button(self.root, text="Запустить детекцию с камеры", command=self.start_camera)
        btn_start_camera.pack(pady=10)

        # Метка для статуса
        self.status_label = tk.Label(self.root, text="Ожидание действия...")
        self.status_label.pack(pady=20)

    def load_dataset(self):
        dataset_path = filedialog.askdirectory(title="Выберите папку с датасетом")
        if dataset_path:
            self.config["dataset_path"] = dataset_path
            with open("config.yaml", "w") as f:
                yaml.safe_dump(self.config, f)
            subprocess.run(["python", "app.py"])
            self.status_label.config(text="Дообучение завершено!")

    def load_media(self):
        media_path = filedialog.askopenfilename(
            title="Выберите изображение или видео",
            filetypes=[("Media files", "*.jpg *.png *.mp4")]
        )
        if media_path:
            self.config["media_path"] = media_path
            with open("config.yaml", "w") as f:
                yaml.safe_dump(self.config, f)
            subprocess.run(["python", "app_test.py"])
            self.status_label.config(text="Инференс выполнен!")

    def start_camera(self):
        camera_index = self.camera_combobox.current()
        if camera_index >= 0:
            self.config["camera_index"] = camera_index
            with open("config.yaml", "w") as f:
                yaml.safe_dump(self.config, f)
            subprocess.run(["python", "app_test.py", "--camera"])
            self.status_label.config(text="Детекция с камеры завершена!")
        else:
            self.status_label.config(text="Выберите камеру!")


if __name__ == "__main__":
    root = tk.Tk()
    app = IcicleDetectorApp(root)
    root.mainloop()
