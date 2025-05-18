import os
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import yaml
from PIL import Image, ImageTk
from ultralytics import YOLO

from camera_utils import get_available_cameras


class IcicleDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Детектор сосулек")
        self.root.geometry("600x400")
        self.root.configure(bg="#333333")
        self.progress_queue = queue.Queue()
        self.processed_path = None
        self.model = YOLO("best.pt")
        self.video_window = None

        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        self.create_widgets()

    def create_widgets(self):
        button_frame = tk.Frame(self.root, bg="#333333")
        button_frame.pack(pady=10)

        self.btn_retrain = tk.Button(
            button_frame,
            text="Провести дообучение",
            command=self.retrain_model,
            bg="#444444",
            fg="#00008b",
            font=("Arial", 12),
            width=20,
        )
        self.btn_retrain.pack(side=tk.LEFT, padx=5)

        self.btn_load_media = tk.Button(
            button_frame,
            text="Загрузить изображение/видео",
            command=self.load_media,
            bg="#444444",
            fg="#00008b",
            font=("Arial", 12),
            width=20,
        )
        self.btn_load_media.pack(side=tk.LEFT, padx=5)

        self.btn_start_camera = tk.Button(
            button_frame,
            text="Запустить детекцию с камеры",
            command=self.start_camera,
            bg="#444444",
            fg="#00008b",
            font=("Arial", 12),
            width=20,
        )
        self.btn_start_camera.pack(side=tk.LEFT, padx=5)

        camera_frame = tk.Frame(self.root, bg="#444444")
        camera_frame.pack(pady=10)
        self.camera_label = tk.Label(
            camera_frame,
            text="Выберите камеру:",
            bg="#444444",
            fg="#00008b",
            font=("Arial", 12),
        )
        self.camera_label.pack(side=tk.LEFT, padx=5)
        self.camera_var = tk.StringVar(value="Выберите камеру")
        self.camera_combobox = tk.OptionMenu(
            camera_frame, self.camera_var, *self.get_camera_list()
        )
        self.camera_combobox.config(
            bg="#444444", fg="#00008b", font=("Arial", 10), width=12
        )
        self.camera_combobox["menu"].config(bg="#444444", fg="#00008b")
        self.camera_combobox.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(
            self.root,
            text="Ожидание действия...",
            bg="#444444",
            fg="#00008b",
            font=("Arial", 12),
        )
        self.status_label.pack(pady=15)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = tk.Canvas(self.root, height=20, width=300, bg="#444444")
        self.progress_rect = self.progress_bar.create_rectangle(
            0, 0, 0, 20, fill="#00ff00"
        )

    @staticmethod
    def get_camera_list():
        cameras = get_available_cameras()
        return [f"Камера {i}" for i in cameras]

    def load_media(self):
        media_path = filedialog.askopenfilename(
            title="Выберите изображение или видео",
            filetypes=[("Media files", "*.jpg *.png *.jpeg *.mp4 *.avi")],
        )
        if media_path:
            self.status_label.config(text="Обработка...")
            processing_label = tk.Label(
                self.root,
                text="Обработка...",
                bg="#444444",
                fg="#00008b",
                font=("Arial", 12),
            )
            processing_label.pack()
            thread = threading.Thread(
                target=self.process_media_thread, args=(media_path, processing_label)
            )
            thread.start()
            self.check_media_thread(thread, processing_label)

    def process_media_thread(self, media_path):
        result = subprocess.run(
            ["python", "app_test.py", media_path], capture_output=True, text=True
        )
        self.processed_path = result.stdout.strip()

    def check_media_thread(self, thread, processing_label):
        if thread.is_alive():
            self.root.after(
                100, lambda: self.check_media_thread(thread, processing_label)
            )
        else:
            processing_label.destroy()
            self.status_label.config(text="Обработка завершена")
            if self.processed_path and os.path.exists(self.processed_path):
                self.show_result_window(self.processed_path)
            else:
                messagebox.showerror("Ошибка", "Не удалось обработать файл")

    def show_result_window(self, output_path):
        result_window = tk.Toplevel(self.root)
        result_window.title("Результат")
        result_window.geometry("600x400")
        result_window.configure(bg="#444444")

        if output_path.lower().endswith((".jpg", ".png", ".jpeg")):
            img = Image.open(output_path)
            img = img.resize((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(result_window, image=photo, bg="#444444")
            label.image = photo
            label.pack(pady=15)
        elif output_path.lower().endswith((".mp4", ".avi")):
            cv2.namedWindow("Детекция сосулек", cv2.WINDOW_NORMAL)
            cap = cv2.VideoCapture(output_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow("Детекция сосулек", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()
            self.video_window = None

        btn_frame = tk.Frame(result_window, bg="#444444")
        btn_frame.pack(pady=15)
        btn_save = tk.Button(
            btn_frame,
            text="Сохранить",
            command=lambda: self.save_result(output_path),
            bg="#444444",
            fg="#00008b",
            font=("Arial", 12),
            width=12,
        )
        btn_save.pack(side=tk.LEFT, padx=10)
        btn_done = tk.Button(
            btn_frame,
            text="Готово",
            command=lambda: self.close_result_window(result_window),
            bg="#444444",
            fg="#00008b",
            font=("Arial", 12),
            width=12,
        )
        btn_done.pack(side=tk.LEFT, padx=10)

    def close_result_window(self, window):
        if self.video_window:
            cv2.destroyAllWindows()
            self.video_window = None
        window.destroy()

    @staticmethod
    def save_result(path):
        ext = ".jpg" if path.lower().endswith((".jpg", ".png", ".jpeg")) else ".mp4"
        save_path = filedialog.asksaveasfilename(defaultextension=ext)
        if save_path:
            import shutil

            shutil.copy(path, save_path)
            messagebox.showinfo("Сохранено", f"Файл сохранён в {save_path}")

    def retrain_model(self):
        dataset_path = filedialog.askdirectory(title="Выберите папку с датасетом")
        if dataset_path:
            self.config["dataset_path"] = dataset_path
            with open("config.yaml", "w") as f:
                yaml.safe_dump(self.config, f)
            self.status_label.config(text="Дообучение...")
            self.progress_bar.pack(pady=15)
            thread = threading.Thread(target=self.train_model)
            thread.start()
            self.check_progress()

    def train_model(self):
        model = YOLO("yolov8n-seg.pt")

        def on_train_epoch_end(trainer):
            progress = (trainer.epoch + 1) / trainer.epochs * 100
            self.progress_queue.put(progress)

        def on_train_end():
            self.progress_queue.put("done")

        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_train_end", on_train_end)
        model.train(
            data=os.path.join(self.config["dataset_path"], "data.yaml"), epochs=100
        )

    def check_progress(self):
        try:
            item = self.progress_queue.get_nowait()
            if item == "done":
                self.progress_bar.delete("all")
                self.progress_bar.pack_forget()
                self.status_label.config(text="Дообучение завершено")
                messagebox.showinfo("Завершено", "Дообучение завершено.")
            else:
                width = (item / 100) * 300
                self.progress_bar.coords(self.progress_rect, 0, 0, width, 20)
                self.root.after(100, self.check_progress)
        except queue.Empty:
            self.root.after(100, self.check_progress)

    def start_camera(self):
        camera_index = self.camera_var.get().split()[-1]
        if camera_index.isdigit():
            self.config["camera_index"] = int(camera_index)
            with open("config.yaml", "w") as f:
                yaml.safe_dump(self.config, f)
            subprocess.run(["python", "app_test.py", "--camera"])
            self.status_label.config(text="Детекция с камеры завершена")
        else:
            self.status_label.config(text="Выберите камеру!")


if __name__ == "__main__":
    root = tk.Tk()
    app = IcicleDetectorApp(root)
    root.mainloop()
