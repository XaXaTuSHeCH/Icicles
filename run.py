import os
import platform
import subprocess
import sys


def activate_venv():
    venv_path = "venv"
    if not os.path.exists(venv_path):
        print(
            "Ошибка: виртуальное окружение не найдено. Выполните сначала setup.exe (Windows) или setup (macOS/Linux)."
        )
        sys.exit(1)

    if platform.system() == "Windows":
        activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
        subprocess.run([activate_script], shell=True)
    else:
        activate_script = os.path.join(venv_path, "bin", "activate")
        subprocess.run(["source", activate_script], shell=True)


def run_gui():
    print("Запуск программы...")
    if platform.system() == "Windows":
        subprocess.run(["python", "gui.py"])
    else:
        subprocess.run(["python3", "gui.py"])


if __name__ == "__main__":
    activate_venv()
    run_gui()
