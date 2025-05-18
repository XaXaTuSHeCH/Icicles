import os
import platform
import shutil
import subprocess
import sys
import venv

import requests


def check_python_version():
    if sys.version_info < (3, 8):
        print("Ошибка: требуется Python 3.8 или новее.")
        sys.exit(1)


def create_virtualenv():
    venv_path = "venv"
    if os.path.exists(venv_path):
        print("Виртуальное окружение уже существует. Пропускаю создание.")
    else:
        print("Создание виртуального окружения...")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(venv_path)
    return venv_path


def install_dependencies(venv_path):
    print("Установка зависимостей...")
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        pip_path = os.path.join(venv_path, "bin", "pip")
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)


def clone_or_update_repo():
    if not os.path.exists(".git"):
        repo_url = "https://github.com/XaXaTuSHeCH/Icicles.git"
        print(f"Клонирование репозитория из {repo_url}...")
        subprocess.run(["git", "clone", repo_url, "."], check=True)
    else:
        print("Обновление репозитория...")
        subprocess.run(["git", "pull"], check=True)


def create_config():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print("Создание конфигурационного файла...")
        default_config = {
            "weights_path": "best.pt",
            "dataset_path": "",
            "camera_index": 0,
        }
        with open(config_path, "w") as f:
            import yaml

            yaml.safe_dump(default_config, f)
        print("Конфигурационный файл создан.")
    else:
        print("Конфигурационный файл уже существует.")


if __name__ == "__main__":
    check_python_version()
    venv_path = create_virtualenv()
    install_dependencies(venv_path)
    clone_or_update_repo()
    download_model_weights()
    create_config()
    print(
        "Настройка завершена! Запустите run.exe (Windows), run (macOS/Linux) для запуска программы."
    )
