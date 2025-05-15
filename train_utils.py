import os

def validate_dataset(dataset_path):
    """
    Проверяет, содержит ли папка датасета необходимые файлы и структуру.
    Ожидается: data.yaml и папки train/, val/ с изображениями и метками.
    """
    data_yaml = os.path.join(dataset_path, "data.yaml")
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")

    if not os.path.exists(data_yaml):
        return False, "data.yaml не найден"
    if not os.path.exists(train_dir):
        return False, "Папка train/ не найдена"
    if not os.path.exists(val_dir):
        return False, "Папка val/ не найдена"

    return True, "Датасет валиден"