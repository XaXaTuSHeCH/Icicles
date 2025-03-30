# Детектор сосулек/наледи
## Тестовая модель 
### Для запуска:
1. Клонируйте репозиторий: 
```
git clone https://github.com/XaXaTuSHeCH/Icicles
```
2. Клонируйте SAM: 
```
git clone https://github.com/facebookresearch/segment-anything.git sam
```
3. Создайте виртуальное окружение: 
```
python3 -m venv venv
```
4. Активируйте виртуальное окружение:
```
source venv/bin/activate
```
5. Установите SAM:
```
pip install -e sam/
```
6. Установите зависимости:
``` 
pip install -r requirements.txt
```
7. Скачайте модель: 
```
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
8. Готово!