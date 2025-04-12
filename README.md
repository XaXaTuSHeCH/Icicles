# :snowflake: Icicle and Ice Detection with YOLOv8

![image](https://github.com/user-attachments/assets/87d0314d-14fe-4c5c-98c5-10daba3ef119)


This project implements a computer vision model for automatic detection of icicles and ice formations using YOLOv8 segmentation version.

## :wrench: Features

- YOLOv8-seg model training for icicle/ice segmentation
- Model validation on test data
- Inference on new images
- Detection results visualization

## :inbox_tray: Installation

1. Clone the repository:
```bash
git clone https://github.com/XaXaTuSHeCH/Icicles
```

2. Clone the SAM
```bash
git clone https://github.com/facebookresearch/segment-anything.git sam
```

3. Create virtual environment
```bash
python3 -m venv venv
```

4. Activate the virtual environment:
```bash
source venv/bin/activate
```

5. Download the SAM:
```bash
pip install -e sam/
```

6. Install all dependencies:
```bash
pip install -r requirements.txt
```

7. Download the model:
```bash
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth 
```

## :computer: Usage

#### Train the model
```bash
python app.py
```
Training configuration can be modified in app.py (train_config dictionary).

#### Test the model
```bash
python app_test.py
```
Before running, make sure to specify correct paths to:
- Trained model weights (weights_path)
- Data config file (data_yaml_path)
- Test image (test_image_path)

## :chart_with_upwards_trend: Results
After training, the model is saved in runs/segment/ including:
- Best weights (best.pt)
- Training metrics plots
- Validation set prediction examples

## :construction: License
This project is licensed under the MIT License - see the LICENSE file for details.

## :coffee: Other collaborators
- :bear: https://github.com/MikeV182
- :bug: https://github.com/KostKP
