from ultralytics import YOLO

model = YOLO('runs/segment/test3/weights/best.pt')
results = model.val(data='data.yaml')
results = model.predict('Dataset_split/images/val/image3.jpg', save=True, conf=0.5)
