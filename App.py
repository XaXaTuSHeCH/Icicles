from ultralytics import YOLO

model = YOLO("./weights/yolov8x-seg.pt")

model.train(
    data="data.yaml",
    epochs=3,
    imgsz=640,
    optimize=False,
    batch=4,
    project="runs/segment",
    name="test",
    device="mps",
    optimizer="adam",
)
