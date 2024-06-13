from ultralytics import YOLO

model = YOLO("iolov8m-football.pt")
model.predict("test.png", save=True, imgsz=640, conf=0.5)