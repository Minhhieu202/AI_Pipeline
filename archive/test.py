from ultralytics import YOLO

model = YOLO("/kaggle/working/runs/detect/train2/weights/best.pt")
result = model("/kaggle/input/framev3/frame/frame299.jpg",show=True)
result[0].show()