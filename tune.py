from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Tune hyperparameters on data.yaml for 300 epochs and across 50 different hyperparameter spaces
model.tune(data='data.yaml', epochs=300, iterations=50, optimizer='AdamW', plots=False, save=False, val=False)