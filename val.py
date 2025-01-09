import torch
from ultralytics import YOLO

# Define a function to load weights into a YOLO model
def load_custom_weights(checkpoint_path, architecture_path='yolov8.yaml'):
    # Load weights from the checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    weights = {k: v for k, v in ckpt.items() if k.startswith('model')}

    # Initialize YOLO model with a specified architecture
    model = YOLO(architecture_path)
    model.model.load_state_dict(weights, strict=False)  # Load weights
    return model

checkpoint_path = '/kaggle/input/nano-softshare-weights/yolov8_softshare_waid.pt'
architecture_path = '/path/to/yolov8.yaml'  # Replace with the correct path to the YOLOv8 model architecture file

try:
    model = load_custom_weights(checkpoint_path, architecture_path)
    model.val(
        data='/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml',
        split='test',
        imgsz=640,
        batch=16,
        project='runs/val',
        name='yolov8m-ASF',
    )
except Exception as e:
    print(f"Error: {e}")

