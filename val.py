import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics import YOLO

# Debugging checkpoint
checkpoint_path = '/kaggle/input/nano-softshare-weights/yolov8_softshare_waid.pt'
try:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    print("Checkpoint keys:", ckpt.keys())
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    raise

if __name__ == '__main__':
    try:
        model = YOLO(checkpoint_path)
        model.val(
            data='/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml',
            split='test',
            imgsz=640,
            batch=16,
            project='runs/val',
            name='yolov8m-ASF',
        )
    except KeyError as e:
        print(f"Model loading error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
