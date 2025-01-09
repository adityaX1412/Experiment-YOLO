import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    model = YOLO('/kaggle/input/nano-softshare-weights/yolov8_softshare_waid.pt')
    checkpoint_path = '/kaggle/input/nano-softshare-weights/yolov8_softshare_waid.pt'
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    print(ckpt.keys())
    model.val(data='/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml',
              split='test',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='yolov8m-ASF',
              )
    