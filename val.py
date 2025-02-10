import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/kaggle/input/bucktales-retdetr-weight/RETDETR.pt')
    # /kaggle/input/yolo-weights/weights/spdnsoap.pt
    model.val(data='/kaggle/input/bucktales-patched/bucktales_patched/dtc2023.yaml',
              split='test',
              imgsz=640,
              batch=16,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='yolov8n-spdnosoap',
              )