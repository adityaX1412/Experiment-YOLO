import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/kaggle/input/bucktales-vanilla-with-soap/vanilla soap.pt')
    # /kaggle/input/yolo-weights/weights/spdnsoap.pt
    model.val(data='/kaggle/input/bucktales-patched/dtc2023.yaml',
              split='test',
              imgsz=1280,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='yolov8n-spdld',
              )