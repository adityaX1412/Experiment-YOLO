import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/kaggle/input/yolo-weights/weights/spdn  soap.pt')
    # /kaggle/input/yolo-weights/weights/spdnsoap.pt
    model.val(data='/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml',
              split='test',
              imgsz=640,
              batch=16,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='yolov8n-spdld',
              )