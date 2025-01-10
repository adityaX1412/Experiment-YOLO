import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/kaggle/input/70-yolospdn-soap/best (1).pt')
    model.val(data='/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml',
              split='test',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='yolov8m-ASF',
              )