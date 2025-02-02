import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov9t_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/archive(1)_patched/dtc2023_local.yaml',
              split='test',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val',
              name='yolov9t-vanilla',
              plots=True,
              )