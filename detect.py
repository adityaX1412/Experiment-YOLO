import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/kaggle/input/ld70-waid-weight/best (3).pt') # select your model.pt path
    model.predict(source='/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # conf=0.2,
                  visualize=True # visualize model features maps
                )