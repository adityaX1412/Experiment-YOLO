import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov8n_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/archive/dtc2023_local_unpatched.yaml',
              split='test',
              imgsz=1280,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val1',
              name='yolov8n-vanilla-unpatched-1280',
              plots=True,
              )
    
    print('\n2\n')


if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_SSFF+loss+p2+soap_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/archive/dtc2023_local_unpatched.yaml',
              split='test',
              imgsz=1280,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val1',
              name='SSFF+loss+p2+soap-unpatched-1280',
              plots=True,
              )
    
    print('\n3\n')

if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov8n_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/archive/dtc2023_local_unpatched.yaml',
              split='test',
              imgsz=2560,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val1',
              name='yolov8n-vanilla-unpatched-2560',
              plots=True,
              )
    
    print('\n4\n')


if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_SSFF+loss+p2+soap_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/archive/dtc2023_local_unpatched.yaml',
              split='test',
              imgsz=2560,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val1',
              name='SSFF+loss+p2+soap-unpatched-2560',
              plots=True,
              )
    
    print('\n5\n')
    
    
if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_yolov8n_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/archive/dtc2023_local_unpatched.yaml',
              split='test',
              imgsz=3840,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val1',
              name='yolov8n-vanilla-unpatched-3840',
              plots=True,
              )
    
    print('\n6\n')


if __name__ == '__main__':
    model = YOLO('C:/Users/hemgo/Desktop/AI/best_SSFF+loss+p2+soap_bucktales_patched.pt')
    model.val(data='C:/Users/hemgo/Desktop/AI/archive/dtc2023_local_unpatched.yaml',
              split='test',
              imgsz=3840,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrics
              project='runs/val1',
              name='SSFF+loss+p2+soap-unpatched-3840',
              plots=True,
              )
    
    print('\n7\n')