import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
import json
from collections import defaultdict
from thop import profile
import time

IMAGE_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
LABEL_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"
MODEL_WEIGHTS = "/kaggle/input/yolo-weights/weights/spdld.pt"
OUTPUT_DIR = "/kaggle/working/model_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5 
NMS_IOU_THRESHOLD = 0.45
DOUBLE_INFERENCE_THRESHOLD = 0.1
NUM_WARMUP = 10
NUM_TRIALS = 100

def calculate_inference_metrics(model, image_path):
    device = next(model.parameters()).device
    img = Image.open(image_path).convert("RGB")
    img = torch.from_numpy(np.array(img)).to(device).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)  # Add batch dimension
    
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = model(img)
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(NUM_TRIALS):
        with torch.no_grad():
            _ = model(img)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_inference_time = (end_time - start_time) / NUM_TRIALS
    macs, params = profile(model, inputs=(img,))
    gflops = macs * 2 / 1e9  
    
    return avg_inference_time, gflops, params

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (area1 + area2 - intersection + 1e-6)

def non_max_suppression(boxes, scores, labels, iou_threshold):
    if len(boxes) == 0:
        return [], [], []
    
    indices = np.argsort(scores)[::-1]
    boxes = np.array(boxes)
    keep = []

    while indices.size > 0:
        current = indices[0]
        keep.append(current)
        if indices.size == 1:
            break
        
        ious = np.array([calculate_iou(boxes[current], boxes[i]) for i in indices[1:]])
        indices = indices[1:][ious < iou_threshold]

    return boxes[keep].tolist(), [scores[i] for i in keep], [labels[i] for i in keep]

def calculate_map(predictions, targets, iou_threshold):
    class_aps = defaultdict(list)
    for preds, tgts in zip(predictions, targets):
        pred_boxes = np.array(preds['boxes'])
        pred_scores = np.array(preds['scores'])
        pred_labels = np.array(preds['labels'])
        true_boxes = np.array(tgts['boxes'])
        true_labels = np.array(tgts['labels'])
        
        unique_classes = np.unique(np.concatenate([pred_labels, true_labels]))
        
        for class_id in unique_classes:
            class_pred_mask = pred_labels == class_id
            class_true_mask = true_labels == class_id
            
            class_pred_boxes = pred_boxes[class_pred_mask]
            class_pred_scores = pred_scores[class_pred_mask]
            class_true_boxes = true_boxes[class_true_mask]
            
            if len(class_true_boxes) == 0:
                continue
            
            score_sort = np.argsort(-class_pred_scores)
            class_pred_boxes = class_pred_boxes[score_sort]
            class_pred_scores = class_pred_scores[score_sort]
            
            tp = np.zeros(len(class_pred_boxes))
            fp = np.zeros(len(class_pred_boxes))
            matched_gt = set()
            
            for pred_idx, pred_box in enumerate(class_pred_boxes):
                best_iou = iou_threshold
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(class_true_boxes):
                    if gt_idx in matched_gt:
                        continue
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx >= 0:
                    tp[pred_idx] = 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp[pred_idx] = 1
            
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / len(class_true_boxes)
            precisions = cum_tp / (cum_tp + cum_fp)
            
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11
            
            class_aps[int(class_id)].append(ap)
    
    mean_aps = {class_id: np.mean(aps) for class_id, aps in class_aps.items()}
    map_value = np.mean(list(mean_aps.values()))
    
    return map_value, mean_aps

model = YOLO(MODEL_WEIGHTS)
image_files = os.listdir(IMAGE_DIR)

sample_size = min(10, len(image_files))
total_inference_time, total_gflops = 0, 0

for image_path in image_files[:sample_size]:
    full_image_path = os.path.join(IMAGE_DIR, image_path)
    inference_time, gflops, _ = calculate_inference_metrics(model, full_image_path)
    total_inference_time += inference_time
    total_gflops += gflops

avg_inference_time = total_inference_time / sample_size
avg_gflops = total_gflops / sample_size

print(f"Average Inference Time: {avg_inference_time*1000:.2f} ms")
print(f"Average GFLOPS: {avg_gflops:.2f}")
