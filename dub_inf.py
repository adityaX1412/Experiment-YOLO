import os
import sys
import time
import glob
import numpy as np
import torch
from PIL import Image
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
import wandb
import numpy as np
import yaml
from torch.autograd import Variable
from ultralytics.utils.loss import v8DetectionLoss
from torchmetrics.detection import MeanAveragePrecision

#counting the next 3
total_predictions = 0
correct_predictions = 0
iou_threshold = 0.5

def simple_nms(boxes, scores, iou_threshold=0):
    # Convert to tensor if needed
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    
    # Sort by descending confidence
    sorted_indices = torch.argsort(scores, descending=True)
    keep = []
    
    while sorted_indices.numel() > 0:
        # Keep the highest confidence box
        current_idx = sorted_indices[0]
        keep.append(current_idx.item())
        
        if sorted_indices.numel() == 1:
            break
            
        # Compute IoU with remaining boxes
        current_box = boxes[current_idx].unsqueeze(0)
        remaining_boxes = boxes[sorted_indices[1:]]
        
        # Calculate intersection
        x1 = torch.max(current_box[:, 0], remaining_boxes[:, 0])
        y1 = torch.max(current_box[:, 1], remaining_boxes[:, 1])
        x2 = torch.min(current_box[:, 2], remaining_boxes[:, 2])
        y2 = torch.min(current_box[:, 3], remaining_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union
        area_current = (current_box[:, 2] - current_box[:, 0]) * (current_box[:, 3] - current_box[:, 1])
        area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = area_current + area_remaining - intersection
        
        iou = intersection / union
        
        # Remove boxes with IoU > threshold
        mask = iou <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return keep

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    
    return intersection / (area1 + area2 - intersection + 1e-6)

def scale_boxes(padded_boxes, pad_x, pad_y, resize_ratio_x, resize_ratio_y, crop_coords):
    """Always returns a numpy array with proper dimensions"""
    try:
        # Handle null/empty inputs
        if padded_boxes is None or not isinstance(padded_boxes, np.ndarray):
            return np.empty((0, 4))
        
        # Ensure 2D array format
        if padded_boxes.size == 0:
            return np.empty((0, 4))
        if padded_boxes.ndim == 1:
            padded_boxes = np.expand_dims(padded_boxes, 0)
            
        # Perform coordinate transformations
        boxes = padded_boxes.copy()
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] - pad_x, 0, crop_coords['resized_w'])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] - pad_y, 0, crop_coords['resized_h'])
        
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * resize_ratio_x + crop_coords['x1']
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * resize_ratio_y + crop_coords['y1']
        
        return boxes
    except Exception as e:
        print(f"Scaling error: {str(e)}")
        return np.empty((0, 4))

image_dir = '/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test'
label_dir = '/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test'

model = YOLO('yolov8-LD-P2.yaml')
state_dict = torch.load("/kaggle/input/yolo-weights/weights/spdld.pt",weights_only=True)
model.model.load_state_dict(state_dict, strict=False)
conf_threshold = 0.7
metric = MeanAveragePrecision(class_metrics=True)
counta = 0

for image_path in os.listdir(image_dir):
    # Load image and initial prediction
    img = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
    img_width, img_height = img.size
    initial_results = model.predict(img, conf=0.1)
    result = initial_results[0]
    
    # Load ground truth
    true_boxes, true_labels = [], []
    label_path = os.path.join(label_dir, os.path.splitext(image_path)[0] + '.txt')
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x1 = (x_center - width/2) * img_width
                y1 = (y_center - height/2) * img_height
                x2 = (x_center + width/2) * img_width
                y2 = (y_center + height/2) * img_height
                true_boxes.append([x1, y1, x2, y2])
                true_labels.append(int(class_id))

    # Initial predictions processing
    predictions = {
        'boxes': [],
        'scores': [],
        'labels': []
    }
    # print(f'true boxes: {true_boxes}')
    # print(f'true labels: {true_labels}')
    for box in result.boxes:
        predictions['boxes'].append(box.xyxy[0].cpu().numpy().tolist())
        predictions['scores'].append(box.conf.item())
        predictions['labels'].append(int(box.cls.item()))
    # print(f'predictions: {predictions}')
    # Find reference box (highest confidence)
    ref_idx = np.argmax(predictions['scores']) if predictions['scores'] else -1
    if ref_idx != -1:
        ref_box = predictions['boxes'][ref_idx]
        rw = ref_box[2] - ref_box[0]
        rh = ref_box[3] - ref_box[1]
    else:
        rw, rh = 0, 0

    # Refinement pass
    replacement_candidates = []
    for i in range(len(predictions['scores'])):
        if predictions['scores'][i] >= conf_threshold or rw == 0 or rh == 0:
            continue
        
        best_match = None
        best_iou = -1
        best_conf = -1
        pre_box = predictions['boxes'][i]
        original_label = predictions['labels'][i]
        original_score = predictions['scores'][i]
        x1, y1, x2, y2 = predictions['boxes'][i]
        sw, sh = x2 - x1, y2 - y1
        
        # Calculate adaptive crop size
        desired_width = (sw * 640) / rw if rw != 0 else 640
        desired_height = (sh * 640) / rh if rh != 0 else 640
        cx, cy = (x1 + x2)/2, (y1 + y2)/2
        
        # Expand ROI with boundary checks
        new_x1 = max(0, int(cx - desired_width/2))
        new_y1 = max(0, int(cy - desired_height/2))
        new_x2 = min(img_width, int(cx + desired_width/2))
        new_y2 = min(img_height, int(cy + desired_height/2))
        
        if (new_x2 <= new_x1) or (new_y2 <= new_y1):
            continue

        # Aspect ratio-preserving resize
        crop = img.crop((new_x1, new_y1, new_x2, new_y2))
        original_w, original_h = crop.size
        ratio = min(640/original_w, 640/original_h)
        new_size = (int(original_w*ratio), int(original_h*ratio))
        resized = crop.resize(new_size, Image.BILINEAR)
        
        # Pad to 640x640
        padded_img = Image.new("RGB", (640, 640), (114, 114, 114))
        pad_x, pad_y = (640 - new_size[0])//2, (640 - new_size[1])//2
        padded_img.paste(resized, (pad_x, pad_y))

        # Second pass inference
        with torch.no_grad():
            new_results = model.predict(padded_img, conf=0.1)
        
        if len(new_results[0].boxes) == 0:
            continue

        # Process detections with dimension checks
        boxes = new_results[0].boxes.xyxy.cpu().numpy()
        # Ensure 2D array even for single detection
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
            
        confs = new_results[0].boxes.conf.cpu().numpy()
        labels = new_results[0].boxes.cls.cpu().numpy().astype(int)
        
        # Null check for scaling parameters
        if new_size[0] == 0 or new_size[1] == 0:
            continue
            
        # Calculate scaling parameters
        crop_w, crop_h = new_x2 - new_x1, new_y2 - new_y1
        scale_x = crop_w / new_size[0]
        scale_y = crop_h / new_size[1]
        
        # Scale boxes using fixed function
        scaled_boxes = scale_boxes(
            boxes.copy(), pad_x, pad_y, scale_x, scale_y,
            {'x1': new_x1, 'y1': new_y1, 
             'resized_w': new_size[0], 'resized_h': new_size[1]}
        )
        
        # Check if any valid boxes exist
        if scaled_boxes.size == 0:
            continue
            
        for box_idx, (scaled_box, label, conf) in enumerate(zip(scaled_boxes, labels, confs)):
            # Skip if class doesn't match original detection
            if label != original_label:
                continue
            
            current_iou = calculate_iou(pre_box, scaled_box)
            
            # Track best match using IoU and confidence
            if current_iou > best_iou or (current_iou == best_iou and conf > best_conf):
                best_iou = current_iou
                best_conf = conf
                best_match = scaled_box
            
        # Only add if confidence improves
        if best_match is not None:
            min_iou_threshold = 0.25  # Adjust based on your use case
            if best_iou >= min_iou_threshold and best_conf > original_score:
                replacement_candidates.append({
                    'idx': i,
                    'box': best_match.tolist(),
                    # 'box': predictions['boxes'][i],
                    'score': best_conf,
                    'label': original_label
                })

    # Apply replacements after full iteration
    final_predictions = {
    'boxes': predictions['boxes'].copy(),
    'scores': predictions['scores'].copy(),
    'labels': predictions['labels'].copy()
    }

    for candidate in replacement_candidates:
        i = candidate['idx']
        if final_predictions['scores'][i] < candidate['score']:
            final_predictions['boxes'][i] = candidate['box']
            final_predictions['scores'][i] = candidate['score']
            final_predictions['labels'][i] = candidate['label']
    
    #next two lines for counting
    img_correct = 0
    used_truth_indices = []
     

    # Confidence-aware NMS
    # if predictions['boxes']:
    #     boxes_tensor = torch.tensor(predictions['boxes'])
    #     scores_tensor = torch.tensor(predictions['scores'])
    #     labels_tensor = torch.tensor(predictions['labels'])
        
    #     keep_indices = simple_nms(boxes_tensor, scores_tensor, 0.7)
        
    #     final_boxes = boxes_tensor[keep_indices].tolist()
    #     final_scores = scores_tensor[keep_indices].tolist()
    #     final_labels = labels_tensor[keep_indices].tolist()
    # else:
    #     final_boxes, final_scores, final_labels = [], [], []

    boxes_tensor = torch.tensor(final_predictions['boxes'])
    scores_tensor = torch.tensor(final_predictions['scores'])
    labels_tensor = torch.tensor(final_predictions['labels'])

    keep_indices = simple_nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

    filtered_predictions = {
    'boxes': boxes_tensor[keep_indices].tolist(),
    'scores': scores_tensor[keep_indices].tolist(),
    'labels': labels_tensor[keep_indices].tolist()
    }

    #code for counting
    pred_boxes = np.array(filtered_predictions['boxes'])
    pred_scores = np.array(filtered_predictions['scores'])
    pred_labels = np.array(filtered_predictions['labels'])
    true_boxes = np.array(true_boxes)
    true_labels = np.array(true_labels)

    #code for counting
    for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        total_predictions += 1
        
        # Find matching ground truth boxes (same class)
        matching_truths = np.where(true_labels == pred_label)[0]
        best_iou = 0
        best_truth_idx = -1
        
        for truth_idx in matching_truths:
            if truth_idx in used_truth_indices:
                continue  # Already matched
            
            iou = calculate_iou(pred_box, true_boxes[truth_idx])
            if iou > best_iou:
                best_iou = iou
                best_truth_idx = truth_idx
        
        if best_iou >= iou_threshold and best_truth_idx != -1:
            correct_predictions += 1
            img_correct += 1
            used_truth_indices.append(best_truth_idx)

    #ends counting
    # Update metrics
    # print(f'Final boxes: {filtered_predictions["boxes"]}')
    # print(f'Final scores: {filtered_predictions["scores"]}')

    # Convert filtered predictions to metric-compatible format
    preds = [{
        'boxes': torch.tensor(filtered_predictions['boxes']) if filtered_predictions['boxes'] else torch.zeros((0, 4)),
        'scores': torch.tensor(filtered_predictions['scores']) if filtered_predictions['scores'] else torch.zeros(0),
        'labels': torch.tensor(filtered_predictions['labels']) if filtered_predictions['labels'] else torch.zeros(0),
    }]

    # Ground truth
    targets = [{
        'boxes': torch.tensor(true_boxes) if true_boxes.size > 0 else torch.zeros((0, 4)),
        'labels': torch.tensor(true_labels) if true_labels.size > 0 else torch.zeros(0),
    }]
    metric.update(preds, targets)

# Final metrics
final_metrics = metric.compute()
print(f"mAP@0.5: {final_metrics['map_50']:.4f}")
print(f"Precision: {final_metrics['map_per_class'].mean():.4f}")
print(f"Recall: {final_metrics['mar_100'].mean():.4f}")

print("\nFinal Statistics:")
print(f"Total Correct Predictions: {correct_predictions}")
print(f"Total Predictions Made: {total_predictions}")
print(f"Precision: {correct_predictions/(total_predictions + 1e-7):.4f}")