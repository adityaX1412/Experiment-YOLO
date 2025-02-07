import os
import json
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
from collections import defaultdict

# Constants - Adjusted thresholds
IMAGE_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
LABEL_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"
MODEL_WEIGHTS = "/kaggle/input/yolo-weights/weights/spdld.pt"
CONF_THRESHOLD = 0.25  # Lowered to match YOLO's default
IOU_THRESHOLD = 0.5    # Standard COCO metric
NMS_IOU_THRESHOLD = 0.45  # Non-Maximum Suppression (NMS) threshold

# Load YOLO model
model = YOLO(MODEL_WEIGHTS)

# Load predictions
predictions_path = "/kaggle/input/waid-preds/predictions.json"
if not os.path.exists(predictions_path):
    raise FileNotFoundError(f"❌ Predictions file not found at {predictions_path}")

with open(predictions_path, "r") as f:
    val_predictions = json.load(f)

# Convert JSON predictions into per-image format with confidence filtering
image_predictions = {}
for pred in val_predictions:
    image_name = pred["image_id"]
    if image_name not in image_predictions:
        image_predictions[image_name] = {"boxes": [], "scores": [], "labels": []}
    
    # Only add predictions above confidence threshold
    if pred["score"] >= CONF_THRESHOLD:
        x, y, w, h = pred["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h
        image_predictions[image_name]["boxes"].append([x1, y1, x2, y2])
        image_predictions[image_name]["scores"].append(pred["score"])
        image_predictions[image_name]["labels"].append(pred["category_id"])

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1 + area2 - intersection + 1e-6)

def non_max_suppression_per_class(boxes, scores, labels, iou_threshold):
    """Apply Non-Maximum Suppression per class to avoid suppressing different categories"""
    unique_classes = np.unique(labels)
    final_boxes, final_scores, final_labels = [], [], []
    for cls in unique_classes:
        cls_mask = (np.array(labels) == cls)
        cls_boxes = np.array(boxes)[cls_mask]
        cls_scores = np.array(scores)[cls_mask]
        cls_labels = np.array(labels)[cls_mask]
        
        if len(cls_boxes) == 0:
            continue
        
        indices = np.argsort(cls_scores)[::-1]
        keep = []
        
        while indices.size > 0:
            current = indices[0]
            keep.append(current)
            
            if indices.size == 1:
                break
            
            ious = np.array([calculate_iou(cls_boxes[current], cls_boxes[i]) for i in indices[1:]])
            indices = indices[1:][ious < iou_threshold]

        final_boxes.extend(cls_boxes[keep])
        final_scores.extend(cls_scores[keep])
        final_labels.extend(cls_labels[keep])

    return final_boxes, final_scores, final_labels

# Initialize metrics
metric = MeanAveragePrecision(class_metrics=True)
total_predictions = 0
correct_predictions = 0
all_predictions = []
all_targets = []

# Process each image
for image_path in os.listdir(IMAGE_DIR):
    img = Image.open(os.path.join(IMAGE_DIR, image_path)).convert("RGB")
    
    # Load ground truth labels
    true_boxes, true_labels = [], []
    label_path = os.path.join(LABEL_DIR, os.path.splitext(image_path)[0] + '.txt')
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x1 = (x_center - width/2) * img.width
                y1 = (y_center - height/2) * img.height
                x2 = (x_center + width/2) * img.width
                y2 = (y_center + height/2) * img.height
                true_boxes.append([x1, y1, x2, y2])
                true_labels.append(int(class_id))

    # Get predictions for current image
    image_name = os.path.splitext(image_path)[0]
    if image_name not in image_predictions:
        continue

    current_predictions = image_predictions[image_name]
    
    # Apply class-aware NMS
    current_predictions['boxes'], current_predictions['scores'], current_predictions['labels'] = \
        non_max_suppression_per_class(
            current_predictions['boxes'],
            current_predictions['scores'],
            current_predictions['labels'],
            NMS_IOU_THRESHOLD
        )

    # Update total predictions AFTER NMS
    total_predictions += len(current_predictions['boxes'])
    
    # Track matched ground truth boxes to avoid double-counting
    matched_gt = set()
    
    # Check each prediction against ground truth
    for i, (pred_box, pred_score, pred_label) in enumerate(zip(
        current_predictions['boxes'], 
        current_predictions['scores'], 
        current_predictions['labels']
    )):
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth box
        for gt_idx, (true_box, true_label) in enumerate(zip(true_boxes, true_labels)):
            if gt_idx in matched_gt:
                continue
                
            iou = calculate_iou(pred_box, true_box)
            if iou > best_iou and pred_label == true_label:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # If good match found, count as correct
        if best_iou >= IOU_THRESHOLD:
            correct_predictions += 1
            matched_gt.add(best_gt_idx)

    # Prepare for mAP evaluation
    preds = [{
        'boxes': torch.tensor(current_predictions['boxes']),
        'scores': torch.tensor(current_predictions['scores']),
        'labels': torch.tensor(current_predictions['labels']),
    }]
    targets = [{
        'boxes': torch.tensor(true_boxes),
        'labels': torch.tensor(true_labels),
    }]
    all_predictions.extend(preds)
    all_targets.extend(targets)
    metric.update(preds, targets)

# Compute final metrics
final_metrics = metric.compute()
precision = correct_predictions / (total_predictions + 1e-6)
recall = correct_predictions / (len(all_targets) + 1e-6)
map50 = final_metrics['map_50'].item()

print(f"\nFinal Metrics:")
print(f"mAP@0.5: {map50:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
if total_predictions > 0:
    print(f"Accuracy: {correct_predictions/total_predictions:.4f}")

