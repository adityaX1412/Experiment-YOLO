import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
import json
from collections import defaultdict
import time
import logging
from dub_inf_utils import *

IMAGE_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
LABEL_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"
DATA_YAML = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml"
MODEL_WEIGHTS = "/kaggle/input/ld70-waid-weight/best (3).pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5 
NMS_IOU_THRESHOLD = 0.45

model = YOLO(MODEL_WEIGHTS)

predictions_path = "/kaggle/input/ld70-json/predictions (1).json"
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
    if pred["score"] >= CONF_THRESHOLD:  # Compare single values, not lists
        x, y, w, h = pred["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h
        image_predictions[image_name]["boxes"].append([x1, y1, x2, y2])
        image_predictions[image_name]["scores"].append(pred["score"])
        image_predictions[image_name]["labels"].append(pred["category_id"])

def perform_double_inference(image_path, model, original_detection):
    """
    Perform double inference with inference time, GFLOPs, and visualizations.
    
    Args:
        image_path: Path to the original image
        model: YOLO model instance
        original_detection: Dictionary containing original detection info
            {'bbox': [x1, y1, x2, y2], 'score': float, 'category_id': int}
            
    Returns:
        Dictionary with refined detection or None if no improvement
    """
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size
    
    # Extract detection details
    x1, y1, x2, y2 = original_detection['bbox']
    sw = x2 - x1  # width of detection
    sh = y2 - y1  # height of detection
    original_score = original_detection['score']
    original_label = original_detection['category_id']
    
    # Early exit if box dimensions are invalid
    if sw <= 0 or sh <= 0:
        return None
    
    # Calculate adaptive crop region
    cx, cy = (x1 + x2)/2, (y1 + y2)/2  # center of detection
    
    # Calculate desired dimensions for the crop
    # Use target size of 640 and maintain aspect ratio
    scale = min(640/sw, 640/sh)
    desired_width = sw * scale
    desired_height = sh * scale
    
    # Calculate crop boundaries with padding
    pad_factor = 0.2  # 20% padding around detection
    new_x1 = max(0, int(cx - (desired_width * (1 + pad_factor))/2))
    new_y1 = max(0, int(cy - (desired_height * (1 + pad_factor))/2))
    new_x2 = min(img_width, int(cx + (desired_width * (1 + pad_factor))/2))
    new_y2 = min(img_height, int(cy + (desired_height * (1 + pad_factor))/2))
    
    # Check if crop region is valid
    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return None
        
    # Perform crop and resize
    crop = img.crop((new_x1, new_y1, new_x2, new_y2))
    original_w, original_h = crop.size
    ratio = min(640/original_w, 640/original_h)
    new_size = (int(original_w*ratio), int(original_h*ratio))
    resized = crop.resize(new_size, Image.BILINEAR)
    
    # Create padded image
    padded_img = Image.new("RGB", (640, 640), (114, 114, 114))
    pad_x = (640 - new_size[0])//2
    pad_y = (640 - new_size[1])//2
    padded_img.paste(resized, (pad_x, pad_y))
    
    # Perform second inference with test time augmentation
    with torch.no_grad():
        new_results = model.predict(padded_img, verbose=False, augment=True)
        
    # Check if any detections were made
    if len(new_results[0].boxes) == 0:
        return None
        
    # Extract and process detections
    boxes = new_results[0].boxes.xyxy.cpu().numpy()
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, axis=0)
        
    confs = new_results[0].boxes.conf.cpu().numpy()
    labels = new_results[0].boxes.cls.cpu().numpy().astype(int)
    
    # Calculate scaling factors
    scale_x = (new_x2 - new_x1) / new_size[0]
    scale_y = (new_y2 - new_y1) / new_size[1]
    
    # Scale boxes back to original image coordinates
    crop_info = {
        'x1': new_x1,
        'y1': new_y1,
        'resized_w': new_size[0],
        'resized_h': new_size[1]
    }
    scaled_boxes = scale_boxes(
        boxes.copy(),
        pad_x,
        pad_y,
        scale_x,
        scale_y,
        crop_info
    )

    for box in scaled_boxes:
        if (box[2] <= box[0]) or (box[3] <= box[1]) or \
        (box[0] < 0) or (box[1] < 0) or \
        (box[2] > img_width) or (box[3] > img_height):
            continue
    
        # Find best matching detection
        best_match = None
        best_combined = -1
        best_conf = -1

        for box, label, conf in zip(scaled_boxes, labels, confs):
            if label != original_label:
                continue
                
            current_iou = calculate_iou(original_detection['bbox'], box)
            combined_score = (conf * 0.1) + (current_iou * 0.9)  # Weighted combination
            
            if combined_score > best_combined and current_iou >= 0.25:
                best_combined = combined_score
                best_conf = conf
                best_match = {
                    'bbox': box.tolist(),
                    'score': float(conf),
                    'category_id': int(label)
                }
        
    # Return refined detection only if it improves on original
    return best_match if best_conf > original_score else None

# Initialize metrics
metric = MeanAveragePrecision(class_metrics=True, extended_summary=True)
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
    high_conf_boxes = []
    high_conf_scores = []
    high_conf_labels = []
    low_conf_boxes = []
    low_conf_scores = []
    low_conf_labels = []
    
    # Split predictions based on confidence
    for box, score, label in zip(current_predictions['boxes'], 
                               current_predictions['scores'], 
                               current_predictions['labels']):
        if score >= 0.5:
            high_conf_boxes.append(box)
            high_conf_scores.append(score)
            high_conf_labels.append(label)
        else:
            low_conf_boxes.append(box)
            low_conf_scores.append(score)
            low_conf_labels.append(label)
    
    # Process low confidence predictions with double inference
    refined_predictions = []
    for idx in range(len(low_conf_boxes)):
        original_detection = {
            'bbox': low_conf_boxes[idx],
            'score': low_conf_scores[idx],
            'category_id': low_conf_labels[idx]
        }
        
        refined = perform_double_inference(
            os.path.join(IMAGE_DIR, image_path),
            model,
            original_detection
        )
        
        if refined is not None:
            refined_predictions.append(refined)
    
    # Combine high confidence and refined predictions
    final_boxes = high_conf_boxes[:]
    final_scores = high_conf_scores[:]
    final_labels = high_conf_labels[:]
    
    for refined in refined_predictions:
        final_boxes.append(refined['bbox'])
        final_scores.append(refined['score'])
        final_labels.append(refined['category_id'])
    
    # Apply NMS if there are any predictions
    if final_boxes and final_scores and final_labels:
        final_boxes, final_scores, final_labels = non_max_suppression(
            final_boxes,
            final_scores,
            final_labels,
            NMS_IOU_THRESHOLD
        )
    
    # Update prediction counters
    total_predictions += len(final_boxes)
    
    # Track matched ground truth boxes
    matched_gt = set()
    
    for i, (pred_box, pred_score, pred_label) in enumerate(zip(
        final_boxes, final_scores, final_labels
    )):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, (true_box, true_label) in enumerate(zip(true_boxes, true_labels)):
            if gt_idx in matched_gt:
                continue
                
            iou = calculate_iou(pred_box, true_box)
            if iou > best_iou and pred_label == true_label:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= IOU_THRESHOLD:
            correct_predictions += 1
            matched_gt.add(best_gt_idx)

    # Convert boxes and labels to tensors for metrics
    if final_boxes and true_boxes:
        preds = [{
            'boxes': torch.tensor(final_boxes),
            'scores': torch.tensor(final_scores),
            'labels': torch.tensor(final_labels),
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
precision, recall = calculate_precision_recall(all_predictions, all_targets)
map50, class_aps = calculate_map(all_predictions, all_targets, IOU_THRESHOLD)

print(f"\nFinal Metrics:")
print(f"mAP@0.5: {final_metrics['map_50']:.4f}")
print(f"Calculated mAP@50: {map50:.4f}")
print(f"Calculated Precision: {precision:.4f}")
print(f"Calculated Recall: {recall:.4f}")
print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
