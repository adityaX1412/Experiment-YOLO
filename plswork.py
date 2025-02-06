import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
import json

IMAGE_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
LABEL_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"
DATA_YAML = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml"
MODEL_WEIGHTS = "/kaggle/input/yolo-weights/weights/spdld.pt"
CONF_THRESHOLD = 0.5  # Threshold for high-confidence detections
LOW_CONF_THRESHOLD = 0.25  # Trigger double inference if below this
IOU_THRESHOLD = 0.1  # IoU matching threshold


predictions_path = "/kaggle/input/waid-preds/predictions.json"  
if not os.path.exists(predictions_path):
    raise FileNotFoundError(f"❌ Predictions file not found at {predictions_path}")

with open(predictions_path, "r") as f:
    val_predictions = json.load(f)  # Load per-image detections

# 3️⃣ Convert JSON predictions into per-image format
image_predictions = {}  # Stores detections for each image
for pred in val_predictions:
    image_name = pred["image_id"]  # Image filename
    if image_name not in image_predictions:
        image_predictions[image_name] = {"boxes": [], "scores": [], "labels": []}

    # Convert COCO-style bbox (x, y, width, height) to YOLO format (x1, y1, x2, y2)
    x, y, w, h = pred["bbox"]
    x1, y1, x2, y2 = x, y, x + w, y + h

    # Store extracted detections
    image_predictions[image_name]["boxes"].append([x1, y1, x2, y2])
    image_predictions[image_name]["scores"].append(pred["score"])
    image_predictions[image_name]["labels"].append(pred["category_id"])
    
# Function to calculate IoU
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

# Function to rescale bounding boxes after second inference
def scale_boxes(boxes, pad_x, pad_y, resize_ratio_x, resize_ratio_y, crop_coords):
    """Rescales boxes to match the original image dimensions"""
    if not isinstance(boxes, np.ndarray) or boxes.size == 0:
        return np.empty((0, 4))

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] - pad_x, 0, crop_coords['resized_w'])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] - pad_y, 0, crop_coords['resized_h'])
    
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * resize_ratio_x + crop_coords['x1']
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * resize_ratio_y + crop_coords['y1']

    return boxes

# Initialize mAP metric
metric = MeanAveragePrecision(class_metrics=True)
total_predictions = 0
correct_predictions = 0

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

    # Retrieve first inference results from `val()`
    image_name = os.path.splitext(image_path)[0]
    if image_name not in image_predictions:
        continue
    predictions = image_predictions[image_name]

    # Perform second inference on low-confidence detections
    replacement_candidates = []
    for i, score in enumerate(predictions['scores']):
        if score >= CONF_THRESHOLD:
            continue  # Skip high-confidence detections

        # Adaptive cropping
        pre_box = predictions['boxes'][i]
        x1, y1, x2, y2 = pre_box
        crop = img.crop((x1, y1, x2, y2)).resize((640, 640))

        # Second pass inference
        refined_results = model.predict(crop, conf=0.1, verbose=False)
        refined_boxes = refined_results[0].boxes.xyxy.cpu().numpy()
        refined_scores = refined_results[0].boxes.conf.cpu().numpy()
        refined_labels = refined_results[0].boxes.cls.cpu().numpy().astype(int)

        # Scale boxes back
        scale_x, scale_y = (x2 - x1) / 640, (y2 - y1) / 640
        scaled_boxes = refined_boxes * [scale_x, scale_y, scale_x, scale_y]

        # Find best match
        best_iou, best_conf, best_match = -1, -1, None
        for j, refined_box in enumerate(scaled_boxes):
            iou = calculate_iou(pre_box, refined_box)
            if iou > best_iou or (iou == best_iou and refined_scores[j] > best_conf):
                best_iou, best_conf, best_match = iou, refined_scores[j], refined_box

        if best_match is not None and best_iou >= 0.25 and best_conf > score:
            replacement_candidates.append({'idx': i, 'box': best_match.tolist(), 'score': best_conf, 'label': predictions['labels'][i]})

    # Apply replacements
    for candidate in replacement_candidates:
        i = candidate['idx']
        predictions['boxes'][i] = candidate['box']
        predictions['scores'][i] = candidate['score']

    # Prepare for mAP evaluation
    preds = [{
        'boxes': torch.tensor(predictions['boxes']),
        'scores': torch.tensor(predictions['scores']),
        'labels': torch.tensor(predictions['labels']),
    }]
    targets = [{
        'boxes': torch.tensor(true_boxes),
        'labels': torch.tensor(true_labels),
    }]
    metric.update(preds, targets)

# Compute final metrics
final_metrics = metric.compute()
print(f"\nFinal Metrics:")
print(f"mAP@0.5: {final_metrics['map_50']:.4f}")
print(f"Precision: {final_metrics['map_per_class'].mean():.4f}")
print(f"Recall: {final_metrics['mar_100'].mean():.4f}")
print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
