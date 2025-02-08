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
DATA_YAML = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml"
MODEL_WEIGHTS = "/kaggle/input/waid-no-soap/vanillanosoap.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5 
NMS_IOU_THRESHOLD = 0.45
DOUBLE_INFERENCE_THRESHOLD = 0.1
NUM_WARMUP = 10
NUM_TRIALS = 100

def calculate_inference_metrics(model, image_path):
    """Calculate inference time and GFLOPS for a single image"""
    device = next(model.parameters()).device
    img = Image.open(image_path).convert("RGB")
    img = torch.from_numpy(np.array(img)).to(device).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)  # Add batch dimension
    
    # Warmup runs
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = model(img)
    
    # Measure inference time
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(NUM_TRIALS):
        with torch.no_grad():
            _ = model(img)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_inference_time = (end_time - start_time) / NUM_TRIALS
    
    # Calculate GFLOPS
    macs, params = profile(model, inputs=(img,))
    gflops = macs * 2 / 1e9  # Convert MACs to GFLOPS
    
    return avg_inference_time, gflops, params

model = YOLO(MODEL_WEIGHTS)

predictions_path = "/kaggle/input/json-files/vannillanosoap.json"
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
    if CONF_THRESHOLD <= pred["score"]:
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

def non_max_suppression(boxes, scores, labels, iou_threshold):
    """Apply Non-Maximum Suppression to remove overlapping boxes"""
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

def calculate_precision_recall(all_predictions, all_targets):
    """Calculate precision and recall across all images"""
    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives
    
    for preds, targets in zip(all_predictions, all_targets):
        pred_boxes = preds['boxes']
        pred_scores = preds['scores']
        pred_labels = preds['labels']
        true_boxes = targets['boxes']
        true_labels = targets['labels']
        
        if len(pred_boxes) == 0 or len(true_boxes) == 0:
            total_fn += len(true_boxes)
            continue
            
        pred_boxes = pred_boxes.numpy()
        pred_labels = pred_labels.numpy()
        true_boxes = true_boxes.numpy()
        true_labels = true_labels.numpy()
        
        matched_gt = set()
        
        for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            best_iou = IOU_THRESHOLD
            best_gt_idx = -1
            
            for j, (true_box, true_label) in enumerate(zip(true_boxes, true_labels)):
                if j in matched_gt:
                    continue
                iou = calculate_iou(pred_box, true_box)
                if iou > best_iou and pred_label == true_label:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_gt_idx >= 0:
                total_tp += 1
                matched_gt.add(best_gt_idx)
            else:
                total_fp += 1
        
        total_fn += len(true_boxes) - len(matched_gt)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    return precision, recall

def calculate_map(predictions, targets, iou_threshold):
    """Calculate mAP at a specific IoU threshold"""
    class_aps = defaultdict(list)
    
    for preds, tgts in zip(predictions, targets):
        pred_boxes = preds['boxes'].numpy()
        pred_scores = preds['scores'].numpy()
        pred_labels = preds['labels'].numpy()
        true_boxes = tgts['boxes'].numpy()
        true_labels = tgts['labels'].numpy()
        
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

def calculate_map50_95(predictions, targets):
    """Calculate mAP@50-95"""
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    maps = []
    class_maps = defaultdict(list)
    
    print("\nCalculating mAP at different IoU thresholds:")
    for iou_threshold in iou_thresholds:
        map_value, class_aps = calculate_map(predictions, targets, iou_threshold)
        maps.append(map_value)
        
        for class_id, ap in class_aps.items():
            class_maps[class_id].append(ap)
            
        print(f"mAP@{iou_threshold:.2f}: {map_value:.4f}")
    
    map50_95 = np.mean(maps)
    class_map50_95 = {class_id: np.mean(aps) for class_id, aps in class_maps.items()}
    
    return map50_95, maps[0], class_map50_95

def scale_boxes(padded_boxes, pad_x, pad_y, resize_ratio_x, resize_ratio_y, crop_coords):
    """Scale boxes to original image coordinates"""
    try:
        if padded_boxes is None or not isinstance(padded_boxes, np.ndarray):
            return np.empty((0, 4))
        
        if padded_boxes.size == 0:
            return np.empty((0, 4))
        if padded_boxes.ndim == 1:
            padded_boxes = np.expand_dims(padded_boxes, 0)
            
        boxes = padded_boxes.copy()
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] - pad_x, 0, crop_coords['resized_w'])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] - pad_y, 0, crop_coords['resized_h'])
        
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * resize_ratio_x + crop_coords['x1']
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * resize_ratio_y + crop_coords['y1']
        
        return boxes
    except Exception as e:
        print(f"Scaling error: {str(e)}")
        return np.empty((0, 4))

def perform_double_inference(image_path, model, original_detection):
    """Perform double inference with ROI refinement"""
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size
    
    x1, y1, x2, y2 = original_detection['bbox']
    sw = x2 - x1
    sh = y2 - y1
    original_score = original_detection['score']
    original_label = original_detection['category_id']
    
    cx, cy = (x1 + x2)/2, (y1 + y2)/2
    desired_width = (sw * 640) / (x2 - x1) if (x2 - x1) != 0 else 640
    desired_height = (sh * 640) / (y2 - y1) if (y2 - y1) != 0 else 640
    
    new_x1 = max(0, int(cx - desired_width/2))
    new_y1 = max(0, int(cy - desired_height/2))
    new_x2 = min(img_width, int(cx + desired_width/2))
    new_y2 = min(img_height, int(cy + desired_height/2))
    
    if (new_x2 <= new_x1) or (new_y2 <= new_y1):
        return None

    crop = img.crop((new_x1, new_y1, new_x2, new_y2))
    original_w, original_h = crop.size
    ratio = min(640/original_w, 640/original_h)
    new_size = (int(original_w*ratio), int(original_h*ratio))
    resized = crop.resize(new_size, Image.BILINEAR)
    
    padded_img = Image.new("RGB", (640, 640), (114, 114, 114))
    pad_x, pad_y = (640 - new_size[0])//2, (640 - new_size[1])//2
    padded_img.paste(resized, (pad_x, pad_y))

    with torch.no_grad():
        new_results = model.predict(padded_img, conf=DOUBLE_INFERENCE_THRESHOLD,verbose=False,augment=True)
    
    if len(new_results[0].boxes) == 0:
        return None

    boxes = new_results[0].boxes.xyxy.cpu().numpy()
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, axis=0)
        
    confs = new_results[0].boxes.conf.cpu().numpy()
    labels = new_results[0].boxes.cls.cpu().numpy().astype(int)
    
    scale_x = (new_x2 - new_x1) / new_size[0]

# Initialize metrics
metric = MeanAveragePrecision(class_metrics=True,extended_summary=True)
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
    
    # Process each prediction for potential refinement
    replacement_candidates = []
    for idx in range(len(current_predictions['scores'])):
        if current_predictions['scores'][idx] >= CONF_THRESHOLD:
            continue
            
        # Create detection object matching JSON format
        original_detection = {
            'bbox': current_predictions['boxes'][idx],
            'score': current_predictions['scores'][idx],
            'category_id': current_predictions['labels'][idx]
        }
        
        # Perform double inference
        refined = perform_double_inference(
            os.path.join(IMAGE_DIR, image_path),
            model,
            original_detection
        )
        
        if refined:
            replacement_candidates.append({
                'idx': idx,
                'bbox': refined['bbox'],
                'score': refined['score'],
                'label': refined['category_id']
            })
    
    # Apply replacements
    for candidate in replacement_candidates:
        i = candidate['idx']
        current_predictions['boxes'][i] = candidate['bbox']
        current_predictions['scores'][i] = candidate['score']
        current_predictions['labels'][i] = candidate['label']
    
    # Apply NMS to remove overlapping boxes
    current_predictions['boxes'], current_predictions['scores'], current_predictions['labels'] = \
        non_max_suppression(
            current_predictions['boxes'],
            current_predictions['scores'],
            current_predictions['labels'],
            NMS_IOU_THRESHOLD
        )

    # Update prediction counters
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
        
        # If good match found, count as correct and mark ground truth as matched
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
precision, recall = calculate_precision_recall(all_predictions, all_targets)
map50_95, map50, class_map50_95 = calculate_map50_95(all_predictions, all_targets)

print(f"\nFinal Metrics:")
print(f"mAP@0.5: {final_metrics['map_50']:.4f}")
print(f"Calculated mAP@50 : {map50:.4f}")
print(f"mAP@0.5-0.95: {final_metrics['map']:.4f}")
print(f"Calculated mAP@50-95: {map50_95:.4f}")
print(f"Recall: {final_metrics['mar_100']:.4f}")
print(f"Precision: {final_metrics['precision'].mean():.4f}")
print(f"calculated Precision: {precision:.4f}")
print(f"calculated Recall: {recall:.4f}")
print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
if total_predictions > 0:
    print(f"Accuracy: {correct_predictions/total_predictions:.4f}")