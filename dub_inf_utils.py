import numpy as np
import pandas as pd 
from collections import defaultdict
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
    
    # Ensure all boxes have exactly 4 elements
    boxes = [box for box in boxes if len(box) == 4]
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
    """
    Calculate precision and recall across all images
    """
    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives
    
    # Group predictions and targets by image
    for preds, targets in zip(all_predictions, all_targets):
        pred_boxes = preds['boxes']
        pred_scores = preds['scores']
        pred_labels = preds['labels']
        true_boxes = targets['boxes']
        true_labels = targets['labels']
        
        # Skip if no predictions or no ground truth
        if len(pred_boxes) == 0 or len(true_boxes) == 0:
            total_fn += len(true_boxes)  # All ground truths are false negatives
            continue
            
        # Convert tensors to numpy for easier handling
        pred_boxes = pred_boxes.numpy()
        pred_labels = pred_labels.numpy()
        true_boxes = true_boxes.numpy()
        true_labels = true_labels.numpy()
        
        # Track matched ground truth boxes
        matched_gt = set()
        
        # For each prediction, find best matching ground truth
        for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            best_iou = 0.5
            best_gt_idx = -1
            
            # Find best matching ground truth box
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
        
        # Count unmatched ground truth boxes as false negatives
        total_fn += len(true_boxes) - len(matched_gt)
    
    # Calculate precision and recall
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
            # Get class-specific predictions and targets
            class_pred_mask = pred_labels == class_id
            class_true_mask = true_labels == class_id
            
            class_pred_boxes = pred_boxes[class_pred_mask]
            class_pred_scores = pred_scores[class_pred_mask]
            class_true_boxes = true_boxes[class_true_mask]
            
            if len(class_true_boxes) == 0:
                continue
                
            # Sort predictions by confidence
            score_sort = np.argsort(-class_pred_scores)
            class_pred_boxes = class_pred_boxes[score_sort]
            class_pred_scores = class_pred_scores[score_sort]
            
            # Calculate precision and recall points
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
            
            # Compute precision and recall
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / len(class_true_boxes)
            precisions = cum_tp / (cum_tp + cum_fp)
            
            # Compute AP using 11-point interpolation
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11
            
            class_aps[int(class_id)].append(ap)
    
    # Calculate mean AP for each class
    mean_aps = {class_id: np.mean(aps) for class_id, aps in class_aps.items()}
    map_value = np.mean(list(mean_aps.values()))
    
    return map_value, mean_aps

def calculate_map50_95(predictions, targets):
    """Calculate mAP@50-95"""
    iou_thresholds = np.linspace(0.5, 0.95, 10)  # [0.5, 0.55, ..., 0.95]
    maps = []
    class_maps = defaultdict(list)
    
    print("\nCalculating mAP at different IoU thresholds:")
    for iou_threshold in iou_thresholds:
        map_value, class_aps = calculate_map(predictions, targets, iou_threshold)
        maps.append(map_value)
        
        # Store per-class APs
        for class_id, ap in class_aps.items():
            class_maps[class_id].append(ap)
            
        print(f"mAP@{iou_threshold:.2f}: {map_value:.4f}")
    
    # Calculate mAP@50-95
    map50_95 = np.mean(maps)
    
    # Calculate per-class mAP@50-95
    class_map50_95 = {class_id: np.mean(aps) for class_id, aps in class_maps.items()}
    
    return map50_95, maps[0], class_map50_95

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