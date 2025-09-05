import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
import json
import time
import logging
import concurrent.futures
from pathlib import Path
import cv2  # Added for faster image operations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("double_inference.log"),
        logging.StreamHandler()
    ]
)

IMAGE_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
LABEL_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"
DATA_YAML = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml"
MODEL_WEIGHTS = "/kaggle/input/yolo-weights/weights/spdp2p2.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5 
NMS_IOU_THRESHOLD = 0.45

# Global model cache to avoid reloading
_model_cache = None

def get_model():
    """Singleton pattern for model loading"""
    global _model_cache
    if _model_cache is None:
        _model_cache = YOLO(MODEL_WEIGHTS)
        # Warm up the model with a dummy inference
        dummy_img = torch.zeros((1, 3, 640, 640))
        with torch.no_grad():
            _model_cache.predict(dummy_img, verbose=False)
    return _model_cache

# ---------------- OPTIMIZED UTILS ----------------
@torch.jit.script
def calculate_iou_tensor(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Vectorized IoU calculation using PyTorch tensors"""
    x1_inter = torch.max(box1[0], box2[0])
    y1_inter = torch.max(box1[1], box2[1])
    x2_inter = torch.min(box1[2], box2[2])
    y2_inter = torch.min(box1[3], box2[3])
    
    # Check for valid intersection
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return torch.tensor(0.0)
    
    area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    if area_box1 <= 0 or area_box2 <= 0:
        return torch.tensor(0.0)
    
    area_union = area_box1 + area_box2 - area_inter
    return area_inter / area_union if area_union > 0 else torch.tensor(0.0)

def calculate_iou(box1, box2):
    """Legacy NumPy version for compatibility"""
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    
    box1_tensor = torch.tensor(box1, dtype=torch.float32)
    box2_tensor = torch.tensor(box2, dtype=torch.float32)
    return calculate_iou_tensor(box1_tensor, box2_tensor).item()

def calculate_optimal_crop_batch(detections, img_width, img_height, pad_factor=0.2):
    """Batch process crop calculations"""
    crops = []
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        sw = max(1, x2 - x1)
        sh = max(1, y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        pad_w = sw * pad_factor
        pad_h = sh * pad_factor
        crop_w = sw + 2 * pad_w
        crop_h = sh + 2 * pad_h
        new_x1 = max(0, int(cx - crop_w / 2))
        new_y1 = max(0, int(cy - crop_h / 2))
        new_x2 = min(img_width, int(cx + crop_w / 2))
        new_y2 = min(img_height, int(cy + crop_h / 2))
        actual_w = new_x2 - new_x1
        actual_h = new_y2 - new_y1
        if actual_w < 10 or actual_h < 10:
            min_size = 32
            new_x1 = max(0, int(cx - min_size / 2))
            new_y1 = max(0, int(cy - min_size / 2))
            new_x2 = min(img_width, int(cx + min_size / 2))
            new_y2 = min(img_height, int(cy + min_size / 2))
        crops.append({'x1': new_x1, 'y1': new_y1, 'x2': new_x2, 'y2': new_y2})
    return crops

def prepare_cropped_image_cv2(img_array, crop_info):
    """Optimized image processing using OpenCV"""
    crop = img_array[crop_info['y1']:crop_info['y2'], crop_info['x1']:crop_info['x2']]
    original_h, original_w = crop.shape[:2]
    
    if original_h == 0 or original_w == 0:
        return None
    
    ratio = min(640 / original_w, 640 / original_h)
    new_size = (int(original_w * ratio), int(original_h * ratio))
    
    # Use OpenCV for faster resizing
    resized = cv2.resize(crop, new_size, interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded_img = np.full((640, 640, 3), 114, dtype=np.uint8)
    pad_x = (640 - new_size[0]) // 2
    pad_y = (640 - new_size[1]) // 2
    padded_img[pad_y:pad_y + new_size[1], pad_x:pad_x + new_size[0]] = resized
    
    return {
        'image': Image.fromarray(padded_img),
        'original_size': (original_w, original_h),
        'new_size': new_size,
        'pad_x': pad_x,
        'pad_y': pad_y,
        'ratio': ratio
    }

def scale_boxes_vectorized(boxes, pad_x, pad_y, crop_info, ratio):
    """Vectorized box scaling"""
    if boxes.size == 0:
        return np.array([])
    
    scaled = boxes.copy()
    # Vectorized operations
    scaled[:, [0, 2]] -= pad_x
    scaled[:, [1, 3]] -= pad_y
    scaled /= ratio
    scaled[:, [0, 2]] += crop_info['x1']
    scaled[:, [1, 3]] += crop_info['y1']
    return scaled

def torchvision_nms(boxes, scores, labels, iou_threshold=0.45):
    """Use PyTorch's optimized NMS implementation"""
    if not boxes or len(boxes) == 0:
        return [], [], []
    
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    
    keep_indices = []
    unique_labels = torch.unique(labels_tensor)
    
    for label in unique_labels:
        mask = labels_tensor == label
        if not mask.any():
            continue
            
        label_boxes = boxes_tensor[mask]
        label_scores = scores_tensor[mask]
        
        # Use torchvision NMS if available, otherwise use custom
        try:
            import torchvision
            keep = torchvision.ops.nms(label_boxes, label_scores, iou_threshold)
            original_indices = torch.where(mask)[0]
            keep_indices.extend(original_indices[keep].tolist())
        except ImportError:
            # Fallback to original implementation for this label
            label_indices = torch.where(mask)[0]
            sorted_indices = torch.argsort(label_scores, descending=True)
            
            selected = []
            remaining = sorted_indices.tolist()
            
            while remaining:
                current = remaining.pop(0)
                selected.append(current)
                
                if not remaining:
                    break
                
                current_box = label_boxes[current]
                remaining_boxes = label_boxes[remaining]
                
                # Calculate IoU with remaining boxes
                ious = torch.zeros(len(remaining))
                for i, rem_idx in enumerate(remaining):
                    ious[i] = calculate_iou_tensor(current_box, remaining_boxes[i])
                
                # Remove boxes with high IoU
                remaining = [remaining[i] for i in range(len(remaining)) if ious[i] <= iou_threshold]
            
            keep_indices.extend(label_indices[selected].tolist())
    
    keep_indices = sorted(keep_indices)
    return (boxes_tensor[keep_indices].tolist(), 
            scores_tensor[keep_indices].tolist(), 
            labels_tensor[keep_indices].tolist())

# ---------------- BATCH DOUBLE INFERENCE ----------------
def perform_batch_double_inference(image_path, model, detections, use_augment=False):
    """Process multiple detections in a single batch"""
    try:
        start_extra = time.time()
        
        # Load image once
        img_pil = Image.open(image_path).convert("RGB")
        img_array = np.array(img_pil)
        img_width, img_height = img_pil.size
        
        # Calculate all crops at once
        crop_infos = calculate_optimal_crop_batch(detections, img_width, img_height)
        
        # Process all crops
        processed_images = []
        valid_detections = []
        valid_crop_infos = []
        
        for detection, crop_info in zip(detections, crop_infos):
            if crop_info['x2'] <= crop_info['x1'] or crop_info['y2'] <= crop_info['y1']:
                continue
                
            processed = prepare_cropped_image_cv2(img_array, crop_info)
            if processed is None:
                continue
                
            processed_images.append(processed)
            valid_detections.append(detection)
            valid_crop_infos.append(crop_info)
        
        if not processed_images:
            return [], time.time() - start_extra
        
        # Batch inference - process multiple crops at once if model supports it
        all_refined = []
        batch_size = min(4, len(processed_images))  # Adjust based on GPU memory
        
        for i in range(0, len(processed_images), batch_size):
            batch_images = processed_images[i:i+batch_size]
            batch_detections = valid_detections[i:i+batch_size]
            batch_crops = valid_crop_infos[i:i+batch_size]
            
            # Single batch prediction
            batch_imgs = [p['image'] for p in batch_images]
            
            with torch.no_grad():
                if len(batch_imgs) == 1:
                    results = model.predict(batch_imgs[0], verbose=False, augment=use_augment)
                else:
                    # For batch processing, predict each individually for now
                    # YOLO doesn't always support true batch prediction in predict()
                    results = []
                    for img in batch_imgs:
                        result = model.predict(img, verbose=False, augment=use_augment)
                        results.extend(result)
            
            # Process results for this batch
            for j, (result, detection, processed, crop_info) in enumerate(zip(results, batch_detections, batch_images, batch_crops)):
                if len(result.boxes) == 0:
                    continue
                
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy().astype(int)
                
                if boxes.ndim == 1:
                    boxes = np.expand_dims(boxes, axis=0)
                    confs = np.expand_dims(confs, axis=0)
                    labels = np.expand_dims(labels, axis=0)
                
                scaled_boxes = scale_boxes_vectorized(
                    boxes, processed['pad_x'], processed['pad_y'], crop_info, processed['ratio']
                )
                
                refined = process_refined_boxes_optimized(scaled_boxes, labels, confs, detection, img_width, img_height)
                if refined is not None:
                    all_refined.append(refined)
        
        extra_time = time.time() - start_extra
        return all_refined, extra_time
        
    except Exception as e:
        logging.error(f"Error in batch double inference: {str(e)}")
        return [], 0.0

def process_refined_boxes_optimized(scaled_boxes, labels, confs, original_detection, img_width, img_height):
    """Optimized version with early exits and vectorized operations"""
    if scaled_boxes.size == 0 or len(labels) == 0 or len(confs) == 0:
        return None
    if not (len(scaled_boxes) == len(labels) == len(confs)):
        return None
    
    original_label = original_detection['category_id']
    original_score = original_detection['score']
    original_bbox = torch.tensor(original_detection['bbox'], dtype=torch.float32)
    
    # Filter by label first
    label_mask = labels == original_label
    if not label_mask.any():
        return None
    
    valid_boxes = scaled_boxes[label_mask]
    valid_confs = confs[label_mask]
    valid_labels = labels[label_mask]
    
    # Vectorized bounds checking
    bounds_mask = (
        (valid_boxes[:, 2] > valid_boxes[:, 0]) & 
        (valid_boxes[:, 3] > valid_boxes[:, 1]) & 
        (valid_boxes[:, 0] >= 0) & 
        (valid_boxes[:, 1] >= 0) & 
        (valid_boxes[:, 2] <= img_width) & 
        (valid_boxes[:, 3] <= img_height)
    )
    
    if not bounds_mask.any():
        return None
    
    final_boxes = valid_boxes[bounds_mask]
    final_confs = valid_confs[bounds_mask]
    final_labels = valid_labels[bounds_mask]
    
    # Vectorized IoU calculation
    best_idx = -1
    best_combined = -1
    
    for i, (box, conf, label) in enumerate(zip(final_boxes, final_confs, final_labels)):
        current_iou = calculate_iou_tensor(original_bbox, torch.tensor(box, dtype=torch.float32)).item()
        if current_iou < 0.25:
            continue
        
        combined_score = (conf * 0.6) + (current_iou * 0.4)
        if combined_score > best_combined:
            best_combined = combined_score
            best_idx = i
    
    if best_idx >= 0 and final_confs[best_idx] > original_score:
        return {
            'bbox': final_boxes[best_idx].tolist(),
            'score': float(final_confs[best_idx]),
            'category_id': int(final_labels[best_idx])
        }
    
    return None

# ---------------- OPTIMIZED METRICS ----------------
@torch.jit.script
def calculate_metrics_optimized(pred_boxes: torch.Tensor, pred_scores: torch.Tensor, pred_labels: torch.Tensor,
                              target_boxes: torch.Tensor, target_labels: torch.Tensor, iou_threshold: float = 0.5):
    """JIT compiled metrics calculation"""
    tp, fp, fn = 0, 0, 0
    
    if pred_boxes.numel() == 0:
        return tp, fp, len(target_boxes)
    
    if target_boxes.numel() == 0:
        return tp, len(pred_boxes), fn
    
    matched_targets = torch.zeros(len(target_boxes), dtype=torch.bool)
    
    for i in range(len(pred_boxes)):
        pred_box = pred_boxes[i]
        pred_label = pred_labels[i]
        
        best_iou = 0.0
        best_idx = -1
        
        for j in range(len(target_boxes)):
            if matched_targets[j] or target_labels[j] != pred_label:
                continue
                
            iou = calculate_iou_tensor(pred_box, target_boxes[j]).item()
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_idx = j
        
        if best_idx >= 0:
            tp += 1
            matched_targets[best_idx] = True
        else:
            fp += 1
    
    fn = len(target_boxes) - matched_targets.sum().item()
    return tp, fp, fn

def calculate_metrics(predictions, targets):
    """Optimized metrics calculation"""
    if not predictions or not targets:
        return {'map_50': 0, 'precision': 0, 'recall': 0}
    
    # Use torchmetrics for mAP
    metric = MeanAveragePrecision(class_metrics=True, max_detection_thresholds=None)
    metric.update(predictions, targets)
    results = metric.compute()
    
    # Calculate precision/recall with optimized function
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for pred, target in zip(predictions, targets):
        tp, fp, fn = calculate_metrics_optimized(
            pred['boxes'], pred['scores'], pred['labels'],
            target['boxes'], target['labels'], IOU_THRESHOLD
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    
    return {
        'map_50': results['map_50'].item(),
        'precision': precision,
        'recall': recall,
        'class_metrics': {f'class_{i}': {'ap': ap.item()} for i, ap in enumerate(results['map_per_class'])}
    }

# ---------------- OPTIMIZED IMAGE PROCESSING ----------------
def process_image_optimized(image_path, image_predictions, model_names, use_augment=False):
    """Optimized image processing with batch inference"""
    try:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        if image_name not in image_predictions:
            logging.warning(f"No predictions found for {image_name}")
            return None, 0.0
        
        # Load ground truth
        img = Image.open(image_path).convert("RGB")
        img_width, img_height = img.size
        label_path = os.path.join(LABEL_DIR, f"{image_name}.txt")
        true_boxes, true_labels = load_ground_truth(label_path, img_width, img_height)
        
        current_predictions = image_predictions[image_name]
        
        # Prepare all detections for batch processing
        detections_to_refine = []
        detection_indices = []
        
        for idx in range(len(current_predictions['scores'])):
            if current_predictions['scores'][idx] >= CONF_THRESHOLD:
                original_detection = {
                    'bbox': current_predictions['boxes'][idx],
                    'score': current_predictions['scores'][idx],
                    'category_id': current_predictions['labels'][idx]
                }
                detections_to_refine.append(original_detection)
                detection_indices.append(idx)
        
        # Batch process all detections
        model = get_model()  # Use cached model
        refined_results, total_extra_time = perform_batch_double_inference(
            image_path, model, detections_to_refine, use_augment
        )
        
        # Apply refined results
        for refined, idx in zip(refined_results, detection_indices):
            if refined is not None:
                current_predictions['boxes'][idx] = refined['bbox']
                current_predictions['scores'][idx] = refined['score']
                current_predictions['labels'][idx] = refined['category_id']
        
        # Apply NMS
        if current_predictions['boxes']:
            current_predictions['boxes'], current_predictions['scores'], current_predictions['labels'] = \
                torchvision_nms(
                    current_predictions['boxes'],
                    current_predictions['scores'],
                    current_predictions['labels'],
                    NMS_IOU_THRESHOLD
                )
        
        results = {
            'image_path': image_path,
            'refined_predictions': {
                'boxes': current_predictions['boxes'],
                'scores': current_predictions['scores'],
                'labels': current_predictions['labels']
            },
            'ground_truth': {
                'boxes': true_boxes,
                'labels': true_labels
            }
        }
        return results, total_extra_time
        
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None, 0.0

# Keep original utility functions for compatibility
def load_image_predictions(predictions_path, conf_threshold=0.25):
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"❌ Predictions file not found at {predictions_path}")
    with open(predictions_path, "r") as f:
        val_predictions = json.load(f)
    image_predictions = {}
    for pred in val_predictions:
        image_name = pred["image_id"]
        if image_name not in image_predictions:
            image_predictions[image_name] = {"boxes": [], "scores": [], "labels": []}
        if pred["score"] >= conf_threshold:
            x, y, w, h = pred["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            image_predictions[image_name]["boxes"].append([x1, y1, x2, y2])
            image_predictions[image_name]["scores"].append(pred["score"])
            image_predictions[image_name]["labels"].append(pred["category_id"])
    return image_predictions

def load_ground_truth(label_path, img_width, img_height):
    boxes, labels = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    x1 = (x_center - width/2) * img_width
                    y1 = (y_center - height/2) * img_height
                    x2 = (x_center + width/2) * img_width
                    y2 = (y_center + height/2) * img_height
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(class_id))
    return boxes, labels

# ---------------- MAIN ----------------
def main():
    start_time = time.time()
    logging.info("Loading YOLO model...")
    model = get_model()  # Use optimized model loading
    model_names = model.names
    
    logging.info("Loading predictions...")
    predictions_path = "/kaggle/input/json-files/spdp2p2.json"
    image_predictions = load_image_predictions(predictions_path, CONF_THRESHOLD)
    
    image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                  if os.path.splitext(f)[0] in image_predictions]
    
    logging.info(f"Processing {len(image_files)} images with optimized double inference...")
    
    results = {}
    all_predictions, all_targets = [], []
    total_extra_time = 0.0
    
    # Use optimal number of workers based on system
    max_workers = min(4, len(image_files))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {
            executor.submit(process_image_optimized, image_path, image_predictions, model_names, True): image_path
            for image_path in image_files
        }
        
        for future in concurrent.futures.as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                result, extra_time = future.result()
                total_extra_time += extra_time
                if result:
                    results[image_path] = result
                    if result['refined_predictions']['boxes'] and result['ground_truth']['boxes']:
                        pred = {
                            'boxes': torch.tensor(result['refined_predictions']['boxes']),
                            'scores': torch.tensor(result['refined_predictions']['scores']),
                            'labels': torch.tensor(result['refined_predictions']['labels']),
                        }
                        target = {
                            'boxes': torch.tensor(result['ground_truth']['boxes']),
                            'labels': torch.tensor(result['ground_truth']['labels']),
                        }
                        all_predictions.append(pred)
                        all_targets.append(target)
            except Exception as e:
                logging.error(f"Error with {image_path}: {str(e)}")
    
    metrics = calculate_metrics(all_predictions, all_targets)
    total_time = time.time() - start_time
    avg_extra_time = total_extra_time / max(1, len(image_files))
    
    logging.info("===== OPTIMIZED DOUBLE INFERENCE METRICS =====")
    logging.info(f"mAP@0.5: {metrics['map_50']:.4f}")
    logging.info(f"Precision: {metrics['precision']:.4f}")
    logging.info(f"Recall: {metrics['recall']:.4f}")
    logging.info(f"Processing time: {total_time:.2f} seconds")
    logging.info(f"Average extra inference time per image: {avg_extra_time:.4f} seconds")
    logging.info(f"Relative overhead ratio: {(avg_extra_time / (total_time - total_extra_time))*100:.2f}%")
    
    return results, metrics

if __name__ == "__main__":
    results, metrics = main()
