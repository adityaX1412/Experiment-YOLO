# combined_double_and_compare.py
# Single-file script that contains:
# 1) Double-stage inference + visualizations (saves GT, single, double as before)
# 2) Compare-mode: runs vanilla YOLO vs your custom model and saves side-by-side
#    visualizations only for images where the detected instance counts differ.

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
import cv2
import argparse

# ---------------- CONFIG ----------------
# Dataset / models
IMAGE_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
LABEL_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"
DATA_YAML = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml"
CUSTOM_MODEL_WEIGHTS = "/kaggle/input/yolo-weights/weights/spdp2p2.pt"  # your model
VANILLA_MODEL_WEIGHTS = "yolov8n.pt"  # change to yolov8s.pt / yolov8m.pt if desired
PREDICTIONS_JSON = "/kaggle/input/json-files/spdp2p2.json"

# Thresholds
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45

# Visualization roots
VIS_ROOT = "/kaggle/working/visualizations"
GT_DIR = os.path.join(VIS_ROOT, "gt")        # BLUE boxes (BGR: (255,0,0))
SINGLE_DIR = os.path.join(VIS_ROOT, "single")# RED boxes  (BGR: (0,0,255))
DOUBLE_DIR = os.path.join(VIS_ROOT, "double")# GREEN boxes (BGR: (0,255,0))

COMPARE_OUT = "/kaggle/working/compare_diff_instances"  # side-by-side saved here
os.makedirs(GT_DIR, exist_ok=True)
os.makedirs(SINGLE_DIR, exist_ok=True)
os.makedirs(DOUBLE_DIR, exist_ok=True)
os.makedirs(COMPARE_OUT, exist_ok=True)

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("combined_double_compare.log"),
        logging.StreamHandler()
    ]
)

# ---------------- MODEL LOADING (singleton) ----------------
_model_cache = None
_custom_model_cache = None
_vanilla_model_cache = None

def get_model(weights_path=None, cache_name='default'):
    """Singleton loader for YOLO models. Use weights_path to load specific weights."""
    global _model_cache, _custom_model_cache, _vanilla_model_cache
    if cache_name == 'default':
        if _model_cache is None:
            logging.info("Loading YOLO model weights: %s", weights_path)
            _model_cache = YOLO(weights_path)
        return _model_cache
    elif cache_name == 'custom':
        if _custom_model_cache is None:
            logging.info("Loading custom model weights: %s", weights_path)
            _custom_model_cache = YOLO(weights_path)
        return _custom_model_cache
    elif cache_name == 'vanilla':
        if _vanilla_model_cache is None:
            logging.info("Loading vanilla model weights: %s", weights_path)
            _vanilla_model_cache = YOLO(weights_path)
        return _vanilla_model_cache
    else:
        # fallback
        return YOLO(weights_path)

# ---------------- UTILITIES ----------------
@torch.jit.script
def calculate_iou_tensor(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    x1_inter = torch.max(box1[0], box2[0])
    y1_inter = torch.max(box1[1], box2[1])
    x2_inter = torch.min(box1[2], box2[2])
    y2_inter = torch.min(box1[3], box2[3])

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
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    b1 = torch.tensor(box1, dtype=torch.float32)
    b2 = torch.tensor(box2, dtype=torch.float32)
    return calculate_iou_tensor(b1, b2).item()


def calculate_optimal_crop_batch(detections, img_width, img_height, pad_factor=0.2):
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
    crop = img_array[crop_info['y1']:crop_info['y2'], crop_info['x1']:crop_info['x2']]
    original_h, original_w = crop.shape[:2]
    if original_h == 0 or original_w == 0:
        return None
    ratio = min(640 / original_w, 640 / original_h)
    new_size = (int(original_w * ratio), int(original_h * ratio))
    resized = cv2.resize(crop, new_size, interpolation=cv2.INTER_LINEAR)
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
    if boxes.size == 0:
        return np.array([])
    scaled = boxes.copy()
    scaled[:, [0, 2]] -= pad_x
    scaled[:, [1, 3]] -= pad_y
    scaled /= ratio
    scaled[:, [0, 2]] += crop_info['x1']
    scaled[:, [1, 3]] += crop_info['y1']
    return scaled


def torchvision_nms(boxes, scores, labels, iou_threshold=0.45):
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
        try:
            import torchvision
            keep = torchvision.ops.nms(label_boxes, label_scores, iou_threshold)
            original_indices = torch.where(mask)[0]
            keep_indices.extend(original_indices[keep].tolist())
        except Exception:
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
                ious = torch.zeros(len(remaining))
                for i, rem_idx in enumerate(remaining):
                    ious[i] = calculate_iou_tensor(current_box, remaining_boxes[i])
                remaining = [remaining[i] for i in range(len(remaining)) if ious[i] <= iou_threshold]
            keep_indices.extend(label_indices[selected].tolist())
    keep_indices = sorted(keep_indices)
    return (boxes_tensor[keep_indices].tolist(),
            scores_tensor[keep_indices].tolist(),
            labels_tensor[keep_indices].tolist())

# ---------------- BATCH DOUBLE INFERENCE ----------------
def perform_batch_double_inference(image_path, model, detections, use_augment=False):
    try:
        start_extra = time.time()
        img_pil = Image.open(image_path).convert("RGB")
        img_array = np.array(img_pil)
        img_width, img_height = img_pil.size
        crop_infos = calculate_optimal_crop_batch(detections, img_width, img_height)
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
        all_refined = []
        batch_size = min(4, len(processed_images))
        for i in range(0, len(processed_images), batch_size):
            batch_images = processed_images[i:i+batch_size]
            batch_detections = valid_detections[i:i+batch_size]
            batch_crops = valid_crop_infos[i:i+batch_size]
            batch_imgs = [p['image'] for p in batch_images]
            with torch.no_grad():
                if len(batch_imgs) == 1:
                    results = model.predict(batch_imgs[0], verbose=False, augment=use_augment)
                else:
                    results = []
                    for img in batch_imgs:
                        result = model.predict(img, verbose=False, augment=use_augment)
                        results.extend(result)
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
    if scaled_boxes.size == 0 or len(labels) == 0 or len(confs) == 0:
        return None
    if not (len(scaled_boxes) == len(labels) == len(confs)):
        return None
    original_label = original_detection['category_id']
    original_score = original_detection['score']
    original_bbox = torch.tensor(original_detection['bbox'], dtype=torch.float32)
    label_mask = labels == original_label
    if not label_mask.any():
        return None
    valid_boxes = scaled_boxes[label_mask]
    valid_confs = confs[label_mask]
    valid_labels = labels[label_mask]
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

# ---------------- METRICS ----------------
@torch.jit.script
def calculate_metrics_optimized(pred_boxes: torch.Tensor, pred_scores: torch.Tensor, pred_labels: torch.Tensor,
                                target_boxes: torch.Tensor, target_labels: torch.Tensor, iou_threshold: float = 0.5):
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
    if not predictions or not targets:
        return {'map_50': 0, 'precision': 0, 'recall': 0}
    metric = MeanAveragePrecision(class_metrics=True, max_detection_thresholds=None)
    metric.update(predictions, targets)
    results = metric.compute()
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

# ---------------- VISUALIZATION ----------------
def draw_boxes_cv2(img, boxes, labels=None, scores=None, color=(0, 255, 0), thickness=2):
    img = img.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if labels is not None:
            text = f"{labels[i]}"
            if scores is not None:
                text += f":{scores[i]:.2f}"
            cv2.putText(
                img,
                text,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )
    return img


def save_visualizations(image_path, single_pred, double_pred, gt):
    img = cv2.imread(image_path)
    if img is None:
        logging.warning("Unable to read image for visualization: %s", image_path)
        return
    img_name = os.path.basename(image_path)
    # GT - BLUE (BGR = (255,0,0))
    if gt.get("boxes"):
        img_gt = draw_boxes_cv2(img, gt["boxes"], gt.get("labels"), scores=None, color=(255, 0, 0))
        cv2.imwrite(os.path.join(GT_DIR, img_name), img_gt)
    # Single - RED (BGR = (0,0,255))
    if single_pred.get("boxes"):
        img_single = draw_boxes_cv2(img, single_pred["boxes"], single_pred.get("labels"), single_pred.get("scores"), color=(0, 0, 255))
        cv2.imwrite(os.path.join(SINGLE_DIR, img_name), img_single)
    # Double - GREEN (BGR = (0,255,0))
    if double_pred.get("boxes"):
        img_double = draw_boxes_cv2(img, double_pred["boxes"], double_pred.get("labels"), double_pred.get("scores"), color=(0, 255, 0))
        cv2.imwrite(os.path.join(DOUBLE_DIR, img_name), img_double)

# ---------------- IMAGE PROCESSING ----------------
def process_image_optimized(image_path, image_predictions, model_names, use_augment=False):
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
        model = get_model(CUSTOM_MODEL_WEIGHTS, cache_name='custom')
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
        # Save visualizations (single stage = original predictions, double stage = current_predictions after refinement)
        save_visualizations(
            image_path=image_path,
            single_pred={
                "boxes": image_predictions[image_name]["boxes"],
                "scores": image_predictions[image_name]["scores"],
                "labels": image_predictions[image_name]["labels"],
            },
            double_pred=current_predictions,
            gt={
                "boxes": true_boxes,
                "labels": true_labels
            }
        )
        results = {
            'image_path': image_path,
            'refined_predictions': current_predictions,
            'ground_truth': {
                'boxes': true_boxes,
                'labels': true_labels
            }
        }
        return results, total_extra_time
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None, 0.0

# ---------------- IO ----------------
def load_image_predictions(predictions_path, conf_threshold=0.25):
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found at {predictions_path}")
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

# ---------------- COMPARE MODE ----------------
def draw_boxes(img, boxes, color=(0, 255, 0)):
    img = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


def run_inference_simple(model, img_path, conf=CONF_THRESHOLD, iou=NMS_IOU_THRESHOLD):
    # returns list of xyxy boxes
    results = model.predict(img_path, conf=conf, iou=iou, verbose=False)
    if results is None or len(results) == 0:
        return []
    res = results[0]
    if getattr(res, 'boxes', None) is None or len(res.boxes) == 0:
        return []
    return res.boxes.xyxy.cpu().numpy().tolist()


def compare_and_save():
    # load models
    vanilla = get_model(VANILLA_MODEL_WEIGHTS, cache_name='vanilla')
    custom = get_model(CUSTOM_MODEL_WEIGHTS, cache_name='custom')

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    saved = 0
    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        try:
            vanilla_boxes = run_inference_simple(vanilla, img_path)
            custom_boxes = run_inference_simple(custom, img_path)
        except Exception as e:
            logging.error(f"Inference error on {img_path}: {e}")
            continue
        if len(vanilla_boxes) == len(custom_boxes):
            continue  # only save when counts differ
        # draw
        vanilla_vis = draw_boxes(img, vanilla_boxes, (0, 0, 255))   # RED
        custom_vis  = draw_boxes(img, custom_boxes,  (0, 255, 0))   # GREEN
        # side-by-side
        try:
            combined = cv2.hconcat([vanilla_vis, custom_vis])
        except Exception:
            # fallback to resizing to same height
            h = max(vanilla_vis.shape[0], custom_vis.shape[0])
            vanilla_vis = cv2.resize(vanilla_vis, (vanilla_vis.shape[1], h))
            custom_vis = cv2.resize(custom_vis, (custom_vis.shape[1], h))
            combined = cv2.hconcat([vanilla_vis, custom_vis])
        # labels
        h, w = img.shape[:2]
        cv2.putText(combined, f"Vanilla ({len(vanilla_boxes)})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.putText(combined, f"Custom ({len(custom_boxes)})", (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        save_path = os.path.join(COMPARE_OUT, img_name)
        cv2.imwrite(save_path, combined)
        saved += 1
    print(f"Saved {saved} comparison images with differing instance counts to {COMPARE_OUT}")

# ---------------- MAIN (DOUBLE INFERENCE) ----------------
def main_double():
    start_time = time.time()
    logging.info("Loading custom YOLO model for double inference...")
    model = get_model(CUSTOM_MODEL_WEIGHTS, cache_name='custom')
    model_names = getattr(model, 'names', None)
    logging.info("Loading predictions JSON...")
    image_predictions = load_image_predictions(PREDICTIONS_JSON, CONF_THRESHOLD)
    image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                  if os.path.splitext(f)[0] in image_predictions]
    logging.info("Processing %d images with optimized double inference...", len(image_files))
    results = {}
    all_predictions, all_targets = [], []
    total_extra_time = 0.0
    max_workers = min(4, len(image_files)) if len(image_files) > 0 else 1
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
    logging.info(f"mAP@0.5: {metrics['map_50']:.4f}")
    logging.info(f"Precision: {metrics['precision']:.4f}")
    logging.info(f"Recall: {metrics['recall']:.4f}")
    logging.info(f"Processing time: {total_time:.2f} seconds")
    logging.info(f"Average extra inference time per image: {avg_extra_time:.4f} seconds")
    try:
        rel_overhead = (avg_extra_time / (total_time - total_extra_time)) * 100
    except Exception:
        rel_overhead = 0.0
    logging.info(f"Relative overhead ratio: {rel_overhead:.2f}%%")
    return results, metrics

# ---------------- ENTRYPOINT ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combined double-inference and compare script')
    parser.add_argument('--mode', choices=['double', 'compare'], default='compare', help='Run double inference (and save GT/single/double) or compare mode')
    args = parser.parse_args()

    if args.mode == 'compare':
        compare_and_save()
    else:
        results, metrics = main_double()
        print('Done.')
