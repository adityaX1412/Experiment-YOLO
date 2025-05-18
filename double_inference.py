import os
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
import json
from collections import defaultdict
import time
import logging
import concurrent.futures
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("double_inference.log"),
        logging.StreamHandler()
    ]
)

# Constants
IMAGE_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
LABEL_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"
DATA_YAML = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml"
MODEL_WEIGHTS = "/kaggle/input/yolo-weights/weights/spdp2p2.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5 
NMS_IOU_THRESHOLD = 0.45
VISUALIZATION_DIR = '/kaggle/working/visualizations'

# Create output directories
os.makedirs(f'{VISUALIZATION_DIR}/initial', exist_ok=True)
os.makedirs(f'{VISUALIZATION_DIR}/final', exist_ok=True)
os.makedirs(f'{VISUALIZATION_DIR}/gt', exist_ok=True)

# Utility functions
def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    
    Args:
        box1: List [x1, y1, x2, y2]
        box2: List [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    # Ensure boxes have correct format
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
        
    # Calculate intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Check if boxes intersect
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    # Calculate area of intersection
    area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate area of both boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Check for zero areas to avoid division by zero
    if area_box1 <= 0 or area_box2 <= 0:
        return 0.0
    
    # Calculate IoU
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union if area_union > 0 else 0.0
    
    return iou

def calculate_optimal_crop(detection, img_width, img_height, pad_factor=0.2):
    """
    Calculate optimal crop dimensions that preserve aspect ratio
    
    Args:
        detection: Dictionary with bbox [x1, y1, x2, y2]
        img_width: Original image width
        img_height: Original image height
        pad_factor: Padding factor around the detection (default: 0.2)
        
    Returns:
        Dictionary with crop coordinates
    """
    x1, y1, x2, y2 = detection['bbox']
    sw = max(1, x2 - x1)  # Ensure width is at least 1
    sh = max(1, y2 - y1)  # Ensure height is at least 1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # Center of detection
    
    # Calculate aspect ratio preserving dimensions
    aspect_ratio = sw / sh if sh > 0 else 1  # Avoid division by zero
    
    # Calculate padding that preserves aspect ratio
    pad_w = sw * pad_factor
    pad_h = sh * pad_factor
    
    # Calculate crop dimensions
    crop_w = sw + 2 * pad_w
    crop_h = sh + 2 * pad_h
    
    # Ensure crop doesn't exceed image boundaries
    new_x1 = max(0, int(cx - crop_w / 2))
    new_y1 = max(0, int(cy - crop_h / 2))
    new_x2 = min(img_width, int(cx + crop_w / 2))
    new_y2 = min(img_height, int(cy + crop_h / 2))
    
    # Adjust to maintain aspect ratio if crop is clipped
    actual_w = new_x2 - new_x1
    actual_h = new_y2 - new_y1
    
    # If crop dimensions are too small, expand to minimum size
    if actual_w < 10 or actual_h < 10:
        min_size = 32
        new_x1 = max(0, int(cx - min_size / 2))
        new_y1 = max(0, int(cy - min_size / 2))
        new_x2 = min(img_width, int(cx + min_size / 2))
        new_y2 = min(img_height, int(cy + min_size / 2))
    
    return {
        'x1': new_x1,
        'y1': new_y1,
        'x2': new_x2,
        'y2': new_y2
    }

def prepare_cropped_image(img, crop_info):
    """
    Prepare cropped image for inference
    
    Args:
        img: PIL Image
        crop_info: Dictionary with crop coordinates
        
    Returns:
        PIL Image: Processed image ready for inference
    """
    # Crop image
    crop = img.crop((crop_info['x1'], crop_info['y1'], crop_info['x2'], crop_info['y2']))
    
    # Resize maintaining aspect ratio
    original_w, original_h = crop.size
    ratio = min(640 / original_w, 640 / original_h)
    new_size = (int(original_w * ratio), int(original_h * ratio))
    resized = crop.resize(new_size, Image.BILINEAR)
    
    # Create padded image (letterbox)
    padded_img = Image.new("RGB", (640, 640), (114, 114, 114))
    pad_x = (640 - new_size[0]) // 2
    pad_y = (640 - new_size[1]) // 2
    padded_img.paste(resized, (pad_x, pad_y))
    
    return {
        'image': padded_img,
        'original_size': (original_w, original_h),
        'new_size': new_size,
        'pad_x': pad_x,
        'pad_y': pad_y,
        'ratio': ratio
    }

def scale_boxes(boxes, pad_x, pad_y, crop_info, ratio):
    """
    Scale boxes from cropped image back to original image coordinates
    
    Args:
        boxes: Numpy array of boxes [x1, y1, x2, y2]
        pad_x: X padding in letterboxed image
        pad_y: Y padding in letterboxed image
        crop_info: Dictionary with crop coordinates
        ratio: Resize ratio applied to crop
        
    Returns:
        Numpy array: Scaled boxes in original image coordinates
    """
    if boxes.size == 0:
        return np.array([])
    
    # Make a copy to avoid modifying original
    scaled = boxes.copy()
    
    # Adjust for padding
    scaled[:, 0] -= pad_x
    scaled[:, 1] -= pad_y
    scaled[:, 2] -= pad_x
    scaled[:, 3] -= pad_y
    
    # Scale back to crop size
    scaled[:, 0] /= ratio
    scaled[:, 1] /= ratio
    scaled[:, 2] /= ratio
    scaled[:, 3] /= ratio
    
    # Translate to original image coordinates
    scaled[:, 0] += crop_info['x1']
    scaled[:, 1] += crop_info['y1']
    scaled[:, 2] += crop_info['x1']
    scaled[:, 3] += crop_info['y1']
    
    return scaled

def non_max_suppression(boxes, scores, labels, iou_threshold=0.45):
    """
    Apply non-maximum suppression to remove redundant detections
    
    Args:
        boxes: List of boxes [x1, y1, x2, y2]
        scores: List of confidence scores
        labels: List of class labels
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Tuple of filtered boxes, scores, and labels
    """
    # Validate input dimensions
    if not boxes or len(boxes) == 0:
        return [], [], []
    
    # Convert to numpy arrays for processing
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Get argsort of scores (descending)
    idxs = np.argsort(scores)[::-1]
    
    # Initialize array to keep track of picked indices
    pick = []
    
    while len(idxs) > 0:
        # Pick box with highest score
        last = len(idxs) - 1
        i = idxs[0]
        pick.append(i)
        
        # Get boxes to compare with the picked box
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        
        # Compute width and height of intersection
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        # Compute intersection area
        area_intersection = w * h
        
        # Compute area of boxes
        area_box1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_boxes = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        
        # Compute IoU
        union = area_box1 + area_boxes - area_intersection
        iou = area_intersection / union
        
        # Remove boxes with IoU > threshold and same label
        same_label_mask = labels[idxs[1:]] == labels[i]
        overlap_mask = iou > iou_threshold
        remove_mask = same_label_mask & overlap_mask
        
        # Update idxs
        idxs = np.delete(idxs, np.concatenate(([0], np.where(remove_mask)[0] + 1)))
    
    # Return filtered boxes, scores, and labels
    return boxes[pick].tolist(), scores[pick].tolist(), labels[pick].tolist()

def perform_double_inference(image_path, model, original_detection, use_augment=False):
    """
    Perform double inference with consistent augmentation settings
    
    Args:
        image_path: Path to the original image
        model: YOLO model instance
        original_detection: Dictionary with original detection info
        use_augment: Whether to use test-time augmentation
        
    Returns:
        Dictionary with refined detection or None if no improvement
    """
    try:
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_width, img_height = img.size
        
        # Calculate optimal crop region
        crop_info = calculate_optimal_crop(original_detection, img_width, img_height)
        
        # Check if crop region is valid
        if crop_info['x2'] <= crop_info['x1'] or crop_info['y2'] <= crop_info['y1']:
            logging.warning(f"Invalid crop dimensions: {crop_info}")
            return None
        
        # Prepare cropped image
        processed = prepare_cropped_image(img, crop_info)
        
        # Perform inference on cropped region
        with torch.no_grad():
            new_results = model.predict(processed['image'], verbose=False, augment=use_augment)
        
        # Check if any detections were found
        if len(new_results[0].boxes) == 0:
            return None
        
        # Extract detection results
        boxes = new_results[0].boxes.xyxy.cpu().numpy()
        confs = new_results[0].boxes.conf.cpu().numpy()
        labels = new_results[0].boxes.cls.cpu().numpy().astype(int)
        
        # Handle single detection case
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
            confs = np.expand_dims(confs, axis=0)
            labels = np.expand_dims(labels, axis=0)
        
        # Scale boxes back to original image coordinates
        scaled_boxes = scale_boxes(
            boxes,
            processed['pad_x'],
            processed['pad_y'],
            crop_info,
            processed['ratio']
        )
        
        # Process refined boxes
        return process_refined_boxes(scaled_boxes, labels, confs, original_detection, img_width, img_height)
    
    except Exception as e:
        logging.error(f"Error in double inference: {str(e)}")
        return None

def process_refined_boxes(scaled_boxes, labels, confs, original_detection, img_width, img_height):
    """
    Process refined boxes from second inference
    
    Args:
        scaled_boxes: Numpy array of scaled boxes
        labels: Numpy array of class labels
        confs: Numpy array of confidence scores
        original_detection: Dictionary with original detection info
        img_width: Original image width
        img_height: Original image height
        
    Returns:
        Dictionary with best refined detection or None
    """
    # Validate inputs
    if scaled_boxes.size == 0 or len(labels) == 0 or len(confs) == 0:
        return None
    
    # Ensure all arrays have the same length
    if not (len(scaled_boxes) == len(labels) == len(confs)):
        logging.warning(f"Mismatched array lengths: boxes={len(scaled_boxes)}, labels={len(labels)}, confs={len(confs)}")
        return None
    
    original_label = original_detection['category_id']
    original_score = original_detection['score']
    
    # Find best matching detection
    best_match = None
    best_combined = -1
    best_conf = -1
    
    for i, (box, label, conf) in enumerate(zip(scaled_boxes, labels, confs)):
        # Skip if label doesn't match original
        if label != original_label:
            continue
        
        # Skip invalid boxes
        if (box[2] <= box[0]) or (box[3] <= box[1]) or \
           (box[0] < 0) or (box[1] < 0) or \
           (box[2] > img_width) or (box[3] > img_height):
            continue
        
        # Calculate IoU with original detection
        current_iou = calculate_iou(original_detection['bbox'], box)
        
        # Skip if IoU is too low (not the same object)
        if current_iou < 0.25:
            continue
        
        # Calculate combined score (weighted average of confidence and IoU)
        combined_score = (conf * 0.6) + (current_iou * 0.4)
        
        # Update best match if this is better
        if combined_score > best_combined:
            best_combined = combined_score
            best_conf = conf
            best_match = {
                'bbox': box.tolist(),
                'score': float(conf),
                'category_id': int(label)
            }
    
    # Return refined detection only if it improves on original
    return best_match if best_match and best_conf > original_score else None

def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics with validation
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        
    Returns:
        Dictionary with metrics
    """
    # Validate inputs
    if not predictions or not targets:
        return {'map_50': 0, 'precision': 0, 'recall': 0}
    
    # Use consistent calculation method
    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(predictions, targets)
    results = metric.compute()
    
    return {
        'map_50': results['map_50'].item(),
        'precision': results['precision'].item(),
        'recall': results['recall'].item(),
        'class_metrics': {
            f'class_{i}': {
                'ap': ap.item()
            } for i, ap in enumerate(results['map_per_class'])
        }
    }

def visualize_results(image_path, data, model_names):
    """
    Visualize detection results
    
    Args:
        image_path: Path to the original image
        data: Dictionary with detection results
        model_names: Dictionary mapping class IDs to names
    """
    try:
        # Load image
        img = Image.open(image_path).convert("RGB")
        
        # Create three copies
        initial_img = img.copy()
        final_img = img.copy()
        gt_img = img.copy()
        
        # Create drawing objects
        draw_initial = ImageDraw.Draw(initial_img)
        draw_final = ImageDraw.Draw(final_img)
        draw_gt = ImageDraw.Draw(gt_img)
        
        # Draw initial predictions
        for box, score, label in zip(data['initial_predictions']['boxes'], 
                                    data['initial_predictions']['scores'], 
                                    data['initial_predictions']['labels']):
            x1, y1, x2, y2 = box
            # Draw rectangle
            draw_initial.rectangle([x1, y1, x2, y2], outline="red", width=2)
            # Add label and confidence
            label_text = f"{model_names[label]}: {score:.2f}"
            draw_initial.text((x1, y1-10), label_text, fill="red")
        
        # Draw final predictions
        for box, score, label in zip(data['refined_predictions']['boxes'], 
                                    data['refined_predictions']['scores'], 
                                    data['refined_predictions']['labels']):
            x1, y1, x2, y2 = box
            # Draw rectangle
            draw_final.rectangle([x1, y1, x2, y2], outline="green", width=2)
            # Add label and confidence
            label_text = f"{model_names[label]}: {score:.2f}"
            draw_final.text((x1, y1-10), label_text, fill="green")
        
        # Draw ground truth
        for box, label in zip(data['ground_truth']['boxes'], 
                             data['ground_truth']['labels']):
            x1, y1, x2, y2 = box
            draw_gt.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            label_text = f"GT: {model_names[label]}"
            draw_gt.text((x1, y1 - 10), label_text, fill="blue")
        
        # Extract base name
        base_name = Path(image_path).stem
        
        # Save visualizations
        initial_path = f'{VISUALIZATION_DIR}/initial/{base_name}_initial.jpg'
        final_path = f'{VISUALIZATION_DIR}/final/{base_name}_final.jpg'
        gt_path = f'{VISUALIZATION_DIR}/gt/{base_name}_gt.jpg'
        
        initial_img.save(initial_path)
        final_img.save(final_path)
        gt_img.save(gt_path)
        
    except Exception as e:
        logging.error(f"Error visualizing results for {image_path}: {str(e)}")

def load_image_predictions(predictions_path, conf_threshold=0.25):
    """
    Load predictions from JSON file and format them for processing
    
    Args:
        predictions_path: Path to JSON predictions file
        conf_threshold: Confidence threshold for filtering predictions
        
    Returns:
        Dictionary mapping image names to predictions
    """
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"❌ Predictions file not found at {predictions_path}")
    
    # Load predictions from JSON
    with open(predictions_path, "r") as f:
        val_predictions = json.load(f)
    
    # Convert to per-image format
    image_predictions = {}
    for pred in val_predictions:
        image_name = pred["image_id"]
        if image_name not in image_predictions:
            image_predictions[image_name] = {"boxes": [], "scores": [], "labels": []}
        
        # Filter by confidence threshold
        if pred["score"] >= conf_threshold:
            x, y, w, h = pred["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            image_predictions[image_name]["boxes"].append([x1, y1, x2, y2])
            image_predictions[image_name]["scores"].append(pred["score"])
            image_predictions[image_name]["labels"].append(pred["category_id"])
    
    return image_predictions

def load_ground_truth(label_path, img_width, img_height):
    """
    Load ground truth from YOLO format label file
    
    Args:
        label_path: Path to label file
        img_width: Image width
        img_height: Image height
        
    Returns:
        Tuple of boxes and labels
    """
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

def process_image(image_path, model, image_predictions, model_names, use_augment=False):
    """
    Process a single image with double inference
    
    Args:
        image_path: Path to image file
        model: YOLO model instance
        image_predictions: Dictionary with initial predictions
        model_names: Dictionary mapping class IDs to names
        use_augment: Whether to use test-time augmentation
        
    Returns:
        Dictionary with processed results
    """
    try:
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_width, img_height = img.size
        
        # Get image name without extension
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Skip if no predictions for this image
        if image_name not in image_predictions:
            logging.warning(f"No predictions found for {image_name}")
            return None
        
        # Load ground truth
        label_path = os.path.join(LABEL_DIR, f"{image_name}.txt")
        true_boxes, true_labels = load_ground_truth(label_path, img_width, img_height)
        
        # Get current predictions
        current_predictions = image_predictions[image_name]
        
        # Make a copy of initial predictions for visualization
        initial_preds = {
            'boxes': current_predictions['boxes'].copy(),
            'scores': current_predictions['scores'].copy(),
            'labels': current_predictions['labels'].copy()
        }
        
        # Process each prediction for potential refinement
        refined_candidates = []
        
        # For each prediction, consider if it should be refined
        for idx in range(len(current_predictions['scores'])):
            # Now refine all detections with confidence score
            if current_predictions['scores'][idx] >= CONF_THRESHOLD:
                original_detection = {
                    'bbox': current_predictions['boxes'][idx],
                    'score': current_predictions['scores'][idx],
                    'category_id': current_predictions['labels'][idx]
                }
                
                # Perform double inference
                try:
                    refined = perform_double_inference(
                        image_path,
                        model,
                        original_detection,
                        use_augment
                    )
                    
                    if refined is not None:
                        refined_candidates.append({
                            'idx': idx,
                            'bbox': refined['bbox'],
                            'score': refined['score'],
                            'label': refined['category_id']
                        })
                except Exception as e:
                    logging.error(f"Error refining detection {idx} in {image_name}: {str(e)}")
        
        # Apply refinements
        for candidate in refined_candidates:
            i = candidate['idx']
            current_predictions['boxes'][i] = candidate['bbox']
            current_predictions['scores'][i] = candidate['score']
            current_predictions['labels'][i] = candidate['label']
        
        # Apply NMS if there are predictions
        if current_predictions['boxes'] and current_predictions['scores'] and current_predictions['labels']:
            current_predictions['boxes'], current_predictions['scores'], current_predictions['labels'] = \
                non_max_suppression(
                    current_predictions['boxes'],
                    current_predictions['scores'],
                    current_predictions['labels'],
                    NMS_IOU_THRESHOLD
                )
        
        # Prepare results
        results = {
            'image_path': image_path,
            'initial_predictions': initial_preds,
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
        
        return results
    
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None

def main():
    """Main execution function"""
    start_time = time.time()
    
    # Load model
    logging.info("Loading YOLO model...")
    model = YOLO(MODEL_WEIGHTS)
    model_names = model.names
    
    # Load predictions
    logging.info("Loading predictions...")
    predictions_path = "/kaggle/input/json-files/spdp2p2.json"
    image_predictions = load_image_predictions(predictions_path, CONF_THRESHOLD)
    
    # Get list of image files
    image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
                  if os.path.splitext(f)[0] in image_predictions]
    
    logging.info(f"Processing {len(image_files)} images with double inference...")
    
    # Process images
    results = {}
    all_predictions = []
    all_targets = []
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit tasks
        future_to_image = {
            executor.submit(
                process_image,
                image_path,
                model,
                image_predictions,
                model_names,
                True  # use_augment
            ): image_path for image_path in image_files
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                result = future.result()
                if result:
                    results[image_path] = result
                    
                    # Convert to torchmetrics format for evaluation
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
                logging.error(f"Error processing {image_path}: {str(e)}")
    
    # Visualize results
    logging.info("Visualizing results...")
    for image_path, data in results.items():
        visualize_results(image_path, data, model_names)
    
    # Calculate metrics
    logging.info("Calculating metrics...")
    metrics = calculate_metrics(all_predictions, all_targets)
    
    # Print results
    logging.info(f"Results after double inference:")
    logging.info(f"mAP@0.5: {metrics['map_50']:.4f}")
    logging.info(f"Precision: {metrics['precision']:.4f}")
    logging.info(f"Recall: {metrics['recall']:.4f}")
    
    # Print per-class metrics
    for class_id, class_metrics in metrics['class_metrics'].items():
        logging.info(f"{class_id} AP: {class_metrics['ap']:.4f}")
    
    # Calculate statistics
    total_time = time.time() - start_time
    num_images = len(results)
    total_detections = sum(len(data['refined_predictions']['boxes']) for data in results.values())
    
    logging.info(f"Processed {num_images} images with {total_detections} detections in {total_time:.2f} seconds")
    logging.info(f"Average time per image: {total_time/max(1, num_images):.2f} seconds")

if __name__ == "__main__":
    main()