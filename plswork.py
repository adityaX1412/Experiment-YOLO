import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
import json
from collections import defaultdict
import time
import logging
from dub_inf_utils import *
import matplotlib.pyplot as plt
import cv2

IMAGE_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
LABEL_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"
DATA_YAML = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml"
MODEL_WEIGHTS = "/kaggle/input/yolo-weights/weights/spdp2p2.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5 
NMS_IOU_THRESHOLD = 0.45
OUTPUT_DIR = "/kaggle/working/visualizations"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load class names from yaml
import yaml
with open(DATA_YAML, 'r') as f:
    data_yaml = yaml.safe_load(f)
    class_names = data_yaml.get('names', {})

model = YOLO(MODEL_WEIGHTS)

predictions_path = "/kaggle/input/json-files/spdp2p2.json"
if not os.path.exists(predictions_path):
    raise FileNotFoundError(f"❌ Predictions file not found at {predictions_path}")

with open(predictions_path, "r") as f:
    val_predictions = json.load(f)

# Create a mapping of image_id to all its predictions
image_to_predictions = defaultdict(list)
for pred in val_predictions:
    image_to_predictions[pred["image_id"]].append(pred)

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

def draw_boxes(image, boxes, labels=None, scores=None, color=(255, 0, 0), thickness=2, label_type="gt"):
    """
    Draw bounding boxes on an image
    
    Args:
        image: PIL Image object
        boxes: List of [x1, y1, x2, y2] coordinates
        labels: List of class labels
        scores: List of confidence scores
        color: RGB tuple for box color
        thickness: Line thickness
        label_type: String to identify the type of box (gt, initial, refined)
    
    Returns:
        PIL Image with boxes drawn
    """
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=thickness)
        
        # Draw label if provided
        if labels is not None and scores is not None:
            label_text = f"{label_type}: {class_names.get(labels[i], labels[i])}"
            if scores is not None:
                label_text += f" {scores[i]:.2f}"
            
            # Background for text
            text_width, text_height = draw.textbbox((0, 0), label_text, font=font)[2:]
            draw.rectangle([(x1, y1 - text_height - 4), (x1 + text_width + 4, y1)], fill=color)
            draw.text((x1 + 2, y1 - text_height - 2), label_text, fill=(255, 255, 255), font=font)
    
    return image

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
            combined_score = (conf * 0.5) + (current_iou * 0.5)  # Weighted combination
            
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
processed_count = 0
for image_path in os.listdir(IMAGE_DIR):
    img = Image.open(os.path.join(IMAGE_DIR, image_path)).convert("RGB")
    img_width, img_height = img.size
    
    # Create a copy for visualization
    vis_image = img.copy()
    
    # Load ground truth labels
    true_boxes, true_labels = [], []
    label_path = os.path.join(LABEL_DIR, os.path.splitext(image_path)[0] + '.txt')
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
    
    # Draw ground truth boxes (green)
    if true_boxes:
        vis_image = draw_boxes(vis_image, true_boxes, true_labels, None, color=(0, 255, 0), label_type="GT")

    # Get predictions for current image
    image_name = os.path.splitext(image_path)[0]
    if image_name not in image_predictions:
        continue
        
    current_predictions = image_predictions[image_name]
    
    # Draw initial prediction boxes (red)
    initial_boxes = current_predictions['boxes'].copy()
    initial_labels = current_predictions['labels'].copy()
    initial_scores = current_predictions['scores'].copy()
    
    vis_image = draw_boxes(
        vis_image, 
        initial_boxes, 
        initial_labels, 
        initial_scores, 
        color=(255, 0, 0), 
        label_type="Initial"
    )
    
    # Process each prediction for potential refinement
    replacement_candidates = []
    refined_boxes = []
    refined_labels = []
    refined_scores = []
    
    # Iterate through predictions using index
    for idx in range(len(current_predictions['scores'])):
        if CONF_THRESHOLD <= current_predictions['scores'][idx] < 0.5:  # Compare single values
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
            
            if refined is not None:  # Check for None explicitly
                replacement_candidates.append({
                    'idx': idx,
                    'bbox': refined['bbox'],
                    'score': refined['score'],
                    'label': refined['category_id']
                })
                
                # Store refined boxes for visualization
                refined_boxes.append(refined['bbox'])
                refined_labels.append(refined['category_id'])
                refined_scores.append(refined['score'])
    
    # Draw refined prediction boxes (blue)
    if refined_boxes:
        vis_image = draw_boxes(
            vis_image, 
            refined_boxes, 
            refined_labels, 
            refined_scores, 
            color=(0, 0, 255), 
            label_type="Refined"
        )
    
    # Apply replacements
    for candidate in replacement_candidates:
        i = candidate['idx']
        current_predictions['boxes'][i] = candidate['bbox']
        current_predictions['scores'][i] = candidate['score']
        current_predictions['labels'][i] = candidate['label']
    
    # Only apply NMS if there are predictions
    if current_predictions['boxes'] and current_predictions['scores'] and current_predictions['labels']:
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

    # Convert boxes and labels to tensors for metrics
    if current_predictions['boxes'] and true_boxes:  # Only add if there are predictions and ground truth
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
    
    # Add legend to image
    legend_image = Image.new('RGB', (200, 100), color=(255, 255, 255))
    legend_draw = ImageDraw.Draw(legend_image)
    
    # Draw legend items
    legend_draw.rectangle([(10, 10), (30, 30)], outline=(0, 255, 0), width=2)
    legend_draw.text((40, 15), "Ground Truth", fill=(0, 0, 0))
    
    legend_draw.rectangle([(10, 40), (30, 60)], outline=(255, 0, 0), width=2)
    legend_draw.text((40, 45), "Initial Prediction", fill=(0, 0, 0))
    
    legend_draw.rectangle([(10, 70), (30, 90)], outline=(0, 0, 255), width=2)
    legend_draw.text((40, 75), "Refined Prediction", fill=(0, 0, 0))
    
    # Paste legend onto visualization
    vis_image.paste(legend_image, (img_width - 210, 10))
    
    # Save visualization
    output_path = os.path.join(OUTPUT_DIR, f"{image_name}_visualization.jpg")
    vis_image.save(output_path)
    
    processed_count += 1
    if processed_count % 10 == 0:
        print(f"Processed {processed_count} images")

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
print(f"\nVisualization images saved to: {OUTPUT_DIR}")

# Create a summary visualization of results
plt.figure(figsize=(12, 6))

# Plot mAP and precision/recall
plt.subplot(1, 2, 1)
plt.bar(['mAP@0.5', 'Precision', 'Recall'], [map50, precision, recall])
plt.title('Detection Performance')
plt.ylim(0, 1)
for i, v in enumerate([map50, precision, recall]):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center')

# Plot sample class APs if available
if class_aps:
    plt.subplot(1, 2, 2)
    class_indices = list(class_aps.keys())
    class_values = list(class_aps.values())
    
    # Show top 5 classes by AP if more than 5 classes
    if len(class_indices) > 5:
        sorted_indices = np.argsort(class_values)[::-1][:5]
        class_indices = [class_indices[i] for i in sorted_indices]
        class_values = [class_values[i] for i in sorted_indices]
    
    class_names_display = [class_names.get(idx, f"Class {idx}") for idx in class_indices]
    plt.bar(class_names_display, class_values)
    plt.title('Class-wise AP@0.5')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    for i, v in enumerate(class_values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "performance_summary.png"))
plt.close()

print(f"Performance summary saved to: {os.path.join(OUTPUT_DIR, 'performance_summary.png')}")

# Create a visualization of the double inference process for a sample image
def create_double_inference_visualization(image_path, model):
    # Find an image with refinable detections
    img_name = os.path.basename(image_path)
    img_id = os.path.splitext(img_name)[0]
    
    if img_id not in image_predictions:
        return None
        
    # Get initial predictions
    initial_preds = image_predictions[img_id]
    
    # Find a prediction with confidence between threshold and 0.5
    refinable_idx = None
    for idx, score in enumerate(initial_preds['scores']):
        if CONF_THRESHOLD <= score < 0.5:
            refinable_idx = idx
            break
            
    if refinable_idx is None:
        return None
        
    # Get original image
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size
    
    # Create original detection object
    original_detection = {
        'bbox': initial_preds['boxes'][refinable_idx],
        'score': initial_preds['scores'][refinable_idx],
        'category_id': initial_preds['labels'][refinable_idx]
    }
    
    # Extract detection details
    x1, y1, x2, y2 = original_detection['bbox']
    sw = x2 - x1  # width of detection
    sh = y2 - y1  # height of detection
    
    # Calculate adaptive crop region
    cx, cy = (x1 + x2)/2, (y1 + y2)/2  # center of detection
    
    # Calculate desired dimensions for the crop
    scale = min(640/sw, 640/sh)
    desired_width = sw * scale
    desired_height = sh * scale
    
    # Calculate crop boundaries with padding
    pad_factor = 0.2  # 20% padding around detection
    new_x1 = max(0, int(cx - (desired_width * (1 + pad_factor))/2))
    new_y1 = max(0, int(cy - (desired_height * (1 + pad_factor))/2))
    new_x2 = min(img_width, int(cx + (desired_width * (1 + pad_factor))/2))
    new_y2 = min(img_height, int(cy + (desired_height * (1 + pad_factor))/2))
    
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
    
    # Perform second inference
    with torch.no_grad():
        new_results = model.predict(padded_img, verbose=False, augment=True)
    
    # Create visualization of the process
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image with initial detection
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)
    axs[0, 0].imshow(np.array(img_copy))
    axs[0, 0].set_title(f"Original Image with Initial Detection\nClass: {class_names.get(original_detection['category_id'], 'Unknown')}, Score: {original_detection['score']:.3f}")
    axs[0, 0].axis('off')
    
    # Show crop region
    img_crop = img.copy()
    draw = ImageDraw.Draw(img_crop)
    draw.rectangle([(new_x1, new_y1), (new_x2, new_y2)], outline=(0, 255, 0), width=2)
    draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)
    axs[0, 1].imshow(np.array(img_crop))
    axs[0, 1].set_title("Crop Region (Green) with Original Detection (Red)")
    axs[0, 1].axis('off')
    
    # Show cropped and resized image
    axs[1, 0].imshow(np.array(padded_img))
    axs[1, 0].set_title(f"Processed Input for Second Inference\n(Cropped, Resized, and Padded to 640×640)")
    axs[1, 0].axis('off')
    
    # Run second inference and show results
    result_img = padded_img.copy()
    draw = ImageDraw.Draw(result_img)
    
    if len(new_results[0].boxes) > 0:
        boxes = new_results[0].boxes.xyxy.cpu().numpy()
        confs = new_results[0].boxes.conf.cpu().numpy()
        labels = new_results[0].boxes.cls.cpu().numpy().astype(int)
        
        for box, score, label in zip(boxes, confs, labels):
            x1, y1, x2, y2 = box
            draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 0, 255), width=2)
            draw.text((x1, y1 - 10), f"{class_names.get(label, 'Unknown')}: {score:.2f}", fill=(0, 0, 255))
    
    axs[1, 1].imshow(np.array(result_img))
    axs[1, 1].set_title("Second Inference Results")
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "double_inference_example.png"))
    plt.close()
    
    return os.path.join(OUTPUT_DIR, "double_inference_example.png")

# Create a sample visualization of the double inference process
sample_image_path = None
for img_path in os.listdir(IMAGE_DIR):
    full_path = os.path.join(IMAGE_DIR, img_path)
    sample_image_path = full_path
    visualization_path = create_double_inference_visualization(full_path, model)
    if visualization_path:
        print(f"Double inference example visualization saved to: {visualization_path}")
        break

print("Processing complete.")
