import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision

# Constants
image_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
label_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"
model_weights = "/kaggle/input/yolo-weights/weights/spdld.pt"
conf_threshold = 0.7
iou_threshold = 0.5

# Initialize YOLO model
model = YOLO("yolov8n-LD-P2.yaml")

# Load model checkpoint
checkpoint = torch.load(model_weights, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
if isinstance(checkpoint, torch.nn.Module):
    state_dict = checkpoint.state_dict()
elif isinstance(checkpoint, dict) and "model" in checkpoint:
    state_dict = checkpoint["model"].state_dict()
elif isinstance(checkpoint, dict):
    state_dict = checkpoint
else:
    raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")

model.model.load_state_dict(state_dict, strict=False)
print("✅ Model weights loaded successfully!")

# Initialize metrics
metric = MeanAveragePrecision(class_metrics=True)
total_predictions = 0
correct_predictions = 0

# ✅ Resize & Pad Function (Newly Added)
def resize_and_pad_image(img, target_size=(640, 640), padding_color=(114, 114, 114)):
    """Resizes an image while preserving aspect ratio and pads it to the target size."""
    original_w, original_h = img.size
    target_w, target_h = target_size

    # Compute scaling ratio to maintain aspect ratio
    ratio = min(target_w / original_w, target_h / original_h)
    new_size = (int(original_w * ratio), int(original_h * ratio))

    # Resize image while preserving aspect ratio
    resized_img = img.resize(new_size, Image.BILINEAR)

    # Create a new image with the target size and paste the resized image onto it
    padded_img = Image.new("RGB", target_size, padding_color)
    pad_x = (target_w - new_size[0]) // 2
    pad_y = (target_h - new_size[1]) // 2
    padded_img.paste(resized_img, (pad_x, pad_y))

    return padded_img, (pad_x, pad_y), ratio

# Helper function: IoU Calculation
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (area1 + area2 - intersection + 1e-6)

# Iterate through images
for image_path in os.listdir(image_dir):
    img = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
    img_width, img_height = img.size

    # ✅ Apply resizing and padding
    padded_img, (pad_x, pad_y), scale_ratio = resize_and_pad_image(img)

    # YOLO Prediction
    with torch.no_grad():
        results = model.predict(padded_img, conf=conf_threshold, verbose=False)
    result = results[0]

    # Load Ground Truth Labels
    true_boxes, true_labels = [], []
    label_path = os.path.join(label_dir, os.path.splitext(image_path)[0] + ".txt")
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x1 = (x_center - width / 2) * img_width
                y1 = (y_center - height / 2) * img_height
                x2 = (x_center + width / 2) * img_width
                y2 = (y_center + height / 2) * img_height
                true_boxes.append([x1, y1, x2, y2])
                true_labels.append(int(class_id))

    # Process Model Predictions
    predictions = {"boxes": [], "scores": [], "labels": []}

    for box in result.boxes:
        pred_box = box.xyxy[0].cpu().numpy().tolist()
        # ✅ Convert YOLO predictions back to original image scale
        pred_box[0] = (pred_box[0] - pad_x) / scale_ratio
        pred_box[1] = (pred_box[1] - pad_y) / scale_ratio
        pred_box[2] = (pred_box[2] - pad_x) / scale_ratio
        pred_box[3] = (pred_box[3] - pad_y) / scale_ratio

        predictions["boxes"].append(pred_box)
        predictions["scores"].append(box.conf.item())
        predictions["labels"].append(int(box.cls.item()))

    # Apply NMS
    if predictions["boxes"]:
        boxes_tensor = torch.tensor(predictions["boxes"]).clone().detach()
        scores_tensor = torch.tensor(predictions["scores"]).clone().detach()
        labels_tensor = torch.tensor(predictions["labels"]).clone().detach()
        keep_indices = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou_threshold)

        filtered_predictions = {
            "boxes": boxes_tensor[keep_indices].tolist(),
            "scores": scores_tensor[keep_indices].tolist(),
            "labels": labels_tensor[keep_indices].tolist()
        }
    else:
        filtered_predictions = {"boxes": [], "scores": [], "labels": []}

    # Calculate Accuracy Metrics
    pred_boxes = np.array(filtered_predictions["boxes"])
    pred_labels = np.array(filtered_predictions["labels"])
    true_boxes = np.array(true_boxes)
    true_labels = np.array(true_labels)

    img_correct = 0
    used_truth_indices = []

    for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        total_predictions += 1
        matching_truths = np.where(true_labels == pred_label)[0]

        best_iou = 0
        best_truth_idx = -1
        for truth_idx in matching_truths:
            if truth_idx in used_truth_indices:
                continue
            iou = calculate_iou(pred_box, true_boxes[truth_idx])
            if iou > best_iou:
                best_iou = iou
                best_truth_idx = truth_idx

        if best_iou >= iou_threshold and best_truth_idx != -1:
            correct_predictions += 1
            img_correct += 1
            used_truth_indices.append(best_truth_idx)

    # Update Mean Average Precision
    preds = [{"boxes": torch.tensor(filtered_predictions["boxes"]), "scores": torch.tensor(filtered_predictions["scores"]), "labels": torch.tensor(filtered_predictions["labels"])}]
    targets = [{"boxes": torch.tensor(true_boxes), "labels": torch.tensor(true_labels)}]

    metric.update(preds, targets)

# Compute and Print Final Metrics
final_metrics = metric.compute()
print(f"mAP@0.5: {final_metrics['map_50']:.4f}")
print(f"Precision: {correct_predictions / (total_predictions + 1e-7):.4f}")
print(f"Recall: {final_metrics['mar_100'].mean():.4f}")
print(f"Total Correct Predictions: {correct_predictions}")
print(f"Total Predictions Made: {total_predictions}")

