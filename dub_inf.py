import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from torchmetrics.detection import MeanAveragePrecision

# Constants
image_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
label_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"
model_weights = "/kaggle/input/yolo-weights/weights/spdld.pt"
conf_threshold = 0.1  # 🔹 Lowered initial threshold
iou_threshold = 0.5  # 🔹 Increased NMS IoU threshold

# Load YOLO model
model = YOLO("yolov8n-LD-P2.yaml")
# Load checkpoint (full model or state_dict)
#torch.serialization.add_safe_globals([DetectionModel])
checkpoint = torch.load(model_weights, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=True)

# Extract the state dictionary correctly
# if isinstance(checkpoint, torch.nn.Module):
#     print("⚠️ Loaded full model instead of state_dict! Extracting weights...")
#     state_dict = checkpoint.state_dict()  # Extract weights from full model
# elif isinstance(checkpoint, dict) and "model" in checkpoint:
#     print("✅ Extracting state_dict from checkpoint dictionary...")
#     state_dict = checkpoint["model"].state_dict()  # Extract wrapped model weights
# elif isinstance(checkpoint, dict):
#     print("✅ Using checkpoint as state_dict directly...")
#     state_dict = checkpoint  # Directly assign if it's already a state_dict
# else:
#     raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")

# Load extracted state dictionary into the model
model.model.load_state_dict(state_dict, strict=True)
print("✅ Model weights loaded successfully!")

# Initialize metrics
metric = MeanAveragePrecision(class_metrics=True)
total_predictions = 0
correct_predictions = 0

# 🔹 Resize & Pad Function
def resize_and_pad_image(img, target_size=(640, 640), padding_color=(114, 114, 114)):
    """Resizes an image while preserving aspect ratio and pads it to the target size."""
    original_w, original_h = img.size
    target_w, target_h = target_size
    ratio = min(target_w / original_w, target_h / original_h)
    new_size = (int(original_w * ratio), int(original_h * ratio))
    
    resized_img = img.resize(new_size, Image.BILINEAR)
    padded_img = Image.new("RGB", target_size, padding_color)
    pad_x = (target_w - new_size[0]) // 2
    pad_y = (target_h - new_size[1]) // 2
    padded_img.paste(resized_img, (pad_x, pad_y))

    return padded_img, (pad_x, pad_y), ratio

# 🔹 Scale Bounding Boxes Function
def scale_boxes(boxes, pad_x, pad_y, scale_ratio):
    """Scales bounding boxes back to the original image size after padding and resizing."""
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale_ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale_ratio
    return boxes

# 🔹 IoU Calculation Function
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (area1 + area2 - intersection + 1e-6)

# 🔹 Non-Maximum Suppression (NMS)
def simple_nms(boxes, scores, iou_threshold=0.5):
    """Applies Non-Maximum Suppression (NMS) to remove overlapping boxes."""
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    sorted_indices = torch.argsort(scores, descending=True)
    keep = []

    while sorted_indices.numel() > 0:
        current_idx = sorted_indices[0]
        keep.append(current_idx.item())
        if sorted_indices.numel() == 1:
            break

        current_box = boxes[current_idx].unsqueeze(0)
        remaining_boxes = boxes[sorted_indices[1:]]

        x1 = torch.max(current_box[:, 0], remaining_boxes[:, 0])
        y1 = torch.max(current_box[:, 1], remaining_boxes[:, 1])
        x2 = torch.min(current_box[:, 2], remaining_boxes[:, 2])
        y2 = torch.min(current_box[:, 3], remaining_boxes[:, 3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area_current = (current_box[:, 2] - current_box[:, 0]) * (current_box[:, 3] - current_box[:, 1])
        area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = area_current + area_remaining - intersection

        iou = intersection / (union + 1e-6)
        mask = iou <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]

    return keep

# Iterate through images
for image_path in os.listdir(image_dir):
    img = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
    img_width, img_height = img.size

    # 🔹 Resize & Pad Image
    padded_img, (pad_x, pad_y), scale_ratio = resize_and_pad_image(img)

    # 🔹 YOLO Initial Predictions (Low Confidence Threshold)
    with torch.no_grad():
        results = model.predict(padded_img, conf=conf_threshold, verbose=False)
    result = results[0]

    # 🔹 Load Ground Truth Labels
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

    # 🔹 Process Model Predictions
    predictions = {"boxes": [], "scores": [], "labels": []}
    for box in result.boxes:
        pred_box = scale_boxes(box.xyxy.cpu().numpy(), pad_x, pad_y, scale_ratio)
        predictions["boxes"].append(pred_box.tolist())
        predictions["scores"].append(box.conf.item())
        predictions["labels"].append(int(box.cls.item()))

    # 🔹 Apply NMS
    if predictions["boxes"]:
        keep_indices = simple_nms(predictions["boxes"], predictions["scores"], iou_threshold=0.5)
        filtered_predictions = {
            "boxes": [predictions["boxes"][i] for i in keep_indices],
            "scores": [predictions["scores"][i] for i in keep_indices],
            "labels": [predictions["labels"][i] for i in keep_indices],
        }
    else:
        filtered_predictions = {"boxes": [], "scores": [], "labels": []}

    # 🔹 Compute Accuracy Metrics
    total_predictions += len(filtered_predictions["boxes"])
    for pred_box, pred_label in zip(filtered_predictions["boxes"], filtered_predictions["labels"]):
        if any(calculate_iou(pred_box, gt_box) >= iou_threshold for gt_box in true_boxes):
            correct_predictions += 1

    # 🔹 Update Mean Average Precision
    metric.update([{"boxes": torch.tensor(filtered_predictions["boxes"]), "scores": torch.tensor(filtered_predictions["scores"]), "labels": torch.tensor(filtered_predictions["labels"])}],
                  [{"boxes": torch.tensor(true_boxes), "labels": torch.tensor(true_labels)}])

# 🔹 Print Final Metrics
final_metrics = metric.compute()
print(f"mAP@0.5: {final_metrics['map_50']:.4f}")
print(f"Precision: {correct_predictions / (total_predictions + 1e-7):.4f}")
print(f"Recall: {final_metrics['mar_100'].mean():.4f}")
