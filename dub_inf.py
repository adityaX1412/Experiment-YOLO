import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
from torchvision import transforms
from torch import amp

# Updated `timm` imports (if used elsewhere)
from timm.layers import DropPath  # Example, update as per requirement

# Constants
image_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
label_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"
model_weights = "/kaggle/input/yolo-weights/weights/spdn  soap.pt"
conf_threshold = 0.1  # Lower for debugging, adjust as needed
iou_threshold = 0.5

model = YOLO("yolov8-ASF-P2.yaml")

# Load the model checkpoint (not just weights)
checkpoint = torch.load(model_weights,map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Check if `checkpoint` is a full model object
if isinstance(checkpoint, torch.nn.Module):
    print("⚠️ Loaded full model instead of state_dict!")
    state_dict = checkpoint.state_dict()  # Extract weights
elif isinstance(checkpoint, dict) and "model" in checkpoint:
    print("✅ Extracting state_dict from checkpoint dictionary...")
    state_dict = checkpoint["model"].state_dict()  # Extract from wrapped model
elif isinstance(checkpoint, dict):
    print("✅ Using checkpoint as state_dict directly...")
    state_dict = checkpoint  # Directly assign if it's already a state_dict
else:
    raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")

# Load the state dictionary into the model
model.model.load_state_dict(state_dict, strict=False)

print("✅ Model weights loaded successfully!")

# Initialize metrics
metric = MeanAveragePrecision(class_metrics=True)
total_predictions = 0
correct_predictions = 0

# Define Tensor Transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Helper function: Non-Maximum Suppression
def simple_nms(boxes, scores, iou_threshold=0.5):
    """Applies Non-Maximum Suppression (NMS) to remove overlapping boxes."""
    boxes = boxes.clone().detach()  # Avoid in-place modifications
    scores = scores.clone().detach()

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
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Initial YOLO Predictions
    with torch.no_grad():
        results = model.predict(img_tensor, conf=conf_threshold,verbose = False)
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
    predictions = {
        "boxes": [],
        "scores": [],
        "labels": []
    }

    for box in result.boxes:
        predictions["boxes"].append(box.xyxy[0].cpu().numpy().tolist())
        predictions["scores"].append(box.conf.item())
        predictions["labels"].append(int(box.cls.item()))

    # Apply NMS
    if predictions["boxes"]:
        boxes_tensor = torch.tensor(predictions["boxes"]).clone().detach()
        scores_tensor = torch.tensor(predictions["scores"]).clone().detach()
        labels_tensor = torch.tensor(predictions["labels"]).clone().detach()
        keep_indices = simple_nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

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
