import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
from collections import defaultdict
import time

# Constants
image_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
label_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"
model_weights = "/kaggle/input/ld70-waid-weight/best (3).pt"
conf_threshold = 0.5
iou_threshold = 0.1

# Initialize model
model = YOLO("yolov8n-LD-P2.yaml")
checkpoint = torch.load(model_weights, map_location="cuda" if torch.cuda.is_available() else "cpu")
state_dict = checkpoint["model"].state_dict() if "model" in checkpoint else checkpoint
model.model.load_state_dict(state_dict, strict=True)

metric = MeanAveragePrecision(class_metrics=True)
inference_times = []

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1, area2 = (box1[2] - box1[0]) * (box1[3] - box1[1]), (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (area1 + area2 - intersection + 1e-6)

def simple_nms(boxes, scores, iou_threshold=0.5):
    boxes, scores = torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32)
    keep = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    return keep.tolist()

def process_image(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    input_tensor = input_tensor.to("cuda" if torch.cuda.is_available() else "cpu")
    return img, input_tensor

def get_ground_truth(label_path, img_width, img_height):
    true_boxes, true_labels = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x1, y1 = (x_center - width / 2) * img_width, (y_center - height / 2) * img_height
                x2, y2 = (x_center + width / 2) * img_width, (y_center + height / 2) * img_height
                true_boxes.append([x1, y1, x2, y2])
                true_labels.append(int(class_id))
    return true_boxes, true_labels

def evaluate_model():
    all_predictions, all_targets = [], []
    for image_name in os.listdir(image_dir):
        image_path, label_path = os.path.join(image_dir, image_name), os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
        img, input_tensor = process_image(image_path)
        true_boxes, true_labels = get_ground_truth(label_path, *img.size)

        start_time = time.time()
        result = model.predict(img, conf=0.25, verbose=False)[0]
        inference_times.append((time.time() - start_time) * 1000)

        predictions = {'boxes': [], 'scores': [], 'labels': []}
        for box in result.boxes:
            predictions['boxes'].append(box.xyxy[0].cpu().numpy().tolist())
            predictions['scores'].append(box.conf.item())
            predictions['labels'].append(int(box.cls.item()))

        keep_indices = simple_nms(predictions['boxes'], predictions['scores'], iou_threshold=0.5)
        filtered_predictions = {k: [v[i] for i in keep_indices] for k, v in predictions.items()}
        preds = [{
            'boxes': torch.tensor(filtered_predictions['boxes']) if filtered_predictions['boxes'] else torch.zeros((0, 4)),
            'scores': torch.tensor(filtered_predictions['scores']) if filtered_predictions['scores'] else torch.zeros(0),
            'labels': torch.tensor(filtered_predictions['labels']) if filtered_predictions['labels'] else torch.zeros(0),
        }]
        targets = [{
            'boxes': torch.tensor(true_boxes) if true_boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(true_labels) if true_labels else torch.zeros(0),
        }]
        metric.update(preds, targets)
        all_predictions.extend(preds)
        all_targets.extend(targets)
    return all_predictions, all_targets

all_predictions, all_targets = evaluate_model()
precision, recall = metric.compute()['map'].item(), metric.compute()['mar'].item()
print(f"Precision: {precision:.4f}\nRecall: {recall:.4f}")
