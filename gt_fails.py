from ultralytics import YOLO
import os
import glob
import numpy as np

def load_labels(label_file):
    """Reads YOLO-format labels and returns bbox in xyxy format."""
    if not os.path.exists(label_file):
        return []

    boxes = []
    with open(label_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            # YOLO format → xyxy (scaled later once image is loaded)
            boxes.append([cls, xc, yc, w, h])
    return boxes

def yolo_to_xyxy(box, img_w, img_h):
    """Convert YOLO (xc,yc,w,h normalized) → absolute xyxy box."""
    _, xc, yc, w, h = box
    xc *= img_w
    yc *= img_h
    w  *= img_w
    h  *= img_h

    x1 = xc - w/2
    y1 = yc - h/2
    x2 = xc + w/2
    y2 = yc + h/2
    return np.array([x1, y1, x2, y2])

def iou(box1, box2):
    """Compute IoU between two xyxy boxes."""
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area1 = max(0, (x2 - x1) * (y2 - y1))
    area2 = max(0, (a2 - a1) * (b2 - b1))

    union = area1 + area2 - inter_area + 1e-6
    return inter_area / union

def count_false_positives(model_path, images_dir, labels_dir, iou_thresh=0.5):
    model = YOLO(model_path)

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    total_fp = 0

    for img_path in image_files:
        img = model.predict(img_path, verbose=False)[0]   # run inference
        preds = img.boxes.xyxy.cpu().numpy()              # predicted boxes (xyxy)
        pred_classes = img.boxes.cls.cpu().numpy()

        # Load GT
        img_name = os.path.basename(img_path).split(".")[0]
        label_file = os.path.join(labels_dir, img_name + ".txt")
        gt_boxes_yolo = load_labels(label_file)

        # Convert GT boxes using image shape
        img_h, img_w = img.orig_shape
        gt_boxes = []
        for g in gt_boxes_yolo:
            gt_boxes.append(yolo_to_xyxy(g, img_w, img_h))
        gt_boxes = np.array(gt_boxes)

        # Track matches
        matched_gt = set()

        for i, pred_box in enumerate(preds):
            found_match = False
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                if iou(pred_box, gt_box) >= iou_thresh:
                    matched_gt.add(j)
                    found_match = True
                    break

            # If this detection did not match ANY gt → FALSE POSITIVE
            if not found_match:
                total_fp += 1

    print(f"\nTotal detections made by the model but not in labels (FP): {total_fp}")
    return total_fp


# -------------------
# Example usage
# -------------------

model_path = "/kaggle/input/yolo-weights/weights/spdp2p2.pt"
images_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
labels_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"

diff = count_false_positives(model_path, images_dir, labels_dir)
