from ultralytics import YOLO
import os
import glob
import numpy as np

CONF_THRESH = 0.25
IOU_THRESH = 0.5

def load_labels(label_file):
    if not os.path.exists(label_file):
        return []
    boxes = []
    for line in open(label_file):
        cls, xc, yc, w, h = map(float, line.split())
        boxes.append([int(cls), xc, yc, w, h])
    return boxes

def yolo_to_xyxy(box, w, h):
    cls, xc, yc, bw, bh = box
    xc *= w; yc *= h; bw *= w; bh *= h
    x1 = xc - bw/2; y1 = yc - bh/2
    x2 = xc + bw/2; y2 = yc + bh/2
    return np.array([x1, y1, x2, y2]), cls

def iou(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    inter_x1, inter_y1 = max(x1, x3), max(y1, y3)
    inter_x2, inter_y2 = min(x2, x4), min(y2, y3)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, (x2 - x1)*(y2 - y1))
    area_b = max(0, (x4 - x3)*(y4 - y3))
    return inter / (area_a + area_b - inter + 1e-6)

def count_fp_percentage(model_path, images_dir, labels_dir):
    model = YOLO(model_path)
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

    total_fp = 0
    total_tp = 0

    for img_path in image_files:

        result = model(img_path, verbose=False)[0]
        img_h, img_w = result.orig_shape

        preds = result.boxes
        preds_xyxy = preds.xyxy.cpu().numpy()
        preds_cls = preds.cls.cpu().numpy()
        preds_conf = preds.conf.cpu().numpy()

        # filter by confidence
        keep = preds_conf >= CONF_THRESH
        preds_xyxy = preds_xyxy[keep]
        preds_cls = preds_cls[keep]

        # Load GT
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(labels_dir, name + ".txt")
        gt_raw = load_labels(lbl_path)

        gt_boxes = []
        for g in gt_raw:
            xyxy, cls = yolo_to_xyxy(g, img_w, img_h)
            gt_boxes.append((xyxy, cls))

        used_gt = set()

        # Match predictions to GT
        for pbox, pcls in zip(preds_xyxy, preds_cls):
            matched = False

            for j, (gtbox, gtcls) in enumerate(gt_boxes):
                if j in used_gt:
                    continue
                if pcls != gtcls:  # class must match
                    continue
                if iou(pbox, gtbox) >= IOU_THRESH:
                    matched = True
                    used_gt.add(j)
                    break

            if matched:
                total_tp += 1
            else:
                total_fp += 1

    # compute FP Percentage
    denom = total_tp + total_fp
    fp_percentage = (total_fp / denom * 100) if denom > 0 else 0

    print(f"True Positives (TP): {total_tp}")
    print(f"False Positives (FP): {total_fp}")
    print(f"FP Percentage: {fp_percentage:.2f}%")

    return total_tp, total_fp, fp_percentage


model_path = "/kaggle/input/yolo-weights/weights/spdp2p2.pt"
images_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
labels_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"

diff = count_fp_percentage(model_path, images_dir, labels_dir)
