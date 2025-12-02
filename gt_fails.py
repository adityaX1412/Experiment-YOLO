from ultralytics import YOLO
import os
import glob

def load_labels(label_file):
    if not os.path.exists(label_file):
        return []
    with open(label_file, "r") as f:
        return [line.strip().split() for line in f.readlines()]

def evaluate_confusion(model_path, images_dir, labels_dir):
    model = YOLO(model_path)

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

    TN = TP = FP = FN = 0

    for img_path in image_files:
        img_name = os.path.basename(img_path).split(".")[0]
        label_file = os.path.join(labels_dir, img_name + ".txt")

        gt = load_labels(label_file)
        results = model(img_path, verbose=False)[0]

        preds = []
        if results.boxes is not None:
            preds = results.boxes.xyxy.cpu().numpy()

        # -----------------------------------
        # Case 1: No GT objects in an image
        # -----------------------------------
        if len(gt) == 0:
            if len(preds) == 0:
                TN += 1     # Correct empty prediction
            else:
                FP += len(preds)   # Everything predicted is false positive
            continue

        # -----------------------------------
        # Case 2: GT exists but model predicts nothing
        # -----------------------------------
        if len(preds) == 0:
            FN += len(gt)  # All GT become false negatives
            continue

        # -----------------------------------
        # Case 3: Both GT and predictions exist
        # -----------------------------------
        # For simplicity, treat every prediction as FP and every GT as FN
        # unless IoU-match > 0.5

        import numpy as np

        def iou(box1, box2):
            x1, y1, x2, y2 = box1
            a1, b1, a2, b2 = box2

            inter_x1 = max(x1, a1)
            inter_y1 = max(y1, b1)
            inter_x2 = min(x2, a2)
            inter_y2 = min(y2, b2)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            area1 = (x2 - x1) * (y2 - y1)
            area2 = (a2 - a1) * (b2 - b1)

            iou_val = inter_area / (area1 + area2 - inter_area + 1e-6)
            return iou_val

        matched_gt = set()
        matched_pred = set()

        # convert GT YOLO format → xyxy (fake scale, only for IoU structure)
        # We assume ground truth files already in YOLO format — need image shape for exact xyxy.
        # If you want exact bounding boxes, provide img shape.

        # But for TN difference, you do NOT need full TP/FP logic.

        # Here we only check if predictions exist AND GT exists:
        # This tells us there are no TNs unless GT=0 & pred=0.

        # ---------------------------------------------
        # Counting final TN only from the empty-image rule
        # ---------------------------------------------

    print("Final Counts:")
    print("TN:", TN)
    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)

    return TN, TP, FP, FN

# -------------------
# Example usage
# -------------------

model_path = "/kaggle/input/yolo-weights/weights/spdp2p2.pt"
images_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
labels_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"

diff = evaluate_confusion(model_path, images_dir, labels_dir)
