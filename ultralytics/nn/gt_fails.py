from ultralytics import YOLO
import os
import glob

def load_labels(label_file):
    """Loads YOLO-format labels (class, x_center, y_center, w, h)."""
    if not os.path.exists(label_file):
        return []
    with open(label_file, "r") as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        bbox = list(map(float, parts[1:5]))
        labels.append((cls, bbox))
    return labels

def compute_true_negative(gt_labels, pred_boxes):
    """
    True Negative (per image):
      TN = 1 if (no GT objects AND no model predictions)
      else 0
    """
    if len(gt_labels) == 0 and len(pred_boxes) == 0:
        return 1
    return 0

def evaluate_tn_difference(model_path, images_dir, labels_dir):
    model = YOLO(model_path)

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    tn_label_total = 0
    tn_pred_total = 0

    for img_path in image_files:
        img_name = os.path.basename(img_path).replace(".jpg", "")
        label_file = os.path.join(labels_dir, img_name + ".txt")

        # load GT labels
        gt = load_labels(label_file)

        # run prediction
        results = model(img_path, verbose=False)[0]
        preds = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []

        # compute TN (labels)
        tn_label = 1 if len(gt) == 0 else 0

        # compute TN (predictions)
        tn_pred = 1 if len(preds) == 0 else 0

        tn_label_total += tn_label
        tn_pred_total += tn_pred

    diff = tn_pred_total - tn_label_total

    print(f"Total True Negatives from labels      : {tn_label_total}")
    print(f"Total True Negatives from predictions : {tn_pred_total}")
    print(f"Difference (Pred - Label)             : {diff}")

    return diff


# -------------------
# Example usage
# -------------------

model_path = "/kaggle/input/yolo-weights/weights/spdp2p2.pt"
images_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
labels_dir = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"

diff = evaluate_tn_difference(model_path, images_dir, labels_dir)
