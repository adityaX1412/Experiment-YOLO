# Run after you've installed ultralytics, pycocotools, pillow, etc.
import os
import glob
import json
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO

# ---------- Utility: read data.yaml safely ----------
def read_yaml(yaml_path):
    import yaml
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

# ---------- Convert YOLO label folder -> COCO JSON ----------
def yolo_to_coco(images_dir, labels_dir, classes, out_json_path, start_ann_id=1):
    """
    images_dir: folder with images
    labels_dir: folder with .txt YOLO labels (same stem as images)
    classes: list of class names (index -> name)
    out_json_path: path to save COCO JSON
    """
    images = []
    annotations = []
    categories = []
    for cid, cname in enumerate(classes):
        categories.append({"id": cid, "name": cname, "supercategory": "none"})
    ann_id = start_ann_id
    img_id = 1
    img_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_paths = [p for p in img_paths if Path(p).suffix.lower() in IMAGE_EXTS]

    for p in tqdm(img_paths, desc="images"):
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:
            # skip unreadable images
            print("Skipping image (cannot open):", p, e)
            continue

        img_info = {"file_name": os.path.relpath(p), "height": h, "width": w, "id": img_id}
        images.append(img_info)
        # find label file with same stem
        stem = Path(p).stem
        label_file = os.path.join(labels_dir, stem + ".txt")
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
                # convert normalized YOLO xywh to absolute [x_min, y_min, width, height]
                x_min = (x_center - bw/2.0) * w
                y_min = (y_center - bh/2.0) * h
                width = bw * w
                height = bh * h
                # clamp
                x_min = max(0.0, x_min)
                y_min = max(0.0, y_min)
                width = max(0.0, min(width, w - x_min))
                height = max(0.0, min(height, h - y_min))
                area = width * height
                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls,
                    "bbox": [x_min, y_min, width, height],
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": []
                }
                annotations.append(ann)
                ann_id += 1
        img_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    with open(out_json_path, 'w') as f:
        json.dump(coco, f)
    return out_json_path

# ---------- Helper: find a COCO JSON in a dataset folder ----------
def find_coco_json_in_folder(folder):
    # common filenames
    candidates = [
        "instances_val2017.json", "instances_val.json", "annotations.json",
        "val_annotations.json", "annotations/instances_val.json"
    ]
    # look for any json in folder tree that contains "annotations" or "instances" in name
    for c in candidates:
        p = os.path.join(folder, c)
        if os.path.exists(p):
            return p
    # fallback: search for any json file that looks like coco (has "images" and "annotations" keys)
    for p in glob.glob(os.path.join(folder, "**", "*.json"), recursive=True):
        try:
            with open(p, 'r') as f:
                j = json.load(f)
            if isinstance(j, dict) and "images" in j and "annotations" in j:
                return p
        except Exception:
            continue
    return None

# ---------- Main: get a COCO GT json for a data.yaml val field ----------
def get_coco_gt(val_images_dir, data_yaml, out_dir="/kaggle/working/converted_coco"):
    """
    val_images_dir: path like .../images/valid
    data_yaml: parsed data.yaml dict (for class names)
    """
    os.makedirs(out_dir, exist_ok=True)

    # sanity
    if not os.path.isdir(val_images_dir):
        raise FileNotFoundError(f"val path is not a directory: {val_images_dir}")

    # 1️⃣ Check if COCO json already exists somewhere nearby
    found = find_coco_json_in_folder(os.path.dirname(val_images_dir))
    if found:
        print("Found existing COCO JSON:", found)
        return found

    # 2️⃣ Infer labels/valid from images/valid
    images_dir = val_images_dir
    labels_dir = images_dir.replace("/images/", "/labels/")

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(
            f"Expected YOLO labels at {labels_dir} but folder does not exist"
        )

    # 3️⃣ Get class names from data.yaml
    names = data_yaml.get("names")
    if names is None:
        raise ValueError("data.yaml must contain `names:` for YOLO→COCO conversion")

    if isinstance(names, dict):
        # YOLOv8 sometimes stores names as dict {0: 'class'}
        classes = [names[k] for k in sorted(names.keys())]
    else:
        classes = list(names)

    out_json = os.path.join(
        out_dir,
        os.path.basename(images_dir.rstrip("/")) + "_coco.json"
    )

    print("Converting YOLO labels → COCO JSON")
    print(" images:", images_dir)
    print(" labels:", labels_dir)
    print(" output:", out_json)

    return yolo_to_coco(
        images_dir=images_dir,
        labels_dir=labels_dir,
        classes=classes,
        out_json_path=out_json
    )


# ---------- Example integration with your existing code ----------
# Suppose WAID_DATA_YAML and BUCK_DATA_YAML are defined (paths to their data.yaml)
# Load each YAML and get proper GT JSON (or convert)
WAID_DATA_YAML = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml"
BUCK_DATA_YAML = "/kaggle/input/bucktales-patched/bucktales_patched/dtc2023.yaml"

# Parse YAMLs
waid_yaml = read_yaml(WAID_DATA_YAML)
buck_yaml = read_yaml(BUCK_DATA_YAML)

# These MUST point to images/valid (which they already do in your case)
waid_val_images = waid_yaml["val"]
buck_val_images = buck_yaml["val"]

# Resolve GT COCO JSONs (auto-convert YOLO labels)
gt_waid_json = get_coco_gt(waid_val_images, waid_yaml)
gt_buck_json = get_coco_gt(buck_val_images, buck_yaml)

print("GT WAID JSON:", gt_waid_json)
print("GT BUCK JSON:", gt_buck_json)
