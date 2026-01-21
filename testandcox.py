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
def get_coco_gt(val_field, dataset_root=None, data_yaml=None, out_dir="/kaggle/working/converted_coco"):
    """
    val_field: value of data.yaml['val'] — may be a json path or a directory path
    dataset_root: optional root folder to resolve relative paths
    data_yaml: parsed yaml dict if available (used to get names)
    """
    os.makedirs(out_dir, exist_ok=True)
    # If val_field is a file and it's json -> good
    if isinstance(val_field, str) and os.path.isfile(val_field) and val_field.lower().endswith(".json"):
        return val_field

    # If val_field is a directory, try to find a COCO json inside
    if isinstance(val_field, str) and os.path.isdir(val_field):
        found = find_coco_json_in_folder(val_field)
        if found:
            return found
        # maybe the YAML points to images dir; try to locate labels in sibling 'labels' folder
        # attempt to infer structure:
        images_dir = val_field
        # possible labels dir candidates near images_dir
        candidates = [
            os.path.join(os.path.dirname(images_dir), "labels"),
            os.path.join(images_dir, "labels"),
            os.path.join(os.path.dirname(images_dir), "annotations"),
            os.path.join(images_dir, "annotations"),
            os.path.dirname(images_dir)
        ]
        labels_dir = None
        for c in candidates:
            if c and os.path.isdir(c):
                # check if there are .txt files
                if len(glob.glob(os.path.join(c, "*.txt"))) > 0:
                    labels_dir = c
                    break
        # if we didn't find labels, also check for .txt in images dir (some people put labels alongside images)
        if labels_dir is None and len(glob.glob(os.path.join(images_dir, "*.txt"))) > 0:
            labels_dir = images_dir

        if labels_dir:
            # get classes list
            classes = None
            if data_yaml:
                # YAML may have names: either list or path to names file
                names_obj = data_yaml.get("names") or data_yaml.get("names_file") or data_yaml.get("nc")
                if isinstance(names_obj, list):
                    classes = names_obj
                elif isinstance(names_obj, str):
                    names_path = names_obj
                    if dataset_root and not os.path.isabs(names_path):
                        names_path = os.path.join(dataset_root, names_path)
                    if os.path.exists(names_path):
                        # file listing names (one per line)
                        with open(names_path, 'r') as f:
                            classes = [ln.strip() for ln in f if ln.strip()]
            if classes is None:
                # fallback: infer max class index from label files and create generic names
                max_cls = -1
                for lf in glob.glob(os.path.join(labels_dir, "*.txt")):
                    with open(lf, 'r') as f:
                        for ln in f:
                            parts = ln.strip().split()
                            if parts:
                                cls_i = int(parts[0])
                                if cls_i > max_cls: max_cls = cls_i
                classes = [f"class{c}" for c in range(max_cls+1)] if max_cls >= 0 else ["class0"]

            out_json = os.path.join(out_dir, os.path.basename(labels_dir.strip("/")) + "_coco.json")
            print("Converting YOLO labels -> COCO JSON:", out_json)
            return yolo_to_coco(images_dir, labels_dir, classes, out_json)
        else:
            raise FileNotFoundError("Could not find COCO JSON or YOLO labels in directory: " + images_dir)

    # if val_field is a path-like that doesn't exist, maybe it's relative to dataset_root
    if dataset_root and isinstance(val_field, str):
        alt = os.path.join(dataset_root, val_field)
        if os.path.exists(alt):
            return get_coco_gt(alt, dataset_root=dataset_root, data_yaml=data_yaml, out_dir=out_dir)

    raise FileNotFoundError("Cannot resolve val path to a COCO JSON or YOLO labels: " + str(val_field))

# ---------- Example integration with your existing code ----------
# Suppose WAID_DATA_YAML and BUCK_DATA_YAML are defined (paths to their data.yaml)
# Load each YAML and get proper GT JSON (or convert)
WAID_DATA_YAML = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml"
BUCK_DATA_YAML = "/kaggle/input/bucktales-patched/bucktales_patched/yolov8_format_v1/data.yaml"

# parse
waid_yaml = read_yaml(WAID_DATA_YAML)
buck_yaml = read_yaml(BUCK_DATA_YAML)

# Resolve val fields (dataset_root helps if YAML uses relative paths)
waid_val_field = waid_yaml.get("val") or waid_yaml.get("val_path") or waid_yaml.get("val_images")
buck_val_field = buck_yaml.get("val") or buck_yaml.get("val_path") or buck_yaml.get("val_images")

# Call helper: this will either return a COCO json if present, or convert YOLO labels -> COCO json
gt_waid_json = get_coco_gt(waid_val_field, dataset_root=os.path.dirname(WAID_DATA_YAML), data_yaml=waid_yaml)
gt_buck_json = get_coco_gt(buck_val_field, dataset_root=os.path.dirname(BUCK_DATA_YAML), data_yaml=buck_yaml)

print("Resolved GT WAID JSON:", gt_waid_json)
print("Resolved GT BUCK JSON:", gt_buck_json)

# Now you can call per_image_ap50(gt_waid_json, pred_waid, ...) from the earlier cell,
# where pred_waid is the predictions.json produced by ultralytics val/save_json.
