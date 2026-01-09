# compare_vanilla_vs_double.py
# Compare vanilla YOLO detections with double-stage (refined) detections.
# Save side-by-side images (Vanilla | Double) only when instance counts differ.

import os
import json
import time
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import torch
from ultralytics import YOLO

# ------------- CONFIG (change these or pass CLI args) -------------
IMAGE_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test"
LABEL_DIR = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/labels/test"  # optional, not used here
PREDICTIONS_JSON = "/kaggle/input/json-files/spdp2p2.json"  # single-stage predictions to refine
CUSTOM_MODEL_WEIGHTS = "/kaggle/input/yolo-weights/weights/spdp2p2.pt"  # model used for refinement
VANILLA_MODEL_WEIGHTS = "/kaggle/input/yolo-weights/weights/vanillasoap.pt"  # vanilla YOLO to compare against

CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45

OUT_DIR = "/kaggle/working/compare_vanilla_vs_double"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------- logging -------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------- lightweight utilities (IoU, NMS, crop helpers) -------------
def calculate_iou(box1, box2):
    # boxes are [x1,y1,x2,y2]
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def torchvision_nms(boxes, scores, labels, iou_threshold=0.45):
    # Keep boxes per-class using torchvision (fallback to simple greedy)
    if len(boxes) == 0:
        return [], [], []
    try:
        import torchvision
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)
        keep_indices = []
        for label in torch.unique(labels_t):
            mask = (labels_t == label).nonzero(as_tuple=True)[0]
            if mask.numel() == 0: 
                continue
            label_boxes = boxes_t[mask]
            label_scores = scores_t[mask]
            keep = torchvision.ops.nms(label_boxes, label_scores, iou_threshold)
            orig_indices = mask[keep].tolist()
            keep_indices.extend(orig_indices)
        keep_indices = sorted(keep_indices)
        return (boxes_t[keep_indices].tolist(), scores_t[keep_indices].tolist(), labels_t[keep_indices].tolist())
    except Exception:
        # simple greedy fallback (class-agnostic)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        kept = []
        while idxs:
            i = idxs.pop(0)
            keep = True
            for j in kept:
                if calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                    keep = False
                    break
            if keep:
                kept.append(i)
        return [boxes[i] for i in kept], [scores[i] for i in kept], [labels[i] for i in kept]

def calculate_optimal_crop_batch(detections, img_width, img_height, pad_factor=0.2):
    crops = []
    for det in detections:
        x1,y1,x2,y2 = det['bbox']
        sw = max(1, x2-x1); sh = max(1, y2-y1)
        cx, cy = (x1+x2)/2, (y1+y2)/2
        pad_w = sw*pad_factor; pad_h = sh*pad_factor
        cw = sw + 2*pad_w; ch = sh + 2*pad_h
        nx1 = max(0, int(cx - cw/2)); ny1 = max(0, int(cy - ch/2))
        nx2 = min(img_width, int(cx + cw/2)); ny2 = min(img_height, int(cy + ch/2))
        if nx2-nx1 < 10 or ny2-ny1 < 10:
            ms = 32
            nx1 = max(0, int(cx - ms/2)); ny1 = max(0, int(cy - ms/2))
            nx2 = min(img_width, int(cx + ms/2)); ny2 = min(img_height, int(cy + ms/2))
        crops.append({'x1':nx1,'y1':ny1,'x2':nx2,'y2':ny2})
    return crops

def prepare_cropped_image_cv2(img_array, crop_info):
    crop = img_array[crop_info['y1']:crop_info['y2'], crop_info['x1']:crop_info['x2']]
    h,w = crop.shape[:2]
    if h==0 or w==0:
        return None
    ratio = min(640/w, 640/h)
    new_size = (int(w*ratio), int(h*ratio))
    resized = cv2.resize(crop, new_size, interpolation=cv2.INTER_LINEAR)
    padded = np.full((640,640,3), 114, dtype=np.uint8)
    pad_x = (640 - new_size[0])//2
    pad_y = (640 - new_size[1])//2
    padded[pad_y:pad_y+new_size[1], pad_x:pad_x+new_size[0]] = resized
    return {'image': Image.fromarray(padded), 'pad_x':pad_x, 'pad_y':pad_y, 'ratio':ratio}

def scale_boxes_vectorized(boxes, pad_x, pad_y, crop_info, ratio):
    if len(boxes) == 0:
        return np.array([])
    scaled = np.array(boxes).copy()
    scaled[:, [0,2]] -= pad_x
    scaled[:, [1,3]] -= pad_y
    scaled = scaled / ratio
    scaled[:, [0,2]] += crop_info['x1']
    scaled[:, [1,3]] += crop_info['y1']
    return scaled

# ------------- double-inference helpers (same logic as your pipeline) -------------
def process_refined_boxes_optimized(scaled_boxes, labels, confs, original_detection, img_width, img_height):
    if scaled_boxes.size == 0 or len(labels)==0 or len(confs)==0:
        return None
    if not (len(scaled_boxes)==len(labels)==len(confs)):
        return None
    orig_label = original_detection['category_id']
    orig_score = original_detection['score']
    orig_bbox = original_detection['bbox']
    # filter by same label
    mask = (labels == orig_label)
    if not mask.any():
        return None
    valid_boxes = scaled_boxes[mask]
    valid_confs = confs[mask]
    valid_labels = labels[mask]
    # bounds
    keep_mask = (valid_boxes[:,2] > valid_boxes[:,0]) & (valid_boxes[:,3] > valid_boxes[:,1]) & \
                (valid_boxes[:,0] >= 0) & (valid_boxes[:,1] >= 0) & \
                (valid_boxes[:,2] <= img_width) & (valid_boxes[:,3] <= img_height)
    if not keep_mask.any():
        return None
    final_boxes = valid_boxes[keep_mask]
    final_confs = valid_confs[keep_mask]
    final_labels = valid_labels[keep_mask]
    best_idx = -1; best_combined = -1
    for i, (box, conf, label) in enumerate(zip(final_boxes, final_confs, final_labels)):
        cur_iou = calculate_iou(orig_bbox, box)
        if cur_iou < 0.25:
            continue
        combined = conf*0.6 + cur_iou*0.4
        if combined > best_combined:
            best_combined = combined
            best_idx = i
    if best_idx >= 0 and final_confs[best_idx] > orig_score:
        return {'bbox': final_boxes[best_idx].tolist(), 'score': float(final_confs[best_idx]), 'category_id': int(final_labels[best_idx])}
    return None

def perform_batch_double_inference(image_path, model, detections, use_augment=False):
    """
    detections: list of dicts {'bbox':[x1,y1,x2,y2], 'score':float, 'category_id':int}
    returns list of refined detections (some may be None)
    """
    try:
        img_pil = Image.open(image_path).convert("RGB")
        img_array = np.array(img_pil)
        img_w, img_h = img_pil.size
        crop_infos = calculate_optimal_crop_batch(detections, img_w, img_h)
        processed_images = []
        valid_dets = []
        valid_crops = []
        for det, cinfo in zip(detections, crop_infos):
            if cinfo['x2'] <= cinfo['x1'] or cinfo['y2'] <= cinfo['y1']:
                continue
            proc = prepare_cropped_image_cv2(img_array, cinfo)
            if proc is None:
                continue
            processed_images.append(proc)
            valid_dets.append(det)
            valid_crops.append(cinfo)
        if not processed_images:
            return []
        refined_all = []
        # We can process in small batches (YOLO predict sometimes supports only individual images)
        batch_size = min(4, len(processed_images))
        for i in range(0, len(processed_images), batch_size):
            batch = processed_images[i:i+batch_size]
            batch_dets = valid_dets[i:i+batch_size]
            batch_crops = valid_crops[i:i+batch_size]
            # predict each crop (Ultralytics predict returns a list-like)
            results = []
            with torch.no_grad():
                for p in batch:
                    res = model.predict(p['image'], verbose=False, augment=use_augment)
                    results.extend(res)
            # results correspond to each crop in order
            for res, orig_det, proc, crop_info in zip(results, batch_dets, batch, batch_crops):
                if getattr(res, 'boxes', None) is None or len(res.boxes)==0:
                    refined_all.append(None)
                    continue
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                labels = res.boxes.cls.cpu().numpy().astype(int)
                if boxes.ndim == 1:
                    boxes = np.expand_dims(boxes, 0)
                    confs = np.expand_dims(confs, 0)
                    labels = np.expand_dims(labels, 0)
                scaled = scale_boxes_vectorized(boxes, proc['pad_x'], proc['pad_y'], crop_info, proc['ratio'])
                refined = process_refined_boxes_optimized(scaled, labels, confs, orig_det, img_w, img_h)
                refined_all.append(refined)
        return refined_all
    except Exception as e:
        logging.error("perform_batch_double_inference error: %s", e)
        return []

# ------------- predictions & IO -------------
def load_image_predictions(predictions_path, conf_threshold=0.25):
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(predictions_path)
    with open(predictions_path, "r") as f:
        preds = json.load(f)
    image_predictions = {}
    for p in preds:
        image_name = p["image_id"]
        if image_name not in image_predictions:
            image_predictions[image_name] = {"boxes": [], "scores": [], "labels": []}
        if p["score"] >= conf_threshold:
            x,y,w,h = p["bbox"]
            x1,y1,x2,y2 = x, y, x+w, y+h
            image_predictions[image_name]["boxes"].append([x1,y1,x2,y2])
            image_predictions[image_name]["scores"].append(p["score"])
            image_predictions[image_name]["labels"].append(p["category_id"])
    return image_predictions

def draw_boxes(img, boxes, color=(0,255,0), labels=None, scores=None):
    out = img.copy()
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        txt = None
        if labels is not None:
            txt = str(labels[i])
            if scores is not None:
                txt += f":{scores[i]:.2f}"
        if txt:
            cv2.putText(out, txt, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out

def run_inference_simple(model, img_path, conf=CONF_THRESHOLD, iou=NMS_IOU_THRESHOLD):
    try:
        res = model.predict(img_path, conf=conf, iou=iou, verbose=False)
        if not res:
            return []
        r = res[0]
        if getattr(r, 'boxes', None) is None or len(r.boxes) == 0:
            return []
        return r.boxes.xyxy.cpu().numpy().tolist()
    except Exception as e:
        logging.error("run_inference_simple error: %s", e)
        return []

# ------------- main compare loop -------------
def main():
    t0 = time.time()
    logging.info("Loading JSON predictions...")
    image_predictions = load_image_predictions(PREDICTIONS_JSON, CONF_THRESHOLD)

    logging.info("Loading models...")
    custom_model = YOLO(CUSTOM_MODEL_WEIGHTS)  # used for refining crops
    vanilla_model = YOLO(VANILLA_MODEL_WEIGHTS)  # vanilla for comparison

    saved = 0
    processed = 0

    for image_name, preds in image_predictions.items():
        processed += 1
        img_filename = image_name
        # If JSON image_id has extension or not - try several candidates
        possible_files = [
            os.path.join(IMAGE_DIR, image_name),
            os.path.join(IMAGE_DIR, image_name + ".jpg"),
            os.path.join(IMAGE_DIR, image_name + ".jpeg"),
            os.path.join(IMAGE_DIR, image_name + ".png"),
        ]
        image_path = None
        for p in possible_files:
            if os.path.exists(p):
                image_path = p
                break
        if image_path is None:
            logging.warning("Image not found for %s, skipping", image_name)
            continue

        # Build detection dicts for refinement
        detections = []
        for b,s,l in zip(preds["boxes"], preds["scores"], preds["labels"]):
            detections.append({'bbox': b, 'score': s, 'category_id': l})

        # Run double/refinement pipeline -> refined_results list (aligned to detections subset)
        refined_list = perform_batch_double_inference(image_path, custom_model, detections, use_augment=False)

        # Apply refined results over original predictions (one-to-one with detections list)
        # We'll create final_refined_predictions = boxes, scores, labels
        refined_boxes = preds["boxes"].copy()
        refined_scores = preds["scores"].copy()
        refined_labels = preds["labels"].copy()
        # Note: refined_list has same ordering as detections that were processed.
        # There may be None values; in that case, we keep original.
        for idx, refined in enumerate(refined_list):
            if refined is None:
                continue
            # Replace the original detection at idx with refined values.
            refined_boxes[idx] = refined['bbox']
            refined_scores[idx] = refined['score']
            refined_labels[idx] = refined['category_id']

        # Apply NMS to refined predictions
        if len(refined_boxes) > 0:
            refined_boxes, refined_scores, refined_labels = torchvision_nms(refined_boxes, refined_scores, refined_labels, NMS_IOU_THRESHOLD)

        # Run vanilla YOLO on the same image
        vanilla_boxes = run_inference_simple(vanilla_model, image_path)

        # Compare counts
        if len(vanilla_boxes) == len(refined_boxes):
            # counts equal -> do not save
            continue

        # Draw and save side-by-side (Vanilla | Double)
        img = cv2.imread(image_path)
        if img is None:
            logging.warning("Failed to read image for drawing: %s", image_path)
            continue
        vanilla_vis = draw_boxes(img, vanilla_boxes, color=(0,0,255))  # RED
        double_vis = draw_boxes(img, refined_boxes, color=(0,255,0), labels=None, scores=None)  # GREEN

        # Ensure same height for hconcat
        try:
            combined = cv2.hconcat([vanilla_vis, double_vis])
        except Exception:
            h = max(vanilla_vis.shape[0], double_vis.shape[0])
            vanilla_vis = cv2.resize(vanilla_vis, (vanilla_vis.shape[1], h))
            double_vis = cv2.resize(double_vis, (double_vis.shape[1], h))
            combined = cv2.hconcat([vanilla_vis, double_vis])

        # Add text labels and counts
        h, w = img.shape[:2]
        cv2.putText(combined, f"Vanilla ({len(vanilla_boxes)})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.putText(combined, f"Double ({len(refined_boxes)})", (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        out_name = os.path.basename(image_path)
        save_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(save_path, combined)
        saved += 1
        logging.info("Saved diff image: %s (vanilla=%d, double=%d)", save_path, len(vanilla_boxes), len(refined_boxes))

    logging.info("Processed %d images; saved %d differences; elapsed %.1f s", processed, saved, time.time()-t0)
    print(f"Saved {saved} comparison images to: {OUT_DIR}")

if __name__ == "__main__":
    main()

