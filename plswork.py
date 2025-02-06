import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    
    return intersection / (area1 + area2 - intersection + 1e-6)

def simple_nms(boxes, scores, iou_threshold=0.5):
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
        
        iou = intersection / union
        mask = iou <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return keep

def scale_boxes(padded_boxes, pad_x, pad_y, resize_ratio_x, resize_ratio_y, crop_coords):
    try:
        if padded_boxes is None or not isinstance(padded_boxes, np.ndarray):
            return np.empty((0, 4))
        
        if padded_boxes.size == 0:
            return np.empty((0, 4))
        if padded_boxes.ndim == 1:
            padded_boxes = np.expand_dims(padded_boxes, 0)
            
        boxes = padded_boxes.copy()
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] - pad_x, 0, crop_coords['resized_w'])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] - pad_y, 0, crop_coords['resized_h'])
        
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * resize_ratio_x + crop_coords['x1']
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * resize_ratio_y + crop_coords['y1']
        
        return boxes
    except Exception as e:
        print(f"Scaling error: {str(e)}")
        return np.empty((0, 4))

def validate_double_inference(model_path, data_yaml, split='val', imgsz=640, batch=16, 
                            conf_threshold=0.25, refinement_conf=0.1, nms_iou=0.5,
                            project='runs/val', name='double-inference'):
    """
    Custom validation function implementing double inference method
    """
    model = YOLO(model_path)
    metric = MeanAveragePrecision(class_metrics=True)
    
    # Parse data.yaml to get paths
    import yaml
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    # Get appropriate paths based on split
    base_path = os.path.dirname(data_yaml)
    image_dir = os.path.join(base_path, data_dict[f'{split}_images'])
    label_dir = os.path.join(base_path, data_dict[f'{split}_labels'])
    
    total_predictions = 0
    correct_predictions = 0
    
    # Process each image
    for image_path in os.listdir(image_dir):
        img = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
        img_width, img_height = img.size
        
        # Initial prediction
        initial_results = model.predict(img, conf=conf_threshold, verbose=False)[0]
        
        # Process predictions
        predictions = {
            'boxes': [box.xyxy[0].cpu().numpy().tolist() for box in initial_results.boxes],
            'scores': [box.conf.item() for box in initial_results.boxes],
            'labels': [int(box.cls.item()) for box in initial_results.boxes]
        }
        
        # Load ground truth
        true_boxes, true_labels = [], []
        label_path = os.path.join(label_dir, os.path.splitext(image_path)[0] + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    x1 = (x_center - width/2) * img_width
                    y1 = (y_center - height/2) * img_height
                    x2 = (x_center + width/2) * img_width
                    y2 = (y_center + height/2) * img_height
                    true_boxes.append([x1, y1, x2, y2])
                    true_labels.append(int(class_id))
        
        # Reference box for scaling
        ref_idx = np.argmax(predictions['scores']) if predictions['scores'] else -1
        if ref_idx != -1:
            ref_box = predictions['boxes'][ref_idx]
            rw = ref_box[2] - ref_box[0]
            rh = ref_box[3] - ref_box[1]
        else:
            rw, rh = 0, 0
        
        # Refinement pass
        replacement_candidates = []
        for i in range(len(predictions['scores'])):
            if predictions['scores'][i] >= conf_threshold or rw == 0 or rh == 0:
                continue
                
            pre_box = predictions['boxes'][i]
            original_label = predictions['labels'][i]
            original_score = predictions['scores'][i]
            x1, y1, x2, y2 = pre_box
            sw, sh = x2 - x1, y2 - y1
            
            # Calculate adaptive crop size
            desired_width = (sw * imgsz) / rw if rw != 0 else imgsz
            desired_height = (sh * imgsz) / rh if rh != 0 else imgsz
            cx, cy = (x1 + x2)/2, (y1 + y2)/2
            
            # Crop and process region
            new_x1 = max(0, int(cx - desired_width/2))
            new_y1 = max(0, int(cy - desired_height/2))
            new_x2 = min(img_width, int(cx + desired_width/2))
            new_y2 = min(img_height, int(cy + desired_height/2))
            
            if (new_x2 <= new_x1) or (new_y2 <= new_y1):
                continue
            
            # Second pass inference on cropped region
            crop = img.crop((new_x1, new_y1, new_x2, new_y2))
            original_w, original_h = crop.size
            ratio = min(imgsz/original_w, imgsz/original_h)
            new_size = (int(original_w*ratio), int(original_h*ratio))
            resized = crop.resize(new_size, Image.BILINEAR)
            
            padded_img = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
            pad_x, pad_y = (imgsz - new_size[0])//2, (imgsz - new_size[1])//2
            padded_img.paste(resized, (pad_x, pad_y))
            
            new_results = model.predict(padded_img, conf=refinement_conf, verbose=False)[0]
            
            if len(new_results.boxes) == 0:
                continue
            
            # Process refined detections
            boxes = new_results.boxes.xyxy.cpu().numpy()
            if boxes.ndim == 1:
                boxes = np.expand_dims(boxes, axis=0)
                
            confs = new_results.boxes.conf.cpu().numpy()
            labels = new_results.boxes.cls.cpu().numpy().astype(int)
            
            # Scale boxes back to original image coordinates
            crop_w, crop_h = new_x2 - new_x1, new_y2 - new_y1
            scale_x = crop_w / new_size[0]
            scale_y = crop_h / new_size[1]
            
            scaled_boxes = scale_boxes(
                boxes.copy(), pad_x, pad_y, scale_x, scale_y,
                {'x1': new_x1, 'y1': new_y1, 
                 'resized_w': new_size[0], 'resized_h': new_size[1]}
            )
            
            if scaled_boxes.size == 0:
                continue
            
            # Find best matching refined detection
            best_match = None
            best_iou = -1
            best_conf = -1
            
            for scaled_box, label, conf in zip(scaled_boxes, labels, confs):
                if label != original_label:
                    continue
                    
                current_iou = calculate_iou(pre_box, scaled_box)
                if current_iou > best_iou or (current_iou == best_iou and conf > best_conf):
                    best_iou = current_iou
                    best_conf = conf
                    best_match = scaled_box
            
            if best_match is not None and best_iou >= 0.25 and best_conf > original_score:
                replacement_candidates.append({
                    'idx': i,
                    'box': best_match.tolist(),
                    'score': best_conf,
                    'label': original_label
                })
        
        # Apply refinements
        final_predictions = {
            'boxes': predictions['boxes'].copy(),
            'scores': predictions['scores'].copy(),
            'labels': predictions['labels'].copy()
        }
        
        for candidate in replacement_candidates:
            i = candidate['idx']
            if final_predictions['scores'][i] < candidate['score']:
                final_predictions['boxes'][i] = candidate['box']
                final_predictions['scores'][i] = candidate['score']
                final_predictions['labels'][i] = candidate['label']
        
        # Apply NMS
        boxes_tensor = torch.tensor(final_predictions['boxes'])
        scores_tensor = torch.tensor(final_predictions['scores'])
        labels_tensor = torch.tensor(final_predictions['labels'])
        
        keep_indices = simple_nms(boxes_tensor, scores_tensor, iou_threshold=nms_iou)
        
        filtered_predictions = {
            'boxes': boxes_tensor[keep_indices].tolist(),
            'scores': scores_tensor[keep_indices].tolist(),
            'labels': labels_tensor[keep_indices].tolist()
        }
        
        # Update metrics
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
        
        # Update statistics
        used_truth_indices = []
        for i, (pred_box, pred_label) in enumerate(zip(np.array(filtered_predictions['boxes']), 
                                                     np.array(filtered_predictions['labels']))):
            total_predictions += 1
            matching_truths = np.where(np.array(true_labels) == pred_label)[0]
            best_iou = 0
            best_truth_idx = -1
            
            for truth_idx in matching_truths:
                if truth_idx in used_truth_indices:
                    continue
                
                iou = calculate_iou(pred_box, true_boxes[truth_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_truth_idx = truth_idx
            
            if best_iou >= 0.5 and best_truth_idx != -1:
                correct_predictions += 1
                used_truth_indices.append(best_truth_idx)
    
    # Compute final metrics
    final_metrics = metric.compute()
    
    # Print results
    print("\nValidation Results:")
    print(f"mAP@0.5: {final_metrics['map_50']:.4f}")
    print(f"Precision: {final_metrics['map_per_class'].mean():.4f}")
    print(f"Recall: {final_metrics['mar_100'].mean():.4f}")
    print(f"\nDetection Statistics:")
    print(f"Total Correct Predictions: {correct_predictions}")
    print(f"Total Predictions Made: {total_predictions}")
    print(f"Accuracy: {correct_predictions/(total_predictions + 1e-7):.4f}")
    
    return final_metrics

if __name__ == '__main__':
    model_path = '/kaggle/input/bucktale-weights/ssfflossp2soapvarifocal.pt'
    data_yaml = '/kaggle/input/bucktales-patched/dtc2023.yaml'
    
    validate_double_inference(
        model_path=model_path,
        data_yaml=data_yaml,
        split='test',
        imgsz=640,
        batch=16,
        # rect=False,
        # save_json=True, # if you need to cal coco metrice
        project='runs/val',
        name='yolov8n-spdld',
        )