import os
import json
import glob
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy import stats

# ============== USER CONFIG ==============
# Path to YOLO weights (put your .pt in /kaggle/input/weights/)
WEIGHTS = "/kaggle/input/yolo-weights/weights/spdld.pt"

# Datasets - point to folder containing a data.yaml and COCO validation JSON
WAID_DATA_YAML = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml"
BUCK_DATA_YAML = "/kaggle/input/bucktales-patched/bucktales_patched/dtc2023.yaml"

# Output folder for predictions and per-image CSVs
OUT_DIR = "/kaggle/working/yolo_eval_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
# =========================================

# Quick check
for p in [WEIGHTS, WAID_DATA_YAML, BUCK_DATA_YAML]:
    if not os.path.exists(p):
        print("WARNING: path missing:", p)

# Utility: find predictions JSON from ultralytics run_dir if needed
def find_predictions_json(search_root="."):
    # search for predictions.json or /predictions.json created by ultralytics val(save_json=True)
    matches = glob.glob(os.path.join(search_root, "**", "predictions.json"), recursive=True)
    return matches[-1] if matches else None

# Utility: compute per-image AP@0.5 by running COCOeval for each image (slow but exact)
def per_image_ap50(gt_json_path, pred_json_path, img_ids=None, use_tqdm=True):
    """
    Returns DataFrame with columns: image_id, ap50
    gt_json_path: path to COCO-format ground truth json
    pred_json_path: path to COCO-format predictions json (list of detections)
    img_ids: optional list of image ids to evaluate (defaults to all in GT)
    """
    cocoGt = COCO(gt_json_path)
    cocoDt = cocoGt.loadRes(pred_json_path)

    if img_ids is None:
        img_ids = cocoGt.getImgIds()
    results = []
    # We'll set COCOeval to use only IoU=0.5
    for imgId in tqdm(img_ids, disable=not use_tqdm):
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        cocoEval.params.imgIds = [imgId]
        cocoEval.params.iouThrs = np.array([0.5])  # only IoU=0.5
        cocoEval.params.maxDets = [100]  # keep default max detections
        # evaluate -> accumulate -> compute precision array
        cocoEval.evaluate()
        cocoEval.accumulate()
        # precision shape: [T, R, K, A, M] where T=len(iouThrs)=1, R=len(recThrs), K=numCats, A=area ranges, M=maxDets
        prec = cocoEval.eval.get('precision')
        if prec is None:
            ap50_val = 0.0
        else:
            # pick the slice for IoU=0.5 -> prec[0, :, :, 0, 0]
            arr = prec[0, :, :, 0, 0]  # shape (recThrs, numCats)
            valid = arr[arr > -1]
            ap50_val = float(np.mean(valid)) if valid.size > 0 else 0.0
        results.append({"image_id": imgId, "ap50": ap50_val})
    df = pd.DataFrame(results)
    return df

# Utility: effect sizes
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_sd = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_sd

def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    greater = 0
    lesser = 0
    for a in x:
        for b in y:
            if a > b:
                greater += 1
            elif a < b:
                lesser += 1
    return (greater - lesser) / (nx * ny)

# Cell 3: Load model
print("Loading model from", WEIGHTS)
model = YOLO(WEIGHTS)

# Cell 4: Run validation on WAID and BUCK to produce predictions.json (COCO-style)
# ultralytics val(..., save_json=True) will create predictions.json in the run folder.
print("Running validation (this will run model.val for each dataset and save predictions.json).")
res_waid = model.val(data=WAID_DATA_YAML, imgsz=640, save_json=True)   # adjust imgsz if needed
res_buck = model.val(data=BUCK_DATA_YAML, imgsz=640, save_json=True)

# Locate predictions JSON files (Ultralytics saves to runs/val/exp*/predictions.json)
pred_waid = find_predictions_json(search_root=".")
pred_buck = find_predictions_json(search_root=".")
print("Predictions (WAID):", pred_waid)
# Note: if both runs created predictions.json, the second call above will likely point to the last file.
# To be explicit, you can search for the latest files by modification time:
all_preds = sorted(glob.glob("**/predictions.json", recursive=True), key=os.path.getmtime)
if len(all_preds) >= 2:
    # heuristics: last two correspond to the two runs
    pred_buck = all_preds[-1]
    pred_waid = all_preds[-2]
elif len(all_preds) == 1:
    # only one found: maybe one dataset only ran
    pred_waid = pred_buck = all_preds[0]

print("Using predictions:")
print(" WAID predictions:", pred_waid)
print(" BUCK predictions:", pred_buck)

# Cell 5: Identify GT annotation files from data.yaml (we will parse them)
def read_coco_gt_from_yaml(yaml_path):
    import yaml
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    # yaml typically contains something like: val: /path/to/annotations/instances_val2017.json
    val = d.get("val") or d.get("val_path") or d.get("val_annotations")
    if isinstance(val, list):
        # take first
        val = val[0]
    return val

gt_waid = read_coco_gt_from_yaml(WAID_DATA_YAML)
gt_buck = read_coco_gt_from_yaml(BUCK_DATA_YAML)
print("GT WAID:", gt_waid)
print("GT BUCK:", gt_buck)

# Cell 6: Compute per-image AP@0.5 (this will take time)
print("Computing per-image AP@0.5 for WAID...")
df_waid = per_image_ap50(gt_waid, pred_waid, use_tqdm=True)
print("Computing per-image AP@0.5 for BUCKTales...")
df_buck = per_image_ap50(gt_buck, pred_buck, use_tqdm=True)

# Save CSVs
waid_csv = os.path.join(OUT_DIR, "waid_per_image_ap50.csv")
buck_csv = os.path.join(OUT_DIR, "buck_per_image_ap50.csv")
df_waid.to_csv(waid_csv, index=False)
df_buck.to_csv(buck_csv, index=False)
print("Saved:", waid_csv)
print("Saved:", buck_csv)

# Cell 7: Statistical testing
# If datasets are independent (different images) -> Mann-Whitney U or t-test
# If paired (same images) -> paired t-test or Wilcoxon
waid_vals = df_waid["ap50"].values
buck_vals = df_buck["ap50"].values

# QUICK DECISION: Are image sets identical (paired) ?
paired = False
# crude check: if any image ids overlap -> could be paired if corresponding images are the same
overlap = set(df_waid["image_id"]).intersection(set(df_buck["image_id"]))
if len(overlap) > 0:
    print(f"Found {len(overlap)} overlapping image ids -> treating as paired comparison.")
    paired = True

# Normality checks
print("Shapiro WAID:", stats.shapiro(waid_vals))
print("Shapiro BUCK:", stats.shapiro(buck_vals))

# Variance check (if independent)
if not paired:
    print("Levene test for equal variances:", stats.levene(waid_vals, buck_vals))

# Perform tests
results = {}
if paired:
    # align values on overlapping image ids
    common_ids = sorted(list(overlap))
    # create maps
    waid_map = df_waid.set_index("image_id").loc[common_ids]["ap50"].values
    buck_map = df_buck.set_index("image_id").loc[common_ids]["ap50"].values
    # paired t-test
    results['paired_ttest'] = stats.ttest_rel(waid_map, buck_map)
    # Wilcoxon
    try:
        results['wilcoxon'] = stats.wilcoxon(waid_map, buck_map)
    except Exception as e:
        results['wilcoxon'] = ("error", str(e))
    results['cohens_d'] = cohens_d(waid_map, buck_map)
    results['cliffs_delta'] = cliffs_delta(waid_map, buck_map)
else:
    # independent case
    # Normal t-test (Welch)
    results['ttest_ind'] = stats.ttest_ind(waid_vals, buck_vals, equal_var=False)
    # Mann-Whitney U
    results['mannwhitneyu'] = stats.mannwhitneyu(waid_vals, buck_vals, alternative="two-sided")
    results['cohens_d'] = cohens_d(waid_vals, buck_vals)
    results['cliffs_delta'] = cliffs_delta(waid_vals, buck_vals)

# Print summary
print("\n=== STATISTICAL TESTS SUMMARY ===")
for k, v in results.items():
    print(k, "->", v)

# Save summary to CSV
summary = {
    "waid_mean_ap50": float(np.mean(waid_vals)),
    "waid_std_ap50": float(np.std(waid_vals, ddof=1)),
    "waid_n": int(len(waid_vals)),
    "buck_mean_ap50": float(np.mean(buck_vals)),
    "buck_std_ap50": float(np.std(buck_vals, ddof=1)),
    "buck_n": int(len(buck_vals)),
    "paired": paired,
    "tests": {k: (str(v) if not hasattr(v, 'pvalue') else (float(getattr(v, 'statistic')), float(getattr(v, 'pvalue')))) for k,v in results.items()}
}
summary_path = os.path.join(OUT_DIR, "stats_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print("Saved summary:", summary_path)

# Cell 8: Print help/hints for reading outputs
print("\nOutputs are in:", OUT_DIR)
print(" - per-image CSVs:", waid_csv, buck_csv)
print(" - stats summary JSON:", summary_path)
