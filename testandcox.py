import numpy as np
import yaml
import random
from scipy import stats
from ultralytics import YOLO  # Or import your custom DEAL-YOLO class
import matplotlib.pyplot as plt
import os

# --- Configuration ---
DEAL_YOLO_WEIGHTS = '/kaggle/input/yolo-weights/weights/spdld.pt'  # Path to your custom weights
BASELINE_WEIGHTS = '/kaggle/input/waid-no-soap/vanillanosoap.pt'             # Path to baseline weights (e.g., standard YOLO)
DATA_YAML = '/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml'
NUM_SAMPLES = 30         # Number of bootstrap samples (30 is good for CLT)
SAMPLE_FRACTION = 0.5    # Use 50% of the test set per sample to speed up
DEVICE = 'gpu'               # GPU ID

def get_image_list(yaml_path):
    """Extracts the list of test images from the YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Assuming 'test' points to a folder or txt file
    test_path = data.get('test')
    
    # Basic logic to get file paths (adjust based on your actual folder structure)
    if os.path.isdir(test_path):
        images = [os.path.join(test_path, img) for img in os.listdir(test_path) if img.endswith(('.jpg', '.png'))]
    elif test_path.endswith('.txt'):
        with open(test_path, 'r') as f:
            images = f.read().splitlines()
    else:
        raise ValueError("Could not parse test path from YAML")
    
    return images

def run_bootstrap_test():
    # 1. Setup Models
    model_deal = YOLO(DEAL_YOLO_WEIGHTS)
    model_base = YOLO(BASELINE_WEIGHTS)
    
    # 2. Get all test images
    all_test_images = get_image_list(DATA_YAML)
    n_size = int(len(all_test_images) * SAMPLE_FRACTION)
    
    deal_scores = []
    base_scores = []
    
    print(f"Starting Bootstrapping with {NUM_SAMPLES} samples of {n_size} images each...")
    
    # 3. Bootstrap Loop
    for i in range(NUM_SAMPLES):
        # Sample with replacement (standard bootstrap) or without (subsampling)
        # Subsampling (without replacement) is often preferred for validation subsets
        subset_images = random.sample(all_test_images, n_size)
        
        # Create a temporary validation file for this subset
        temp_yaml = 'temp_val_subset.yaml'
        temp_txt = 'temp_val_list.txt'
        
        with open(temp_txt, 'w') as f:
            f.write('\n'.join(subset_images))
            
        # Create a temp yaml config that points to this subset
        with open(DATA_YAML, 'r') as f:
            base_config = yaml.safe_load(f)
        
        base_config['val'] = temp_txt # Override val source
        base_config['test'] = temp_txt
        
        with open(temp_yaml, 'w') as f:
            yaml.dump(base_config, f)
            
        print(f"\n--- Iteration {i+1}/{NUM_SAMPLES} ---")
        
        # Evaluate DEAL-YOLO
        # We use map50 (mAP@0.5) or map50-95 depending on your preference
        results_deal = model_deal.val(data=temp_yaml, device=DEVICE, verbose=False, plots=False)
        score_deal = results_deal.box.map50  # Change to .map for mAP@0.5:0.95
        deal_scores.append(score_deal)
        
        # Evaluate Baseline
        results_base = model_base.val(data=temp_yaml, device=DEVICE, verbose=False, plots=False)
        score_base = results_base.box.map50
        base_scores.append(score_base)
        
        print(f"Deal: {score_deal:.4f} | Base: {score_base:.4f} | Diff: {score_deal - score_base:.4f}")

    return deal_scores, base_scores

# --- Execute ---
deal_scores, base_scores = run_bootstrap_test()

# --- Statistics ---
print("\n" + "="*30)
print("SIGNIFICANCE TESTING RESULTS")
print("="*30)

# 1. Shapiro-Wilk Test (Check for Normality)
# If p < 0.05, data is NOT normal -> Use Wilcoxon
_, p_norm_deal = stats.shapiro(deal_scores)
_, p_norm_diff = stats.shapiro(np.array(deal_scores) - np.array(base_scores))

print(f"Normality Check (p-value): {p_norm_diff:.4f}")
if p_norm_diff > 0.05:
    print("-> Distribution of differences is Normal. You can rely on t-test.")
else:
    print("-> Distribution is NOT Normal. Rely on Wilcoxon.")

# 2. Paired T-Test
t_stat, p_ttest = stats.ttest_rel(deal_scores, base_scores)
print(f"\nPaired T-Test:")
print(f"  statistic: {t_stat:.4f}")
print(f"  p-value:   {p_ttest:.4e}")

# 3. Wilcoxon Signed-Rank Test
w_stat, p_wilcoxon = stats.wilcoxon(deal_scores, base_scores)
print(f"\nWilcoxon Signed-Rank Test:")
print(f"  statistic: {w_stat:.4f}")
print(f"  p-value:   {p_wilcoxon:.4e}")

# 4. Final Verdict
alpha = 0.05
print("-" * 30)
if p_wilcoxon < alpha:
    print(f"✅ SIGNIFICANT DIFFERENCE FOUND (p < {alpha})")
    print(f"Deal-YOLO Mean mAP: {np.mean(deal_scores):.4f}")
    print(f"Baseline Mean mAP:  {np.mean(base_scores):.4f}")
else:
    print(f"❌ NO SIGNIFICANT DIFFERENCE (p >= {alpha})")