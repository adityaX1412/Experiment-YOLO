import os
import json
import yaml
import copy
import random
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch

# --- Configuration (edit paths as needed) ---
DEAL_YOLO_WEIGHTS = '/kaggle/input/yolo-weights/weights/spdld.pt'
BASELINE_WEIGHTS = '/kaggle/input/waid-no-soap/vanillanosoap.pt'
DATA_YAML = '/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml'

NUM_SAMPLES = 30         # number of bootstrap iterations (can increase)
SAMPLE_FRACTION = 0.5    # fraction of testset per sample
RNG_SEED = 42            # reproducibility
BOOTSTRAP_CI_ITERS = 2000  # bootstrap iterations for CI of mean diff

# Device auto-detect (accepts 'gpu' friendly values)
_requested_device = 'gpu'  # change to 'cpu' if you want to force cpu; kept for backward compatibility
if _requested_device.lower() in ('gpu', 'cuda', 'cuda:0', '0'):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    DEVICE = 'cpu'

# --- Utility functions ---
def resolve_path(base_path, maybe_rel):
    if os.path.isabs(maybe_rel):
        return maybe_rel
    return os.path.join(os.path.dirname(base_path), maybe_rel)

def get_image_list(yaml_path):
    """
    Reads the dataset YAML and returns a list of absolute image paths for the test set.
    Handles:
      - test: <directory>
      - test: <file.txt> listing images
      - test: <list> (inline)
      - test: <coco.json> (extracts images->file_name)
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    test_entry = data.get('test') or data.get('val') or data.get('images')  # try a few keys

    if test_entry is None:
        raise ValueError("YAML does not contain a 'test' or 'val' key.")

    # If YAML gave an inline list of files
    if isinstance(test_entry, (list, tuple)):
        return [resolve_path(yaml_path, p) if not os.path.isabs(p) else p for p in test_entry]

    # If entry is a path
    test_path = resolve_path(yaml_path, test_entry)

    if os.path.isdir(test_path):
        imgs = []
        for root, _, files in os.walk(test_path):
            for fname in files:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    imgs.append(os.path.join(root, fname))
        if not imgs:
            raise ValueError(f"No images found under directory: {test_path}")
        return sorted(imgs)

    if test_path.lower().endswith('.txt'):
        with open(test_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        # resolve relative lines wrt yaml dir
        return [resolve_path(yaml_path, p) if not os.path.isabs(p) else p for p in lines]

    if test_path.lower().endswith('.json'):
        with open(test_path, 'r') as f:
            coco = json.load(f)
        imgdir = os.path.dirname(test_path)
        images = []
        for img in coco.get('images', []):
            file_name = img.get('file_name')
            if not file_name:
                continue
            candidate = os.path.join(imgdir, file_name)
            # if file_name already absolute, prefer it
            if os.path.isabs(file_name) and os.path.exists(file_name):
                images.append(file_name)
            elif os.path.exists(candidate):
                images.append(candidate)
            else:
                # last resort: keep file_name as given
                images.append(file_name)
        if not images:
            raise ValueError(f"No images found in COCO JSON: {test_path}")
        return sorted(images)

    raise ValueError(f"Unrecognized test path in YAML: {test_path}")


def extract_map50_from_results(results):
    """
    Try a few common result attributes to extract mAP@0.5 robustly.
    Raise a helpful error if not found.
    """
    # ultralytics sometimes returns a list of Result objects, or a single Result
    candidates = []
    if isinstance(results, (list, tuple)):
        candidates.extend(results)
    else:
        candidates.append(results)

    for r in candidates:
        # 1) r.box.map50 (common in some ultralytics versions)
        try:
            box = getattr(r, 'box', None)
            if box is not None:
                v = getattr(box, 'map50', None) or getattr(box, 'map', None)
                if v is not None:
                    # if map is a dict or numpy scalar, convert
                    try:
                        return float(v)
                    except Exception:
                        pass
        except Exception:
            pass

        # 2) r.metrics or r.stats dict:
        metrics = getattr(r, 'metrics', None) or getattr(r, 'stats', None)
        if isinstance(metrics, dict):
            for key in ('map50', 'mAP_0.5', 'mAP@0.5', 'map_0.5'):
                if key in metrics and metrics[key] is not None:
                    return float(metrics[key])

        # 3) r.box.pr or r.box.map but nested differently
        try:
            if hasattr(r, 'box') and hasattr(r.box, 'map50'):
                return float(r.box.map50)
        except Exception:
            pass

    # last resort: try to stringify the result to help debugging
    raise RuntimeError("Could not extract mAP@0.5 from model.val() result object. "
                       "Inspect the `results` object shape/attributes for your ultralytics version.")


# --- Main bootstrap loop ---
def run_bootstrap_test():
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    model_deal = YOLO(DEAL_YOLO_WEIGHTS)
    model_base = YOLO(BASELINE_WEIGHTS)

    all_test_images = get_image_list(DATA_YAML)
    if len(all_test_images) == 0:
        raise RuntimeError("No test images found.")

    n_size = max(1, int(len(all_test_images) * SAMPLE_FRACTION))
    print(f"Using device: {DEVICE}. Found {len(all_test_images)} test images. "
          f"Each sample uses {n_size} images. Running {NUM_SAMPLES} iterations.")

    deal_scores = []
    base_scores = []
    records = []

    # Load base YAML once and modify per iteration
    with open(DATA_YAML, 'r') as f:
        base_config = yaml.safe_load(f)

    for i in tqdm(range(NUM_SAMPLES), desc="Bootstrap iterations"):
        # Sample WITH replacement for bootstrap
        subset = list(np.random.choice(all_test_images, size=n_size, replace=True))

        # Create temporary txt file listing selected images
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tf:
            temp_txt = tf.name
            tf.write('\n'.join(subset))

        # Create a temp yaml pointing val/test to this txt (some ultralytics versions accept dicts too)
        temp_cfg = copy.deepcopy(base_config)
        temp_cfg['val'] = temp_txt
        temp_cfg['test'] = temp_txt
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as ty:
            temp_yaml = ty.name
            yaml.dump(temp_cfg, ty)

        try:
            # Evaluate DEAL
            results_deal = model_deal.val(data=temp_yaml, device=DEVICE, verbose=False, plots=False)
            score_deal = extract_map50_from_results(results_deal)
        except Exception as e:
            # Clean up temp files before raising so you don't leave garbage
            os.remove(temp_yaml)
            os.remove(temp_txt)
            raise RuntimeError(f"Error while validating DEAL model on iteration {i+1}: {e}")

        try:
            # Evaluate baseline
            results_base = model_base.val(data=temp_yaml, device=DEVICE, verbose=False, plots=False)
            score_base = extract_map50_from_results(results_base)
        except Exception as e:
            os.remove(temp_yaml)
            os.remove(temp_txt)
            raise RuntimeError(f"Error while validating Baseline model on iteration {i+1}: {e}")

        # remove temp files
        os.remove(temp_yaml)
        os.remove(temp_txt)

        deal_scores.append(float(score_deal))
        base_scores.append(float(score_base))

        records.append({
            'iter': i + 1,
            'deal_map50': float(score_deal),
            'base_map50': float(score_base),
            'diff': float(score_deal) - float(score_base)
        })

    # Save results as CSV for later inspection
    df = pd.DataFrame(records)
    df.to_csv('bootstrap_map50_results.csv', index=False)
    return np.array(deal_scores), np.array(base_scores), df


# --- Execute and Stats ---
if __name__ == '__main__':
    deal_scores, base_scores, df = run_bootstrap_test()
    diffs = deal_scores - base_scores
    n = len(diffs)

    print("\nSummary statistics:")
    print(f"Deal-YOLO mean mAP@0.5: {deal_scores.mean():.6f}")
    print(f"Baseline mean mAP@0.5:   {base_scores.mean():.6f}")
    print(f"Mean difference (deal - base): {diffs.mean():.6f}")
    print(f"Std of differences: {diffs.std(ddof=1):.6f}")

    # 1. Normality check of differences (Shapiro)
    try:
        _, p_norm_diff = stats.shapiro(diffs)
    except Exception as e:
        p_norm_diff = np.nan
        print("Shapiro-Wilk failed:", e)

    print(f"Shapiro-Wilk p-value for differences: {p_norm_diff:.4f}")

    # 2. Paired t-test
    t_stat, p_ttest = stats.ttest_rel(deal_scores, base_scores)
    print(f"\nPaired t-test: t={t_stat:.4f}, p={p_ttest:.4e}")

    # 3. Wilcoxon signed-rank (handle ties / zeros robustly)
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(deal_scores, base_scores, zero_method='wilcox', alternative='two-sided')
    except TypeError:
        # older scipy versions may not accept 'alternative' arg
        w_stat, p_wilcoxon = stats.wilcoxon(deal_scores, base_scores, zero_method='wilcox')
    except Exception as e:
        w_stat, p_wilcoxon = np.nan, np.nan
        print("Wilcoxon failed:", e)

    print(f"\nWilcoxon signed-rank: W={w_stat}, p={p_wilcoxon:.4e}")

    # 4. 95% CI for mean difference using t-interval
    mean_diff = diffs.mean()
    stderr = diffs.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - 0.025, df=n - 1)
    ci_lower, ci_upper = mean_diff - t_crit * stderr, mean_diff + t_crit * stderr
    print(f"\n95% CI (t-interval) for mean difference: [{ci_lower:.6f}, {ci_upper:.6f}]")

    # 5. Bootstrap CI for mean difference (robust)
    rng = np.random.RandomState(RNG_SEED)
    bs_means = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(BOOTSTRAP_CI_ITERS)])
    bs_ci = np.percentile(bs_means, [2.5, 97.5])
    print(f"95% Bootstrap CI for mean difference: [{bs_ci[0]:.6f}, {bs_ci[1]:.6f}]")

    # 6. Effect size (Cohen's d for paired samples)
    sd_diff = diffs.std(ddof=1)
    if sd_diff == 0:
        cohens_d = np.nan
    else:
        cohens_d = mean_diff / sd_diff
    print(f"\nCohen's d (paired): {cohens_d:.4f}")

    # Final verdict
    alpha = 0.05
    print("\nFinal decision (alpha = 0.05):")
    if not np.isnan(p_wilcoxon) and p_wilcoxon < alpha:
        print(f"✅ Significant difference (Wilcoxon p = {p_wilcoxon:.4e})")
    elif p_ttest < alpha:
        print(f"✅ Significant difference (paired t-test p = {p_ttest:.4e})")
    else:
        print("❌ No significant difference detected by either test.")

    # Optional: plot distribution of differences
    plt.figure(figsize=(6,4))
    plt.hist(diffs, bins=min(15, n), edgecolor='k')
    plt.axvline(mean_diff, linestyle='--', label=f"mean diff = {mean_diff:.4f}")
    plt.title("Distribution of per-iteration mAP@0.5 differences (deal - baseline)")
    plt.xlabel("mAP@0.5 difference")
    plt.legend()
    plt.tight_layout()
    plt.savefig('diffs_histogram.png')
    print("\nSaved: bootstrap_map50_results.csv and diffs_histogram.png")
