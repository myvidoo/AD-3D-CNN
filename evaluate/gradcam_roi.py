# -*- coding: utf-8 -*-
"""
Aggregate Grad-CAM statistical analysis of the confusion-matrix structure for the 5-fold ensemble (ROI overlap + group statistics).

Corresponds to the paper §2.5 / §3.3 / §4.2 / S6 (correct vs incorrect CAM intensity, MTL mass fraction,
Harvard-Oxford ROI overlap, Spearman correlation, Kruskal-Wallis).

Requires: 5-fold weights + ADNI test images + Harvard-Oxford atlas + MNI template (fill in the paths in config.yaml).

Usage:
    python evaluate/gradcam_roi.py

This module is also reused by evaluate/random_init_control.py and evaluate/gradcam_quadgrid.py.
"""

import os
import gc
import json
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import ants
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# Repository root (so the script can be run from any working directory)
REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT))

from config import load_config, resolve_path
from common.utils import (
    NUM_CLASSES, GLOBAL_SEED, get_data_transforms, parse_config_from_string,
    prepare_model_for_gradcam,
)
from model.densenet169_attention import get_model

# ==================== module-level configuration (read from config.yaml) ====================
_CFG = load_config()
BEST_RUN_ID = _CFG["best_run_id"]
DATA_ROOT = _CFG["data"].get("data_root") or None
GRID_RESULTS = resolve_path(_CFG, "data.grid_results")
FIXED_SPLIT = resolve_path(_CFG, "data.fixed_split")
WEIGHTS_DIR = resolve_path(_CFG, "weights_dir")
OUTPUT_DIR = Path(_CFG["output_dir"]) / "gradcam_aggregate_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATLAS_SUB_PATH = _CFG["atlas"].get("harvard_oxford_sub") or ""
ATLAS_CORT_PATH = _CFG["atlas"].get("harvard_oxford_cort") or ""
TEMPLATE_PATH = _CFG["atlas"].get("mni_template") or ""

CLASS_NAMES_LIST = ['CN', 'MCI', 'AD']

RANDOM_SEED = GLOBAL_SEED + 5
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# [IMPORTANT · CORRECTED] The label **value** in the Harvard-Oxford atlas NIfTI is 1 larger than the index in the same-named XML:
# In the XML, index 0 = "Left Cerebral White Matter", whereas in the image value 0 is background and only value 1 is
# "Left Cerebral White Matter". Therefore image value = XML index + 1.
# The values below have been corrected accordingly, and each structure's MNI centroid was checked anatomically one by one (21/21 hits);
# e.g. value 8, centroid (0.6,-31.0,-34.2), is the brainstem; value 9, (-24.8,-22.2,-14.4), is the left hippocampus.
#   Subcortical: left/right hippocampus = 9/19, left/right amygdala = 10/20
#   Cortical  : anterior parahippocampal gyrus = 34, posterior = 35, temporal pole = 8
# Historical defect: before correction, 8/18 was in fact "brainstem + right globus pallidus", 9/19 was in fact "bilateral hippocampus", and
#           cortical 33/34 was in fact "frontal orbital cortex + anterior parahippocampal gyrus", which misaligned the ROI values anatomically.
SUB_LABELS = {
    'Left Hippocampus': 9, 'Right Hippocampus': 19,
    'Left Amygdala': 10, 'Right Amygdala': 20,
}
CORT_LABELS = {
    'Parahippocampal Gyrus, anterior division': 34,
    'Parahippocampal Gyrus, posterior division': 35,
    'Temporal Pole': 8,
}

CACHE_FILE = OUTPUT_DIR / "per_sample_metrics_cache.json"


def _resolve_data_path(p):
    """Join the path-stripped file name back onto data_root (when it is a relative path)."""
    if DATA_ROOT and not os.path.isabs(p):
        return os.path.join(DATA_ROOT, p)
    return p


# ==================== model loading ====================
def load_best_config_from_csv(run_id):
    df = pd.read_csv(GRID_RESULTS)
    row = df[df['run_id'] == run_id]
    if row.empty:
        raise ValueError(f"run_id = {run_id} not found")
    return parse_config_from_string(row.iloc[0]['config'])


def load_ensemble_models(config, device):
    models = []
    for fold_idx in range(5):
        ckpt_path = Path(WEIGHTS_DIR) / f"fold_{fold_idx + 1}_best_geo.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {ckpt_path}")
        model = get_model(
            model_name=config['model_name'],
            num_classes=NUM_CLASSES,
            device=device,
            dropout_rate=config.get('dropout_rate', 0.0),
            attention_type=config.get('attention_type', 'none')
        )
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        if hasattr(model, 'base_model') and not any(k.startswith('base_model.') for k in state_dict.keys()):
            model.base_model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        model = prepare_model_for_gradcam(model)
        model = model.eval()
        models.append(model)
    print(f"[INFO] successfully loaded {len(models)} models.")
    return models


# ==================== ensemble inference and Grad-CAM ====================
def ensemble_inference(models, input_tensor, device):
    all_logits = []
    with torch.no_grad():
        for model in models:
            logits = model(input_tensor.to(device))
            all_logits.append(logits)
    mean_logits = torch.mean(torch.stack(all_logits), dim=0)
    probs = F.softmax(mean_logits, dim=1)
    pred_class = int(torch.argmax(probs, dim=1).item())
    pred_prob = float(probs[0, pred_class].item())
    return pred_class, pred_prob


def find_target_layer(model):
    search_model = model.base_model if hasattr(model, 'base_model') else model
    if hasattr(search_model, 'features'):
        feature_layers = list(search_model.features.children())
        if feature_layers:
            return feature_layers[-1]
    raise ValueError("Could not find the Grad-CAM target layer.")


def compute_single_gradcam(model, input_tensor, target_class, device):
    target_layer = find_target_layer(model)
    gradients, activations = [], []

    def fw_hook(m, inp, out):
        activations.append(out.detach())

    def bw_hook(m, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_fw = target_layer.register_forward_hook(fw_hook)
    handle_bw = target_layer.register_full_backward_hook(bw_hook)

    model.zero_grad()
    output = model(input_tensor.to(device))
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1.0
    output.backward(gradient=one_hot, retain_graph=True)

    handle_fw.remove()
    handle_bw.remove()

    grad = gradients[0]
    act = activations[0]
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    if cam.shape[2:] != input_tensor.shape[2:]:
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)
    return cam.squeeze().cpu().numpy()


def compute_ensemble_gradcam(models, input_tensor, target_class, device):
    all_cams = []
    for model in models:
        cam = compute_single_gradcam(model, input_tensor, target_class, device)
        all_cams.append(cam)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ensemble_cam = np.mean(all_cams, axis=0)
    return np.maximum(ensemble_cam, 0.0)


# ==================== image preprocessing ====================
def prepare_model_input(registered_path, config):
    from monai.data import Dataset
    sample = [{'image': str(registered_path), 'label': 0}]
    _, test_transforms = get_data_transforms(
        augmentation_intensity=0, a_max=config.get('a_max', 1100))
    dataset = Dataset(data=sample, transform=test_transforms)
    item = dataset[0]
    input_tensor = item['image'].unsqueeze(0)
    return input_tensor


# ==================== ROI mask construction ====================
def build_roi_masks():
    if not ATLAS_SUB_PATH or not os.path.exists(ATLAS_SUB_PATH):
        raise FileNotFoundError(f"Harvard-Oxford subcortical atlas not configured or does not exist: {ATLAS_SUB_PATH}")
    if not ATLAS_CORT_PATH or not os.path.exists(ATLAS_CORT_PATH):
        raise FileNotFoundError(f"Harvard-Oxford cortical atlas not configured or does not exist: {ATLAS_CORT_PATH}")
    if not TEMPLATE_PATH or not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f"MNI template not configured or does not exist: {TEMPLATE_PATH}")

    sub_atlas = ants.image_read(str(ATLAS_SUB_PATH))
    cort_atlas = ants.image_read(str(ATLAS_CORT_PATH))
    sub_data = sub_atlas.numpy()
    cort_data = cort_atlas.numpy()

    hipp_mask = np.isin(sub_data, [SUB_LABELS['Left Hippocampus'], SUB_LABELS['Right Hippocampus']])
    amyg_mask = np.isin(sub_data, [SUB_LABELS['Left Amygdala'], SUB_LABELS['Right Amygdala']])
    phg_mask = np.isin(cort_data, [CORT_LABELS['Parahippocampal Gyrus, anterior division'],
                                   CORT_LABELS['Parahippocampal Gyrus, posterior division']])
    mtl_mask = hipp_mask | amyg_mask | phg_mask

    template = ants.image_read(str(TEMPLATE_PATH))
    tmpl_data = template.numpy()
    brain_mask = tmpl_data > np.percentile(tmpl_data[tmpl_data > 0], 5)

    print(f"[ROI] hippocampus voxels: {hipp_mask.sum()}, amygdala: {amyg_mask.sum()}, "
          f"parahippocampal gyrus: {phg_mask.sum()}, MTL: {mtl_mask.sum()}, whole brain: {brain_mask.sum()}")

    # Anatomical self-check: prevent the label misalignment from recurring.
    # Centroids are reported in MNI millimetres; hippocampus/amygdala must lie within the medial temporal lobe (|x|∈[15,40], y<-5, z∈[-40,0]),
    # otherwise the atlas label mapping is wrong.
    # ANTs/ITK physical coordinates are LPS; convert to RAS (negate x and y) before interpreting them anatomically in MNI
    conv = np.diag([-1.0, -1.0, 1.0])
    aff = conv @ (sub_atlas.direction * sub_atlas.spacing)
    org = (conv @ np.asarray(sub_atlas.origin, dtype=float).reshape(3, 1)).ravel()
    check = {'hippocampus': hipp_mask, 'amygdala': amyg_mask, 'parahippocampal': phg_mask}
    for name, m in check.items():
        idx = np.array(np.nonzero(m)).mean(1)
        cx, cy, cz = (aff @ idx + org)
        print(f"[ROI self-check] {name}: centroid MNI=({cx:6.1f},{cy:6.1f},{cz:6.1f})")
        if name in ('hippocampus', 'amygdala'):
            assert abs(cx) < 5 or (12 <= abs(cx) <= 45), f"{name} ROI centroid x={cx:.1f} is outside the medial temporal lobe range; the atlas labels may be misaligned"
            assert cy < 0 and -45 <= cz <= 5, f"{name} ROI centroid (y={cy:.1f}, z={cz:.1f}) is outside the medial temporal lobe range; the atlas labels may be misaligned"

    return {
        'hippocampus': hipp_mask, 'amygdala': amyg_mask,
        'parahippocampal': phg_mask, 'mtl': mtl_mask, 'brain': brain_mask,
    }


# ==================== main analysis flow ====================
def run_full_analysis(config, device):
    models = load_ensemble_models(config, device)
    roi_masks = build_roi_masks()
    brain_mask = roi_masks['brain']
    mtl_mask = roi_masks['mtl']
    hipp_mask = roi_masks['hippocampus']

    with open(FIXED_SPLIT, 'r', encoding='utf-8') as f:
        split = json.load(f)
    test_items = split.get('test_split', [])
    print(f"\n[INFO] test set: {len(test_items)} samples")

    agg_cams = defaultdict(lambda: {'sum': None, 'n': 0})
    per_sample_records = []

    for i, item in enumerate(test_items):
        path = _resolve_data_path(item.get('file_path') or item.get('image'))
        true_label = int(item.get('label', -1))
        file_id = Path(path).name.replace('.nii.gz', '')

        if not os.path.exists(path):
            print(f"  [WARN] skipping non-existent path: {path}")
            continue

        try:
            input_tensor = prepare_model_input(path, config)
            pred_class, pred_prob = ensemble_inference(models, input_tensor, device)
            cam = compute_ensemble_gradcam(models, input_tensor, pred_class, device)

            cam_brain = cam * brain_mask
            cam_max = float(np.max(cam_brain)) if np.max(cam_brain) > 0 else 1.0
            cam_norm = cam_brain / cam_max

            cam_in_brain = cam_brain[brain_mask]
            cam_sum_total = float(np.sum(cam_brain)) + 1e-12
            mtl_mass = float(np.sum(cam_brain[mtl_mask])) / cam_sum_total
            hipp_mass = float(np.sum(cam_brain[hipp_mask])) / cam_sum_total

            record = {
                'file_id': file_id, 'true_label': true_label,
                'true_name': CLASS_NAMES_LIST[true_label],
                'pred_label': pred_class, 'pred_name': CLASS_NAMES_LIST[pred_class],
                'correct': int(true_label == pred_class), 'confidence': pred_prob,
                'cam_mean': float(np.mean(cam_in_brain)),
                'cam_p95': float(np.percentile(cam_in_brain, 95)),
                'cam_peak': float(np.max(cam_in_brain)),
                'mtl_mass_frac': mtl_mass, 'hipp_mass_frac': hipp_mass,
            }
            per_sample_records.append(record)

            key = (true_label, pred_class)
            if agg_cams[key]['sum'] is None:
                agg_cams[key]['sum'] = cam_norm.astype(np.float32)
            else:
                agg_cams[key]['sum'] += cam_norm.astype(np.float32)
            agg_cams[key]['n'] += 1

            status = "✓" if true_label == pred_class else "✗"
            if (i + 1) % 10 == 0 or true_label != pred_class:
                print(f"  [{i+1}/{len(test_items)}] {status} "
                      f"{CLASS_NAMES_LIST[true_label]}→{CLASS_NAMES_LIST[pred_class]} "
                      f"conf={pred_prob:.3f} MTL%={mtl_mass*100:.1f}")

        except Exception as e:
            print(f"  [ERROR] {path}: {e}")
            continue
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return per_sample_records, agg_cams, roi_masks


# ==================== statistical analysis ====================
def _cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def run_statistical_analysis(df):
    results = {}
    correct = df[df['correct'] == 1]
    incorrect = df[df['correct'] == 0]
    print(f"\n[stats] correct predictions n={len(correct)}, incorrect predictions n={len(incorrect)}")

    for metric in ['cam_mean', 'cam_p95', 'cam_peak', 'mtl_mass_frac', 'hipp_mass_frac']:
        c_vals = correct[metric].dropna().values
        i_vals = incorrect[metric].dropna().values
        if len(c_vals) > 0 and len(i_vals) > 0:
            u_stat, p_val = scipy_stats.mannwhitneyu(c_vals, i_vals, alternative='two-sided')
            cliffs_d = _cliffs_delta(c_vals, i_vals)
            results[metric] = {
                'correct_mean': float(np.mean(c_vals)), 'correct_std': float(np.std(c_vals)),
                'incorrect_mean': float(np.mean(i_vals)), 'incorrect_std': float(np.std(i_vals)),
                'mannwhitney_u': float(u_stat), 'p_value': float(p_val),
                'cliffs_delta': float(cliffs_d), 'n_correct': len(c_vals), 'n_incorrect': len(i_vals),
            }
            print(f"  {metric}: correct={np.mean(c_vals):.4f} vs incorrect={np.mean(i_vals):.4f}, "
                  f"U={u_stat:.1f}, p={p_val:.4g}, δ={cliffs_d:.3f}")

    for metric in ['cam_mean', 'cam_p95', 'mtl_mass_frac', 'hipp_mass_frac']:
        valid = df[['confidence', metric]].dropna()
        if len(valid) > 10:
            rho, p_val = scipy_stats.spearmanr(valid['confidence'], valid[metric])
            results[f'spearman_conf_vs_{metric}'] = {'rho': float(rho), 'p_value': float(p_val), 'n': len(valid)}

    for group in CLASS_NAMES_LIST:
        sub = df[df['true_name'] == group]
        results[f'mtl_frac_{group}'] = {'mean': float(sub['mtl_mass_frac'].mean()),
                                        'std': float(sub['mtl_mass_frac'].std()), 'n': len(sub)}

    groups_data = [df[df['true_name'] == g]['mtl_mass_frac'].dropna().values for g in CLASS_NAMES_LIST]
    if all(len(g) > 0 for g in groups_data):
        h_stat, p_val = scipy_stats.kruskal(*groups_data)
        n_kw = int(sum(len(g) for g in groups_data))
        eta_sq_kw = float(h_stat) / (n_kw - 1)
        results['kruskal_mtl_across_groups'] = {'H': float(h_stat), 'p_value': float(p_val),
                                                'n': n_kw, 'eta_squared': eta_sq_kw}
        print(f"  Kruskal-Wallis MTL% across the three groups: H={h_stat:.2f}, p={p_val:.4g}, η²={eta_sq_kw:.3f}")

    conf_mat = np.zeros((3, 3), dtype=int)
    for _, row in df.iterrows():
        conf_mat[int(row['true_label']), int(row['pred_label'])] += 1
    results['confusion_matrix'] = conf_mat.tolist()
    print("\n[confusion matrix] (rows = true, columns = predicted)")
    print(conf_mat)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / 'statistical_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    return results


# ==================== ROI overlap quantitative analysis ====================
def analyze_roi_overlap(agg_cams, roi_masks):
    results = {}
    mtl = roi_masks['mtl']
    hipp = roi_masks['hippocampus']

    for key, data in agg_cams.items():
        if data['n'] == 0:
            continue
        mean_cam = data['sum'] / data['n']
        tl, pl = key
        name = f"{CLASS_NAMES_LIST[tl]}2{CLASS_NAMES_LIST[pl]}"

        cam_sum = float(np.sum(mean_cam)) + 1e-12
        mtl_frac = float(np.sum(mean_cam[mtl])) / cam_sum
        hipp_frac = float(np.sum(mean_cam[hipp])) / cam_sum

        cam_bin = mean_cam > (0.5 * np.max(mean_cam))
        dice_mtl = 2.0 * np.sum(cam_bin & mtl) / (np.sum(cam_bin) + np.sum(mtl) + 1e-12)
        dice_hipp = 2.0 * np.sum(cam_bin & hipp) / (np.sum(cam_bin) + np.sum(hipp) + 1e-12)

        results[name] = {
            'n': data['n'], 'mtl_mass_frac': mtl_frac, 'hipp_mass_frac': hipp_frac,
            'dice_mtl_50pct': float(dice_mtl), 'dice_hipp_50pct': float(dice_hipp),
            'cam_peak': float(np.max(mean_cam)),
        }
        print(f"  {name} (n={data['n']}): MTL mass={mtl_frac*100:.1f}%, "
              f"hippocampus mass={hipp_frac*100:.1f}%, Dice(MTL)={dice_mtl:.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / 'roi_overlap_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


# ==================== main program ====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    config = load_best_config_from_csv(BEST_RUN_ID)
    print(f"[INFO] locked config: Run {BEST_RUN_ID} "
          f"({config['model_name']}, attention={config.get('attention_type')})")

    per_sample, agg_cams, roi_masks = run_full_analysis(config, device)

    df = pd.DataFrame(per_sample)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "per_sample_results_final.csv", index=False, encoding='utf-8-sig')
    stats_results = run_statistical_analysis(df)
    roi_results = analyze_roi_overlap(agg_cams, roi_masks)

    print(f"\nAll done! Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
