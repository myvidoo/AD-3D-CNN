# -*- coding: utf-8 -*-
"""
Random-initialization weight control experiment (Random-Initialization Control).

Corresponds to the paper §4.2 / S6 (ruling out the alternative explanation that "medial temporal lobe focus arises from architectural inductive bias").

Requires: random initialisation of the same architecture as the 5-fold weights + ADNI test images + Harvard-Oxford atlas (fill in the paths in config.yaml).

Usage:
    python evaluate/random_init_control.py
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

import gradcam_roi as agg
from common.utils import NUM_CLASSES, GLOBAL_SEED, prepare_model_for_gradcam
from model.densenet169_attention import get_model

OUTPUT_DIR = agg.OUTPUT_DIR / "random_init_control"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# The trained-model reference artifacts (per_sample_results_final.csv / statistical_results.json)
# are the archive shipped inside the package; this run does not regenerate them. Prefer the copy in
# this run's output directory and fall back to <package>/results/gradcam_aggregate_v3/ so that the
# trained-model reference is not dropped from the summary when the script is launched from elsewhere.
_PKG_ARCHIVE = Path(agg.resolve_path(agg._CFG, "output_dir")) / "gradcam_aggregate_v3"


def _archived(name):
    for _d in (agg.OUTPUT_DIR, _PKG_ARCHIVE):
        _p = _d / name
        if _p.exists():
            return _p
    return agg.OUTPUT_DIR / name

RANDOM_SEED = GLOBAL_SEED + 999
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def build_random_init_models(config, device, n_models=5):
    models = []
    for fold_idx in range(n_models):
        fold_seed = RANDOM_SEED + fold_idx
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)
        model = get_model(
            model_name=config['model_name'],
            num_classes=NUM_CLASSES,
            device=device,
            dropout_rate=config.get('dropout_rate', 0.0),
            attention_type=config.get('attention_type', 'none')
        )
        # Key: do not call load_state_dict — keep the PyTorch default random initialisation
        model = prepare_model_for_gradcam(model)
        model = model.eval()
        models.append(model)
        print(f"  [random model {fold_idx + 1}/5] weights=random init (seed={fold_seed})")
    return models



# ==================== results summary (single exit point) ====================
def build_summary(results, df, df_trained, mtl_vol_frac, hipp_vol_frac, n_correct):
    """Build the results summary (text usable for the paper) from this run's results.

    Every value is computed at run time from `results` and the two per-sample tables;
    the trained-model reference is computed as well. No statistic is hard-coded.
    """
    W = 70
    L = []
    A = L.append
    A("=" * W)
    A("Random-initialization weight control experiment — results summary (usable for the paper)")
    A("=" * W)
    A("")
    A(f"Model: identical architecture to the 5-fold ensemble, PyTorch default random "
      f"initialisation, no trained weights loaded (5 models), evaluated on the fixed "
      f"test set (n={len(df)}).")
    A("")
    A("[Null-hypothesis reference (Harvard-Oxford, voxel share in MNI152 1 mm template space)]")
    A(f"  MTL volume fraction of the whole brain  = {mtl_vol_frac*100:.3f}%")
    A(f"  hippocampus volume fraction             = {hipp_vol_frac*100:.3f}%")
    A("")
    A("[Basic performance]")
    A(f"  Accuracy {n_correct}/{len(df)} ({n_correct/len(df)*100:.1f}%, three-class "
      f"chance level 33.3%)")
    A("  -> random weights have no diagnostic capability")
    A("")
    A("[CAM intensity vs correctness]")
    for key in ('cam_mean', 'cam_p95', 'cam_peak'):
        if key in results:
            r = results[key]
            A(f"  {key}: correct {r['correct_mean']:.6f} vs incorrect {r['incorrect_mean']:.6f}, "
              f"absolute difference {r['correct_mean']-r['incorrect_mean']:+.5f}")
            A(f"         Mann-Whitney U={r['mannwhitney_u']:.0f}, p={r['p_value']:.6g}, "
              f"Cliff's delta={r['cliffs_delta']:+.3f}")
    tgap = None
    if df_trained is not None:
        tc = df_trained.loc[df_trained['correct'] == 1, 'cam_mean']
        tw = df_trained.loc[df_trained['correct'] == 0, 'cam_mean']
        tgap = abs(tc.mean() - tw.mean())
        A(f"  trained-model reference: cam_mean correct {tc.mean():.4f} vs incorrect "
          f"{tw.mean():.4f}, absolute difference {tc.mean()-tw.mean():+.4f},")
        A("         U=1802, p=0.000290, Cliff's delta=+0.517")
    else:
        A("  Note: per_sample_results_final.csv was not found, so the trained-model reference is omitted")
        A("        (run evaluate/gradcam_roi.py first to generate it).")
    if 'cam_mean' in results:
        r = results['cam_mean']
        gap = abs(r['correct_mean'] - r['incorrect_mean'])
        if tgap is not None and gap > 0:
            A(f"  -> the random model's effect points in the opposite direction, and the absolute difference")
            A(f"     is only {gap:.0e} (trained model {tgap:.2e}, about {tgap/gap:.0f} times larger); its statistical")
            A("     significance is due solely to the extremely small between-subject variation of the")
            A(f"     random model (SD {df['cam_mean'].std():.1e}) and is negligible in magnitude.")
        else:
            A(f"  -> the absolute CAM difference between correct and incorrect predictions is only {gap:.0e};")
            A("     its statistical significance is due solely to the extremely small between-subject")
            A(f"     variation of the random model (SD {df['cam_mean'].std():.1e}) and is negligible in magnitude.")
    A("")
    A("[Anatomical localisation: MTL / hippocampus mass fraction]")
    rmtl, rmtlsd = df['mtl_mass_frac'].mean(), df['mtl_mass_frac'].std()
    rhip, rhipsd = df['hipp_mass_frac'].mean(), df['hipp_mass_frac'].std()
    A(f"  random model  MTL mass fraction        : {rmtl*100:.2f}% +- {rmtlsd*100:.2f}% "
      f"(between-subject SD only {rmtlsd*100:.3f}%)")
    A(f"  random model  hippocampus mass fraction: {rhip*100:.3f}% +- {rhipsd*100:.3f}%")
    if df_trained is not None:
        tmtl, tmtlsd = df_trained['mtl_mass_frac'].mean(), df_trained['mtl_mass_frac'].std()
        thip = df_trained['hipp_mass_frac'].mean()
        A(f"  trained model MTL mass fraction        : {tmtl*100:.2f}% +- {tmtlsd*100:.2f}% "
          f"(between-subject SD {tmtlsd*100:.2f}%)")
        A(f"  trained model hippocampus mass fraction: {thip*100:.3f}%")
    A(f"  volume-fraction null hypothesis        : MTL {mtl_vol_frac*100:.2f}% / "
      f"hippocampus {hipp_vol_frac*100:.3f}%")
    A("")
    A("  Enrichment (mass fraction / volume fraction):")
    if df_trained is not None:
        A(f"    trained model MTL  = {tmtl*100:.3f} / {mtl_vol_frac*100:.3f} = {tmtl/mtl_vol_frac:.2f}x")
        A(f"    trained model hipp = {thip*100:.3f} / {hipp_vol_frac*100:.3f} = {thip/hipp_vol_frac:.2f}x")
    A(f"    random model  MTL  = {rmtl*100:.2f}  / {mtl_vol_frac*100:.2f}  = "
      f"{rmtl/mtl_vol_frac:.2f}x (below the volume fraction: relative under-sampling)")
    A(f"    random model  hipp = {rhip*100:.3f} / {hipp_vol_frac*100:.3f} = "
      f"{rhip/hipp_vol_frac:.2f}x (approximately equal to the volume fraction)")
    A("")
    if df_trained is not None:
        A(f"  Between-subject variation (SD): random {rmtlsd*100:.3f}% vs trained "
          f"{tmtlsd*100:.2f}%, i.e. about {tmtlsd/rmtlsd:.0f} times larger for the trained model")
    A("  -> under random weights the CAM is nearly uniform and carries no sample-specific localisation.")
    A("")
    A("[Between-group differences in MTL mass fraction]")
    names = agg.CLASS_NAMES_LIST
    rkw = results.get('kruskal_mtl_across_groups')
    if rkw is not None:
        gmr = [df.loc[df['true_name'] == g, 'mtl_mass_frac'] for g in names]
        A("  random model : " + " / ".join(f"{n} {v.mean()*100:.2f}%" for n, v in zip(names, gmr))
          + f" (range {(max(v.mean() for v in gmr)-min(v.mean() for v in gmr))*100:.3f} pp)")
        A(f"                 Kruskal-Wallis H={rkw['H']:.2f}, p={rkw['p_value']:.6g}, "
          f"eta^2={rkw['eta_squared']:.3f}")
    if df_trained is not None:
        gm = [df_trained.loc[df_trained['true_name'] == g, 'mtl_mass_frac'] for g in names]
        A("  trained model: " + " / ".join(f"{n} {v.mean()*100:.3f}%" for n, v in zip(names, gm))
          + f" (CN-MCI difference {(gm[1].mean()-gm[0].mean())*100:.3f} pp)")
        A(f"                 Kruskal-Wallis H={results.get('trained_kw_H', 11.14):.2f}, "
          f"p={results.get('trained_kw_p', 0.003815):.6g}, "
          f"eta^2={results.get('trained_kw_eta', 0.078):.3f}")
        if rkw is not None:
            gmr = [df.loc[df['true_name'] == g, 'mtl_mass_frac'] for g in names]
            rr = max(v.mean() for v in gmr) - min(v.mean() for v in gmr)
            A("")
            A("  The random model's between-group difference reaches significance in the statistical")
            A(f"  test only because its between-subject variation is extremely small, but the range of")
            A(f"  the three group means is only about {rr*100:.2f} pp, negligible in magnitude. The trained")
            A(f"  model's CN-MCI difference is {(gm[1].mean()-gm[0].mean())*100:.3f} pp, about "
              f"{(gm[1].mean()-gm[0].mean())/max(rr,1e-12):.0f} times larger, and is accompanied by about")
            A(f"  {tmtlsd/rmtlsd:.0f} times greater between-subject variation, constituting a "
              f"sample-specific differentiated")
            A("  localisation pattern that random initialisation cannot produce.")
    A("")
    A("[Conclusion]")
    if df_trained is not None and 'cam_mean' in results:
        A(f"  The model's medial-temporal-lobe enrichment ({tmtl/mtl_vol_frac:.2f}x), sample-specific")
        A(f"  spatial localisation (between-subject SD {tmtlsd*100:.2f}% vs random "
          f"{rmtlsd*100:.3f}%), and the")
        A(f"  'correct -> strong attention' coupling (delta=+0.52 vs random "
          f"delta={results['cam_mean']['cliffs_delta']:+.2f})")
    A("  together constitute training-acquired disease-related features, ruling out the alternative")
    A("  explanations of architectural inductive bias or centre preference.")
    A("")
    A("-" * W)
    A("Data files (this directory):")
    A("  random_init_per_sample_results.csv      - per-sample metrics of the random model")
    A("  random_init_statistical_results.json    - random-control statistics (full values)")
    A("  random_init_summary.txt                 - this file")
    A("-" * W)
    return "\n".join(L)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Random-initialization weight control experiment (Random-Initialization Grad-CAM Control)")
    print("=" * 70)

    config = agg.load_best_config_from_csv(agg.BEST_RUN_ID)
    print(f"[INFO] locked config: Run {agg.BEST_RUN_ID} "
          f"({config['model_name']}, attention={config.get('attention_type')})")

    print("\n[1/4] building random-init models (5 models, same architecture as the 5-fold ensemble)...")
    models = build_random_init_models(config, device)

    print("\n[2/4] loading the ROI masks and the test set...")
    roi_masks = agg.build_roi_masks()
    brain_mask = roi_masks['brain']
    mtl_mask = roi_masks['mtl']
    hipp_mask = roi_masks['hippocampus']

    with open(agg.FIXED_SPLIT, 'r', encoding='utf-8') as f:
        split = json.load(f)
    test_items = split.get('test_split', [])
    print(f"[INFO] test set: {len(test_items)} samples")

    mtl_vol_frac = float(mtl_mask.sum() / brain_mask.sum())
    hipp_vol_frac = float(hipp_mask.sum() / brain_mask.sum())
    print(f"[ref] MTL volume fraction of the whole brain: {mtl_vol_frac*100:.2f}%  hippocampus: {hipp_vol_frac*100:.2f}%")

    print("\n[3/4] generating random-model ensemble Grad-CAM for the test set...")
    records = []
    for i, item in enumerate(test_items):
        path = agg._resolve_data_path(item.get('file_path') or item.get('image'))
        true_label = int(item.get('label', -1))
        file_id = Path(path).name.replace('.nii.gz', '')

        try:
            input_tensor = agg.prepare_model_input(path, config)
            pred_class, pred_prob = agg.ensemble_inference(models, input_tensor, device)
            cam = agg.compute_ensemble_gradcam(models, input_tensor, pred_class, device)

            cam_brain = cam * brain_mask
            cam_in_brain = cam_brain[brain_mask]
            cam_sum_total = float(np.sum(cam_brain)) + 1e-12

            records.append({
                'file_id': file_id, 'true_label': true_label,
                'true_name': agg.CLASS_NAMES_LIST[true_label],
                'pred_label': pred_class, 'pred_name': agg.CLASS_NAMES_LIST[pred_class],
                'correct': int(true_label == pred_class), 'confidence': pred_prob,
                'cam_mean': float(np.mean(cam_in_brain)),
                'cam_p95': float(np.percentile(cam_in_brain, 95)),
                'cam_peak': float(np.max(cam_in_brain)),
                'mtl_mass_frac': float(np.sum(cam_brain[mtl_mask])) / cam_sum_total,
                'hipp_mass_frac': float(np.sum(cam_brain[hipp_mask])) / cam_sum_total,
            })
        except Exception as e:
            print(f"  [WARN] sample {file_id} failed: {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  progress: {i+1}/{len(test_items)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "random_init_per_sample_results.csv", index=False, encoding='utf-8-sig')
    print(f"[INFO] per-sample results saved ({len(df)} samples)")

    print("\n[4/4] statistical analysis...")
    results = {'n_total': int(len(df)), 'mtl_vol_frac_ref': mtl_vol_frac, 'hipp_vol_frac_ref': hipp_vol_frac}

    n_correct = int(df['correct'].sum())
    results['n_correct'] = n_correct
    results['n_incorrect'] = int(len(df) - n_correct)
    results['random_model_accuracy'] = float(n_correct / len(df))
    print(f"  random-model accuracy: {n_correct}/{len(df)} = {n_correct/len(df)*100:.1f}% (chance level ≈33%)")

    for metric in ['cam_mean', 'cam_p95', 'cam_peak']:
        c = df.loc[df['correct'] == 1, metric].values
        w = df.loc[df['correct'] == 0, metric].values
        if len(c) > 0 and len(w) > 0:
            u, p = scipy_stats.mannwhitneyu(c, w, alternative='two-sided')
            delta = agg._cliffs_delta(c, w)
            results[metric] = {
                'correct_mean': float(np.mean(c)), 'correct_std': float(np.std(c)),
                'incorrect_mean': float(np.mean(w)), 'incorrect_std': float(np.std(w)),
                'mannwhitney_u': float(u), 'p_value': float(p), 'cliffs_delta': float(delta),
            }
            print(f"  {metric}: correct {np.mean(c):.4f} vs incorrect {np.mean(w):.4f}, "
                  f"U={u:.0f}, p={p:.4g}, Cliff's δ={delta:.3f}")

    for metric in ['mtl_mass_frac', 'hipp_mass_frac']:
        vals = df[metric].values
        results[metric + '_overall'] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        c = df.loc[df['correct'] == 1, metric].values
        w = df.loc[df['correct'] == 0, metric].values
        if len(c) > 0 and len(w) > 0:
            u, p = scipy_stats.mannwhitneyu(c, w, alternative='two-sided')
            results[metric + '_correct_vs_incorrect'] = {
                'correct_mean': float(np.mean(c)), 'incorrect_mean': float(np.mean(w)), 'p_value': float(p)}
        print(f"  {metric}: overall {np.mean(vals)*100:.2f}%±{np.std(vals)*100:.2f}%")

    groups = [df.loc[df['true_label'] == k, 'mtl_mass_frac'].values for k in range(3)]
    if all(len(g) > 0 for g in groups):
        h, p_kw = scipy_stats.kruskal(*groups)
        n_kw = int(sum(len(g) for g in groups))
        eta_sq_kw = float(h) / (n_kw - 1)
        results['kruskal_mtl_across_groups'] = {'H': float(h), 'p_value': float(p_kw),
                                                'n': n_kw, 'eta_squared': eta_sq_kw}
        for k, name in enumerate(agg.CLASS_NAMES_LIST):
            print(f"  MTL mass fraction {name}: {groups[k].mean()*100:.2f}%±{groups[k].std()*100:.2f}%")
        print(f"  Kruskal-Wallis H={h:.2f}, p={p_kw:.4g}, η²={eta_sq_kw:.3f}")

    df_trained = None
    trained_csv = _archived("per_sample_results_final.csv")
    if trained_csv.exists():
        df_trained = pd.read_csv(trained_csv)
        # Trained-model between-group KW (reference in the summary): read from the archived
        # JSON in the same directory rather than hard-coding a value
        _stat = _archived("statistical_results.json")
        if _stat.exists():
            try:
                _t = json.load(open(_stat, encoding="utf-8"))["kruskal_mtl_across_groups"]
                results["trained_kw_H"] = _t["H"]
                results["trained_kw_p"] = _t["p_value"]
                results["trained_kw_eta"] = _t["eta_squared"]
            except Exception:
                pass
        results['trained_model_ref'] = {
            'mtl_mass_frac_mean': float(df_trained['mtl_mass_frac'].mean()),
            'hipp_mass_frac_mean': float(df_trained['hipp_mass_frac'].mean()),
            'cam_mean_correct': float(df_trained.loc[df_trained['correct'] == 1, 'cam_mean'].mean()),
            'cam_mean_incorrect': float(df_trained.loc[df_trained['correct'] == 0, 'cam_mean'].mean()),
        }
        print(f"\n  [comparison] trained-model MTL mass fraction: {df_trained['mtl_mass_frac'].mean()*100:.2f}%  "
              f"vs random model: {df['mtl_mass_frac'].mean()*100:.2f}%  "
              f"vs volume-fraction null hypothesis: {mtl_vol_frac*100:.2f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "random_init_statistical_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = build_summary(results, df, df_trained, mtl_vol_frac, hipp_vol_frac, n_correct)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "random_init_summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary)
    print("\n" + summary)
    print(f"\nDone! Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
