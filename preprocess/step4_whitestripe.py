# -*- coding: utf-8 -*-
# ============================================================================
# step4_whitestripe.py — paper 2.2/S1.7 intensity normalisation (WhiteStripe-inspired histogram-peak method)
#
# Source: paper workspace 01_data_and_preprocessing_.../2.1.1data_processing_whitestripe_normalisation_optimisation.ipynb  (translated from the authors' archive path)
#       (the algorithm was reorganised verbatim into a command-line script; the core
#        functions match the notebook line by line).
#
# Algorithm (S1.7): voxels in the central 50% region avoid edge interpolation artefacts →
#   the 20%-99% quantile voxels exclude CSF → 200-bin histogram + Gaussian smoothing
#   (sigma=2) → the tallest peak is the white-matter peak → the mean of the white-matter
#   stripe at peak ±10% is mapped linearly to 1000 → clip to [0, 3000].
#
# Input: registered (MNI152 space) NIfTI; output: the same directory structure, with the _ws suffix appended to the file name.
# Resume: subjects whose *_ws.nii.gz already exists are skipped automatically.
#
# Usage:
#   python step4_whitestripe.py --input <registered directory> --output <output directory>
#   python step4_whitestripe.py --input ... --output ... --max-files 2   # smoke test
#   python step4_whitestripe.py --evaluate --input <normalised directory>  # quality assessment
# ============================================================================
import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def white_stripe_normalization(img_data, mask_data=None, stripe_width=0.1):
    """
    WhiteStripe normalisation (version optimised for already-registered data)
    Goal: map the brain white-matter peak to 1000
    """
    # 1. Obtain the brain-tissue voxels used for the statistics
    if mask_data is not None:
        brain_voxels = img_data[mask_data > 0]
    else:
        # [Optimisation for registered data]: use the centre-crop method
        d, h, w = img_data.shape
        center_crop = img_data[d//4:3*d//4, h//4:3*h//4, w//4:3*w//4]
        brain_voxels = center_crop.flatten()

    # 2. Quantile thresholds: 20% - 99% (actively filter out dark CSF and background, focusing only on gray matter + white matter)
    p_low = np.percentile(brain_voxels, 20)
    p_high = np.percentile(brain_voxels, 99)
    tissue_voxels = brain_voxels[(brain_voxels >= p_low) & (brain_voxels <= p_high)]

    if len(tissue_voxels) < 100:
        tissue_voxels = brain_voxels[brain_voxels <= p_high]

    # 3-5. Histogram estimation, smoothing, peak detection
    hist, bin_edges = np.histogram(tissue_voxels, bins=200, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_smooth = gaussian_filter(hist, sigma=2)
    peaks, properties = find_peaks(hist_smooth,
                                   height=0.1 * hist_smooth.max(),
                                   distance=20)

    if len(peaks) == 0:
        white_matter_peak = np.percentile(tissue_voxels, 85)
    else:
        max_peak_idx = peaks[np.argmax(properties['peak_heights'])]
        white_matter_peak = bin_centers[max_peak_idx]
        if white_matter_peak < np.percentile(tissue_voxels, 40):
            white_matter_peak = np.percentile(tissue_voxels, 85)

    # 6. Mean within the white stripe
    stripe_lower = white_matter_peak * (1 - stripe_width)
    stripe_upper = white_matter_peak * (1 + stripe_width)

    if mask_data is not None:
        stripe_voxels = img_data[(img_data >= stripe_lower) &
                                 (img_data <= stripe_upper) &
                                 (mask_data > 0)]
    else:
        d, h, w = img_data.shape
        center_mask = np.zeros_like(img_data, dtype=bool)
        center_mask[d//4:3*d//4, h//4:3*h//4, w//4:3*w//4] = True
        stripe_voxels = img_data[(img_data >= stripe_lower) &
                                 (img_data <= stripe_upper) &
                                 center_mask]

    if len(stripe_voxels) < 100:
        white_matter_mean = white_matter_peak
    else:
        white_matter_mean = np.mean(stripe_voxels)

    # 7. Normalisation: map the white-matter mean to 1000; 8. clip to [0, 3000]
    if white_matter_mean > 0:
        normalized_data = img_data * (1000.0 / white_matter_mean)
    else:
        normalized_data = img_data
    normalized_data = np.clip(normalized_data, 0, 3000)

    return normalized_data, white_matter_mean


def find_mri_files(input_folder):
    """Recursively find .nii.gz files, skipping files that already carry the normalisation marker"""
    mri_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.nii.gz'):
                if '_ws' not in file and '_whitestripe' not in file:
                    mri_files.append(os.path.join(root, file))
    return mri_files


def batch_white_stripe_normalization(input_root, output_root, max_files=None):
    """Main batch-normalisation function (supports resuming)"""
    os.makedirs(output_root, exist_ok=True)

    print(f"Scanning folder: {input_root}")
    mri_files = find_mri_files(input_root)
    if max_files is not None:
        mri_files = mri_files[:max_files]
    print(f"Found {len(mri_files)} .nii.gz files to process")

    if len(mri_files) == 0:
        print("No files to process found!")
        return

    stats = {'total': len(mri_files), 'success': 0, 'failed': 0, 'skipped': 0, 'wm_vals': []}
    wm_log_path = os.path.join(output_root, "whitestripe_wm_values.csv")
    wm_log = []

    for file_path in tqdm(mri_files, desc="WhiteStripe normalisation"):
        try:
            rel_path = os.path.relpath(file_path, input_root)
            if rel_path == '.':
                output_subdir = output_root
            else:
                output_subdir = os.path.join(output_root, os.path.dirname(rel_path))
            file_name = os.path.basename(file_path)
            os.makedirs(output_subdir, exist_ok=True)

            base_name = file_name.replace('.nii.gz', '')
            output_path = os.path.join(output_subdir, f"{base_name}_ws.nii.gz")

            if os.path.exists(output_path):        # resume
                stats['skipped'] += 1
                continue

            img = nib.load(file_path)
            img_data = img.get_fdata()
            normalized_data, wm_value = white_stripe_normalization(img_data, mask_data=None)
            stats['wm_vals'].append(wm_value)
            wm_log.append((os.path.basename(file_path), wm_value))

            normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)
            nib.save(normalized_img, output_path)
            stats['success'] += 1
        except Exception as e:
            print(f"\nProcessing failed: {file_path} -> {e}")
            stats['failed'] += 1

    print("\n" + "=" * 60)
    print("Processing complete! Statistics report:")
    print(f"  Total files: {stats['total']}  succeeded: {stats['success']}  "
          f"skipped: {stats['skipped']}  failed: {stats['failed']}")
    if stats['wm_vals']:
        print(f"[White-matter intensity statistics] (raw values before normalisation) mean {np.mean(stats['wm_vals']):.2f} "
              f"± {np.std(stats['wm_vals']):.2f}, range "
              f"[{np.min(stats['wm_vals']):.2f}, {np.max(stats['wm_vals']):.2f}]"
              f" (white-matter peak after normalisation ≈ 1000)")
    print("=" * 60)

    if wm_log:
        with open(wm_log_path, "w", encoding="utf-8") as f:
            f.write("file,wm_mean_before_norm\n")
            for name, v in wm_log:
                f.write(f"{name},{v:.4f}\n")
        print(f"Per-subject white-matter means saved: {wm_log_path}")


def evaluate_normalized(folder, n_samples=50, fig_out=None):
    """Quantitative assessment: white-matter peak alignment (target 1000)"""
    import glob
    file_list = glob.glob(os.path.join(folder, "**", "*.nii.gz"), recursive=True)
    if not file_list:
        print("No files found!")
        return
    print(f"Scanned {len(file_list)} files; assessing the first {n_samples}...")

    wm_peaks = []
    for file_path in file_list[:n_samples]:
        try:
            data = nib.load(file_path).get_fdata()
            voxels = get_brain_voxels(data)
            peak = find_white_matter_peak(voxels)
            if peak:
                wm_peaks.append(peak)
        except Exception:
            continue

    if wm_peaks:
        mean_peak, std_peak = np.mean(wm_peaks), np.std(wm_peaks)
        print("=" * 60)
        print(f"White-matter peak statistics (n={len(wm_peaks)}): mean {mean_peak:.2f} (target 1000) ± {std_peak:.2f}")
        print("  Assessment passed: white-matter peaks are well aligned" if abs(mean_peak - 1000) < 50
              else "  Warning: white-matter peaks deviate considerably from the target value")
        print("=" * 60)


def get_brain_voxels(img_data):
    """Extract brain-tissue voxels (centre-crop method + simple background removal)"""
    d, h, w = img_data.shape
    center_crop = img_data[d//4:3*d//4, h//4:3*h//4, w//4:3*w//4]
    return center_crop[center_crop > np.percentile(center_crop, 10)]


def find_white_matter_peak(voxels):
    """Find the white-matter peak (consistent with the internal logic of the normalisation)"""
    if len(voxels) == 0:
        return None
    p_low, p_high = np.percentile(voxels, 20), np.percentile(voxels, 99)
    tissue_voxels = voxels[(voxels >= p_low) & (voxels <= p_high)]
    if len(tissue_voxels) < 100:
        return np.percentile(voxels, 85)
    hist, bin_edges = np.histogram(tissue_voxels, bins=200, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_smooth = gaussian_filter(hist, sigma=2)
    peaks, properties = find_peaks(hist_smooth, height=0.1 * hist_smooth.max(), distance=20)
    if len(peaks) == 0:
        return np.percentile(tissue_voxels, 85)
    max_peak_idx = peaks[np.argmax(properties['peak_heights'])]
    return bin_centers[max_peak_idx]


def main():
    ap = argparse.ArgumentParser(description="WhiteStripe intensity normalisation (paper S1.7)")
    ap.add_argument("--input", help="input directory (registered NIfTI)")
    ap.add_argument("--output", help="output directory (with _ws appended to file names)")
    ap.add_argument("--max-files", type=int, default=None, help="limit the number of files processed (smoke test)")
    ap.add_argument("--evaluate", action="store_true", help="assessment mode: measure white-matter peak alignment on already-normalised data")
    args = ap.parse_args()

    if args.evaluate:
        if not args.input:
            ap.error("--evaluate requires --input to point to a normalised directory")
        evaluate_normalized(args.input)
        return
    if not (args.input and args.output):
        ap.error("--input and --output are required (or use --evaluate)")
    batch_white_stripe_normalization(args.input, args.output, max_files=args.max_files)


if __name__ == "__main__":
    main()
