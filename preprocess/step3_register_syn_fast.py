# -*- coding: utf-8 -*-
# ============================================================================
# step3_register_syn_fast.py — paper 2.2/S1.6 spatial normalisation (ANTs SyN fast registration)
#
# Source: paper workspace 01_data_and_preprocessing_.../2.0.4registration_alignment_..._SyN_batch_processing_20260525.py  (translated from the authors' archive path)
#       (the registration script actually used for the 954 ADNI cases, strategy syn_fast). This
#       file makes only minimal changes to it: hard-coded paths become command-line arguments
#       and the interactive menu becomes direct batch execution; the registration algorithm,
#       the resuming behaviour and the output organisation are identical to the original.
#
# Pipeline: N4 bias correction → quantile histogram matching ([0.5,99.5]%, 1024 quantiles) → ANTs SyN registration
#       default strategy syn_fast: reg_iterations=(50,30,20), metric=CC, sampling=4,
#       flow_sigma=3, total_sigma=0 (the one actually used in the paper; the remaining
#       strategies such as syn_standard are kept as options).
#
# Output (under --output):
#   registered/<file_id>.nii.gz          registered image (MNI152 space)
#   warpfields/<file_id>_warp.nii.gz     forward deformation (warp) field
#   overlays/<file_id>_overlay.png       registration overlay QC image
#   reports/completed_files.txt          resuming completion list
#   reports/registration_report_*.csv    per-case Dice/CC/MAE metrics (for the S1.9 QC summary)
#   (the paper pipeline does not generate Jacobian/inverse warp fields by default; use --save-jacobian if needed)
#
# Resume: after an interruption simply re-run the same command; completed subjects are skipped
#   automatically (based on a double check of completed_files.txt + the existence of the output files).
#
# Usage example:
#   python step3_register_syn_fast.py --input <skull-stripped NIfTI directory> #       --template <MNI152 skull-stripped template mni.nii.gz> --output <output directory>
#   # smoke test (single case): --single <file-name substring>
# ============================================================================
# ================================================================
# ADNI brain MRI batch registration system - final version v3.2
# Fix: use quantile histogram matching (preserving pathological information)
# Date: 2024-06-25 the histogram-matching strategy was modified: the quantile bounds were
# changed from lower_bound[2, 98] to [0.5, 99.5], the number of sampling points n_quantiles was
# raised from 200 to 1024, and the matching quality was improved, which suits the characteristics
# of ADNI data particularly well (atrophic brain + pathological signal)               
# Date: 2024-06-25 mask=self.template_mask,  # ← add this line!
# ================================================================
import os
import ants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import time
import logging
import warnings
from scipy import ndimage
from scipy.interpolate import interp1d
from skimage.filters import threshold_otsu
import gc
import traceback
import shutil

warnings.filterwarnings('ignore')

# ============================ Logging configuration ============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'registration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================ Configuration ============================
# [AD_CNN_code change] Paths are no longer hard-coded; they are injected through the command-line arguments --input/--template/--output.
class Config:
    """Global configuration (paths injected by the CLI)"""
    INPUT_DIR = None       # --input  (skull-stripped NIfTI directory)
    TEMPLATE_PATH = None   # --template (MNI152 skull-stripped template)
    OUTPUT_DIR = None      # --output (output root directory)

    # Memory management
    GC_FREQUENCY = 20
    CLEAR_CACHE_FREQUENCY = 20

# ============================ Helper functions ============================
def safe_del(*args):
    """Safely delete variables"""
    for var in args:
        try:
            del var
        except:
            pass

def force_gc():
    """Force garbage collection"""
    gc.collect()

def check_output_exists(output_dir, file_id, required_files=None):
    """Check whether the output files already exist (resume)"""
    if required_files is None:
        required_files = ['registered']

    checks = {
        'registered': output_dir / 'registered' / f"{file_id}.nii.gz",
        'warpfield': output_dir / 'warpfields' / f"{file_id}_warp.nii.gz",
        'jacobian': output_dir / 'jacobians' / f"{file_id}_jacobian.nii.gz",
        'overlay': output_dir / 'overlays' / f"{file_id}_overlay.png",
    }

    missing = []
    for file_type in required_files:
        if file_type in checks:
            if not checks[file_type].exists():
                missing.append(file_type)

    return len(missing) == 0, missing

def create_brain_mask(img):
    """Create a brain mask"""
    try:
        data = img.numpy()
        
        # Detect whether the image has already been brain-extracted
        zero_ratio = np.mean(data == 0)
        is_brain_extracted = zero_ratio > 0.3
        
        if is_brain_extracted:
            mask_data = (data > 0).astype(np.float32)
        else:
            non_zero = data[data > 0]
            if len(non_zero) > 0:
                try:
                    thresh = threshold_otsu(non_zero)
                    mask_data = (data > thresh * 0.3).astype(np.float32)
                except:
                    thresh = np.percentile(non_zero, 25)
                    mask_data = (data > thresh).astype(np.float32)
            else:
                mask_data = np.ones_like(data, dtype=np.float32)
            
            # Morphological cleanup
            structure = np.ones((3,3,3)) if data.ndim >= 3 else np.ones((3,3))
            mask_data = ndimage.binary_opening(mask_data, structure=structure)
            mask_data = ndimage.binary_closing(mask_data, structure=structure)
            if data.ndim >= 3:
                mask_data = ndimage.binary_fill_holes(mask_data)
            
            # Keep the largest connected component
            if data.ndim >= 3:
                labeled, n_features = ndimage.label(mask_data)
                if n_features > 1:
                    sizes = [np.sum(labeled == i) for i in range(1, n_features + 1)]
                    largest = np.argmax(sizes) + 1
                    mask_data = (labeled == largest)
        
        mask_img = ants.from_numpy(
            mask_data.astype(np.float32),
            origin=img.origin,
            spacing=img.spacing,
            direction=img.direction
        )
        return mask_img
        
    except Exception as e:
        logger.warning(f"Brain-mask creation failed: {e}")
        data = img.numpy()
        threshold = np.percentile(data[data > 0], 20) if np.any(data > 0) else 0
        mask_data = (data > threshold).astype(np.float32)
        return ants.from_numpy(mask_data, origin=img.origin, spacing=img.spacing, direction=img.direction)

def compute_registration_metrics(fixed, warped, mask=None):
    """Compute the registration evaluation metrics"""
    f_data = fixed.numpy()
    w_data = warped.numpy()
    
    if mask is not None:
        mask_data = mask.numpy() > 0.5
        if mask_data.shape != f_data.shape:
            mask_data = (f_data > 0.01) & (w_data > 0.01)
    else:
        mask_data = (f_data > 0.01) & (w_data > 0.01)
    
    metrics = {}
    
    # Dice coefficient
    f_brain = (f_data > 0.01) & mask_data
    w_brain = (w_data > 0.01) & mask_data
    intersection = np.sum(f_brain & w_brain)
    metrics['Dice'] = 2 * intersection / (np.sum(f_brain) + np.sum(w_brain) + 1e-8)
    
    # Correlation coefficient
    if mask_data.sum() > 100:
        f_roi = f_data[mask_data]
        w_roi = w_data[mask_data]
        cc = np.corrcoef(f_roi, w_roi)[0, 1]
        metrics['CC'] = float(np.clip(abs(cc), 0, 1))
        metrics['MAE'] = float(np.mean(np.abs(f_roi - w_roi)))
    else:
        metrics['CC'] = 0.0
        metrics['MAE'] = float('inf')
    
    # Mutual information
    try:
        metrics['MI'] = float(ants.image_mutual_information(fixed, warped))
    except:
        metrics['MI'] = 0.0
    
    return metrics

def compute_jacobian(warp_path, domain_image, mask=None):
    """Compute the Jacobian determinant of the deformation field"""
    try:
        jacobian = ants.create_jacobian_determinant_image(domain_image, warp_path, do_log=False)
        jac_data = jacobian.numpy()
        
        if mask is not None:
            mask_data = mask.numpy() > 0
            if mask_data.shape == jac_data.shape:
                jac_roi = jac_data[mask_data]
            else:
                jac_roi = jac_data[jac_data != 0]
        else:
            jac_roi = jac_data[jac_data != 0]
        
        if len(jac_roi) == 0:
            return None, None
        
        stats = {
            'mean': float(np.mean(jac_roi)),
            'std': float(np.std(jac_roi)),
            'min': float(np.min(jac_roi)),
            'max': float(np.max(jac_roi)),
            'median': float(np.median(jac_roi)),
            'negative_pct': float(np.mean(jac_roi < 0) * 100),
            'expansion_pct': float(np.mean(jac_roi > 1.01) * 100),
            'contraction_pct': float(np.mean(jac_roi < 0.99) * 100)
        }
        
        return jacobian, stats
    except Exception as e:
        logger.warning(f"Jacobian computation failed: {e}")
        return None, None

def create_overlay_plot(fixed, warped, title, save_path):
    """Create the red/green overlay image"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        f_data = fixed.numpy()
        w_data = warped.numpy()
        mid = f_data.shape[0] // 2
        
        ax = axes[0]
        f_slice = f_data[mid, :, :]
        w_slice = w_data[mid, :, :]
        
        overlay = np.zeros((f_slice.shape[0], f_slice.shape[1], 3))
        
        # Red channel: moving
        w_mask = w_slice > 0.01
        if w_mask.sum() > 0:
            w_vals = w_slice[w_mask]
            p2, p98 = np.percentile(w_vals, [1, 99])
            w_norm = np.zeros_like(w_slice)
            w_norm[w_mask] = np.clip((w_slice[w_mask] - p2) / (p98 - p2 + 1e-8), 0, 1)
            overlay[:, :, 0] = w_norm
        
        # Green channel: fixed
        f_mask = f_slice > 0.01
        if f_mask.sum() > 0:
            f_vals = f_slice[f_mask]
            p2, p98 = np.percentile(f_vals, [1, 99])
            f_norm = np.zeros_like(f_slice)
            f_norm[f_mask] = np.clip((f_slice[f_mask] - p2) / (p98 - p2 + 1e-8), 0, 1)
            overlay[:, :, 1] = f_norm
        
        ax.imshow(overlay, origin='lower')
        ax.set_title(f'{title}\nRed=Moving, Green=Fixed, Yellow=Good', fontsize=10)
        ax.axis('off')
        
        # Difference map
        ax = axes[1]
        diff = np.zeros_like(f_slice)
        roi = (f_slice > 0.01) | (w_slice > 0.01)
        if roi.sum() > 0:
            diff[roi] = (f_slice[roi] - w_slice[roi]) / (np.max(np.abs(f_slice[roi] - w_slice[roi])) + 1e-8)
        
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title('Difference Map', fontsize=10)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        del fig, axes, f_data, w_data, overlay, diff
        force_gc()
    except Exception as e:
        logger.warning(f"Overlay creation failed: {e}")

def visualize_jacobian(jacobian_img, title, save_path, mask=None):
    """Visualise the Jacobian determinant"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        jac_data = jacobian_img.numpy()
        mid = jac_data.shape[0] // 2
        
        slices = [
            ('Sagittal', jac_data[mid, :, :].T),
            ('Coronal', jac_data[:, mid, :].T),
            ('Axial', jac_data[:, :, mid].T)
        ]
        
        if mask is not None:
            mask_data = mask.numpy() > 0
            if mask_data.shape == jac_data.shape:
                jac_roi = jac_data[mask_data]
            else:
                jac_roi = jac_data[jac_data != 0]
        else:
            jac_roi = jac_data[jac_data != 0]
        
        if len(jac_roi) > 0:
            p2, p98 = np.percentile(jac_roi, [1, 99])
            vmin = max(0, p2 * 0.8)
            vmax = min(3, p98 * 1.2)
        else:
            vmin, vmax = 0.5, 1.5
        
        for idx, (name, data) in enumerate(slices):
            im = axes[idx].imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            axes[idx].set_title(f'{name} [{vmin:.2f}, {vmax:.2f}]', fontsize=10)
            axes[idx].axis('off')
        
        plt.colorbar(im, ax=axes[-1], shrink=0.8, label='Jacobian')
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        del fig, axes, jac_data, slices
        force_gc()
    except Exception as e:
        logger.warning(f"Jacobian visualisation failed: {e}")

# ============================ Core class ============================
class ADNIRegistrationPipeline:
    """ADNI brain MRI batch registration pipeline"""
    
    def __init__(self, input_dir, template_path, output_dir):
        self.input_dir = Path(input_dir)
        self.template_path = Path(template_path)
        self.output_dir = Path(output_dir)
        
        self.dirs = {
            'registered': self.output_dir / 'registered',
            'warpfields': self.output_dir / 'warpfields',
            'jacobians': self.output_dir / 'jacobians',
            'overlays': self.output_dir / 'overlays',
            'metrics': self.output_dir / 'metrics',
            'reports': self.output_dir / 'reports'
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        
        self.template_img = None
        self.template_mask = None
        self.strategies = self._define_strategies()
        self.results = []
        self.jacobian_stats = {}
        self.failed_files = []
        self.completed_files = set()
        self._load_completed_files()
        self.process_count = 0
        
    def _load_completed_files(self):
        completed_list_path = self.dirs['reports'] / 'completed_files.txt'
        if completed_list_path.exists():
            with open(completed_list_path, 'r') as f:
                self.completed_files = set(line.strip() for line in f if line.strip())
            logger.info(f"Resume: found {len(self.completed_files)} completed files")
    
    def _save_completed_file(self, file_id):
        self.completed_files.add(file_id)
        completed_list_path = self.dirs['reports'] / 'completed_files.txt'
        with open(completed_list_path, 'a') as f:
            f.write(f"{file_id}\n")
    
    def _define_strategies(self):
        return {
            'rigid': {
                'name': 'Rigid-fast',
                'transform': 'Rigid',
                'params': {'reg_iterations': (50, 25, 10)}
            },
            'affine': {
                'name': 'Affine-standard',
                'transform': 'Affine',
                'params': {'reg_iterations': (100, 50, 25)}
            },
            'syn_fast': {
                'name': 'SyN-fast',
                'transform': 'SyN',
                'params': {
                    'reg_iterations': (50, 30, 20),
                    'syn_metric': 'CC',
                    'syn_sampling': 4,
                    'flow_sigma': 3,
                    'total_sigma': 0
                },
                'note': 'fast registration, about 400s'
            },
            'syn_standard': {
                'name': 'SyN-standard',
                'transform': 'SyN',
                'params': {
                    'reg_iterations': (100, 70, 50),
                    'syn_metric': 'CC',
                    'syn_sampling': 4,
                    'flow_sigma': 3,
                    'total_sigma': 0
                },
                'note': 'standard registration, recommended'
            },
            'syn_precise': {
                'name': 'SyN-precise',
                'transform': 'SyN',
                'params': {
                    'reg_iterations': (120, 80, 60),
                    'syn_metric': 'CC',
                    'syn_sampling': 2,
                    'flow_sigma': 3,
                    'total_sigma': 0
                },
                'note': 'precise registration, higher quality'
            },
            'syn_mi': {
                'name': 'SyN-MI-standard',
                'transform': 'SyN',
                'params': {
                    'reg_iterations': (100, 70, 50),
                    'syn_metric': 'Mattes',
                    'syn_sampling': 4,
                    'flow_sigma': 3,
                    'total_sigma': 0
                },
                'note': 'mutual-information metric, suitable for images with different contrast'
            }
        }
    
    def load_template(self):
        logger.info(f"Loading template: {self.template_path}")
        try:
            self.template_img = ants.image_read(str(self.template_path))
            self.template_mask = create_brain_mask(self.template_img)
            logger.info(f"Template loaded successfully: shape={self.template_img.shape}")
            return True
        except Exception as e:
            logger.error(f"Template loading failed: {e}")
            return False

    def perform_registration(self, moving_path, strategy_key, use_moving_mask=True):
        """Run the registration but do not save files. Returns (metrics, reg_result, warped_img, moving_mask)"""
        file_id = Path(moving_path).stem.replace('.nii', '')
        strategy = self.strategies.get(strategy_key)
        if strategy is None:
            logger.error(f"Unknown strategy: {strategy_key}")
            return None, None, None, None

        try:
            moving_raw = ants.image_read(str(moving_path))
            moving_mask = create_brain_mask(moving_raw)

            # Decide in the preprocessing stage whether to apply the moving-image mask, based on use_moving_mask
            moving_processed = self.preprocess_moving_image(moving_raw, moving_mask if use_moving_mask else None)

            start_time = time.time()
            # Pass in the fixed-image mask (self.template_mask). Note: ANTsPy's ants.registration
            # does not support an argument named `moving_mask` (it raises unexpected keyword
            # argument), so the moving-image mask is applied in the preprocessing stage
            # (create_brain_mask / preprocess_moving_image).
            reg = ants.registration(
                fixed=self.template_img,
                moving=moving_processed,
                type_of_transform=strategy['transform'],
                mask=self.template_mask,  # fixed-image mask
                **strategy['params'],
                verbose=False
            )
            elapsed = time.time() - start_time
            warped = reg.get('warpedmovout', None)

            metrics = compute_registration_metrics(self.template_img, warped, self.template_mask)
            metrics['processing_time'] = elapsed
            metrics['file_id'] = file_id
            metrics['strategy'] = strategy_key
            metrics['transform_type'] = strategy['transform']

            return metrics, reg, warped, moving_mask
        except Exception as e:
            logger.error(f"Registration failed {file_id}: {e}")
            # Print the full stack at error level so that it is visible at the default log level
            logger.error(traceback.format_exc())
            self.failed_files.append({'file': file_id, 'error': str(e)})
            return None, None, None, None
    
    def _quantile_histogram_matching(self, source_img, reference_img, mask=None):
        """
        Quantile-based histogram matching
        
        Principle:
        - match only the mid-tones (gray matter/white matter) and ignore the extreme values
        - the background (value 0) is left unchanged
        - hyperintense regions (pathology/blood vessels) keep their relative relationships
        - particularly suitable for ADNI data (preserving the pathological information of the atrophic brain)

        Parameters:
            source_img: the image to be matched
            reference_img: the reference image (template)
            mask: optional mask

        Returns:
            the matched image
        """
        source_data = source_img.numpy().astype(np.float64)
        ref_data = reference_img.numpy().astype(np.float64)
        
        # Determine the valid region
        if mask is not None:
            mask_data = mask.numpy() > 0.5
            if mask_data.sum() > 100 and mask_data.shape == source_data.shape:
                source_valid = source_data[mask_data]
                if mask_data.shape == ref_data.shape:
                    ref_valid = ref_data[mask_data]
                else:
                    logger.debug("  the template and moving-image shapes differ; using all valid brain voxels for the template")
                    ref_valid = ref_data[ref_data > 0]
            else:
                source_valid = source_data[source_data > 0]
                ref_valid = ref_data[ref_data > 0]
        else:
            source_valid = source_data[source_data > 0]
            ref_valid = ref_data[ref_data > 0]
        
        if len(source_valid) < 100 or len(ref_valid) < 100:
            logger.warning("  insufficient valid voxels; skipping histogram matching")
            return source_img
        
        # ============================================================
        # Quantile-matching strategy
        # ============================================================
        # Define the mapping range: match only the 2%–98% of voxels
        # Keep the darkest 1%–2% (background/CSF edges)
        # Keep the brightest 98%–99% (vessels/pathological hyperintensity)
        # ============================================================
        
        lower_bound = 0.5   # ignore the darkest 2%
        upper_bound = 99.5  # ignore the brightest 2%
        
        # Sample quantile points uniformly over the [2%, 98%] interval
        n_quantiles = 1024  # number of quantile points (the more, the smoother)
        quantiles = np.linspace(lower_bound, upper_bound, n_quantiles)
        
        # Compute the quantiles of the source and the reference
        source_quantiles = np.percentile(source_valid, quantiles)
        ref_quantiles = np.percentile(ref_valid, quantiles)
        
        # Ensure monotonic increase (removing duplicate values)
        diff = np.diff(source_quantiles)
        valid_idx = np.concatenate([[True], diff > 1e-10])
        
        if valid_idx.sum() < 10:
            logger.warning("  insufficient quantile points; using simple scaling")
            # Simple scaling to the same range
            s_min, s_max = np.percentile(source_valid, [lower_bound, upper_bound])
            r_min, r_max = np.percentile(ref_valid, [lower_bound, upper_bound])
            if s_max > s_min and r_max > r_min:
                matched_data = np.clip(
                    (source_data - s_min) / (s_max - s_min) * (r_max - r_min) + r_min,
                    0, None
                )
            else:
                matched_data = source_data
        else:
            s_q = source_quantiles[valid_idx]
            r_q = ref_quantiles[valid_idx]
            
            # Create the piecewise-linear interpolation mapping
            f_map = interp1d(
                s_q, r_q,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'  # extrapolate values outside the range
            )
            
            # Apply the mapping
            matched_data = f_map(source_data)
            
            # Handle NaN and Inf
            matched_data = np.nan_to_num(matched_data, nan=0.0, posinf=ref_valid.max(), neginf=0.0)
            
            # Ensure non-negative
            matched_data = np.maximum(matched_data, 0)
        
        # ============================================================
        # Key point: keep the background 0 values unchanged
        # ============================================================
        background_mask = source_data == 0
        matched_data[background_mask] = 0
        
        # Report the statistics before and after matching
        if logger.isEnabledFor(logging.DEBUG):
            old_p50 = np.percentile(source_valid, 50)
            new_p50 = np.percentile(matched_data[matched_data > 0], 50)
            ref_p50 = np.percentile(ref_valid, 50)
            logger.debug(f"  histogram matching: Source median={old_p50:.1f} → "
                        f"Matched median={new_p50:.1f} (Ref={ref_p50:.1f})")
        
        # Create the ANTs image
        matched_img = ants.from_numpy(
            matched_data.astype(np.float32),
            origin=source_img.origin,
            spacing=source_img.spacing,
            direction=source_img.direction
        )
        
        return matched_img
    
    def preprocess_moving_image(self, moving_img, moving_mask):
        """
        Preprocess the moving image: N4 correction → quantile histogram matching
        """
        # Step 1: N4 bias-field correction
        try:
            moving_n4 = ants.n4_bias_field_correction(
                moving_img,
                mask=moving_mask,
                shrink_factor=2,
                convergence={'iters': [50, 50, 50, 50], 'tol': 1e-7}
            )
            logger.debug("  N4 correction complete")
        except Exception as e:
            logger.warning(f"  N4 correction failed: {e}; using the original image")
            moving_n4 = moving_img
        
        # Step 1.5: if a moving-image mask is provided, zero the voxels outside it so that they do not affect the histogram matching or the similarity metric during registration
        try:
            if moving_mask is not None:
                try:
                    mask_data = moving_mask.numpy() > 0
                    mn4_data = moving_n4.numpy()
                    if mask_data.shape == mn4_data.shape:
                        mn4_data_masked = mn4_data.copy()
                        mn4_data_masked[~mask_data] = 0
                        moving_n4 = ants.from_numpy(
                            mn4_data_masked.astype(np.float32),
                            origin=moving_n4.origin,
                            spacing=moving_n4.spacing,
                            direction=moving_n4.direction
                        )
                        logger.debug("  mask applied to the moving image (voxels outside the mask set to zero)")
                except Exception:
                    logger.debug("  failed to apply the moving mask; skipping mask application")
        except Exception as e:
            logger.warning(f"  failed to apply the moving mask: {e}; skipping")
        
        # Step 2: quantile histogram matching
        try:
            # Prefer the built-in ANTs function
            if hasattr(ants, 'histogram_match'):
                moving_matched = ants.histogram_match(moving_n4, self.template_img)
                logger.debug("  histogram matching complete (built-in ANTS)")
            elif hasattr(ants, 'iMath_histogram_matching'):
                moving_matched = ants.iMath_histogram_matching(
                    moving_n4, self.template_img, mask=moving_mask
                )
                logger.debug("  histogram matching complete (iMath)")
            else:
                # Use our own quantile method
                moving_matched = self._quantile_histogram_matching(
                    moving_n4, self.template_img, moving_mask
                )
                logger.debug("  histogram matching complete (quantile method)")
        except Exception as e:
            logger.warning(f"  ANTS histogram matching failed: {e}; using the quantile method")
            try:
                moving_matched = self._quantile_histogram_matching(
                    moving_n4, self.template_img, moving_mask
                )
                logger.debug("  histogram matching complete (quantile fallback)")
            except Exception as e2:
                logger.warning(f"  quantile matching also failed: {e2}; using the N4 result")
                moving_matched = moving_n4
        
        return moving_matched
    
    def register_single(self, moving_path, strategy_key, save_outputs=True, check_existing=True, save_warp=None, save_jacobian=None, use_moving_mask=True):
        """Register a single image: always run the registration and save `registered`;
        optionally save warp and jacobian

        save_warp/save_jacobian: None means decide automatically from the strategy
        (they are saved for the SyN-type strategies)
        """
        file_id = Path(moving_path).stem.replace('.nii', '')
        strategy = self.strategies.get(strategy_key)

        if strategy is None:
            logger.error(f"Unknown strategy: {strategy_key}")
            return None

        # Decide the default behaviour: if output saving is requested, save the warp field by default
        if save_warp is None:
            save_warp = save_outputs
        if save_jacobian is None:
            save_jacobian = False

        # Resume check
        if check_existing and save_outputs:
            if file_id in self.completed_files:
                logger.info(f"⏭️ skipping completed: {file_id}")
                return {'file_id': file_id, 'skipped': True, 'reason': 'already_completed'}

            required_files = ['registered', 'overlay']
            if save_warp:
                required_files.append('warpfield')
            if save_jacobian:
                required_files.append('jacobian')

            all_exist, missing = check_output_exists(self.output_dir, file_id, required_files)
            if all_exist:
                self._save_completed_file(file_id)
                logger.info(f"⏭️ skipping (files already exist): {file_id}")
                return {'file_id': file_id, 'skipped': True, 'reason': 'files_exist'}

        logger.info(f"Registering: {file_id} | strategy: {strategy['name']}")

        # Run the registration (warp/jacobian are not saved unless requested later)
        metrics, reg, warped, moving_mask = self.perform_registration(moving_path, strategy_key, use_moving_mask=use_moving_mask)
        if metrics is None:
            return None

        try:
            if save_outputs and warped is not None:
                reg_path = self.dirs['registered'] / f"{file_id}.nii.gz"
                ants.image_write(warped, str(reg_path))

                overlay_path = self.dirs['overlays'] / f"{file_id}_overlay.png"
                create_overlay_plot(self.template_img, warped, strategy['name'], str(overlay_path))

                # Save warp and jacobian as required
                if save_warp or save_jacobian:
                    self._save_deformation_fields(reg, file_id, moving_mask, save_warp=save_warp, save_jacobian=save_jacobian)

                self._save_completed_file(file_id)

            # Quality-assessment information
            if metrics['CC'] >= 0.85:
                quality = 'Excellent'
            elif metrics['CC'] >= 0.80:
                quality = 'Good'
            elif metrics['CC'] >= 0.75:
                quality = 'Acceptable'
            else:
                quality = 'Needs review'

            # NOTE: the label "Done:" and the field label "time=" below form the parsing
            # interface with the registration-QC summary script in this directory, which reads
            # this line with r'Done:\s*Dice=([\d.]+),\s*CC=([\d.]+),\s*time=([\d.]+)'.
            # Both sides must be changed together. This matches the log format emitted by the
            # archived original script in legacy/.
            logger.info(f"  Done: Dice={metrics['Dice']:.4f}, CC={metrics['CC']:.4f}, "
                       f"time={metrics['processing_time']:.1f}s [{quality}]")

            self.results.append(metrics)
            return metrics

        finally:
            safe_del(reg, warped, moving_mask)
            self.process_count += 1
            if self.process_count % Config.GC_FREQUENCY == 0:
                force_gc()
    
    def _save_deformation_fields(self, reg_result, file_id, mask=None, save_warp=True, save_jacobian=True):
        """Save or compute the deformation field and the Jacobian; the flags control the behaviour"""
        warp_field = None
        jacobian = None
        try:
            if 'fwdtransforms' not in reg_result:
                return

            warp_path = None
            for t in reg_result['fwdtransforms']:
                if 'Warp' in t or 'warp' in t.lower():
                    warp_path = t
                    break

            if warp_path is None:
                return

            warp_save = self.dirs['warpfields'] / f"{file_id}_warp.nii.gz"
            if save_warp:
                try:
                    # If warp_path is a temporary file, copy it immediately to the permanent output directory
                    if os.path.exists(warp_path):
                        shutil.copy2(warp_path, warp_save)
                    else:
                        warp_field = ants.image_read(warp_path)
                        ants.image_write(warp_field, str(warp_save))
                    warp_field = ants.image_read(str(warp_save))
                except Exception as e:
                    logger.warning(f"Failed to save the warp field {file_id}: {e}")

            if save_jacobian:
                jacobian, jac_stats = compute_jacobian(str(warp_save) if save_warp else warp_path, self.template_img, mask)
                if jacobian is not None and jac_stats is not None:
                    jac_save = self.dirs['jacobians'] / f"{file_id}_jacobian.nii.gz"
                    ants.image_write(jacobian, str(jac_save))

                    jac_viz = self.dirs['jacobians'] / f"{file_id}_jacobian.png"
                    visualize_jacobian(jacobian, f'Jacobian - {file_id}', str(jac_viz), mask)

                    self.jacobian_stats[file_id] = jac_stats
        except Exception as e:
            logger.warning(f"Deformation-field/Jacobian processing failed {file_id}: {e}")
        finally:
            safe_del(warp_field, jacobian)
    
    def grid_search(self, sample_size=2, n_combinations=10):
        """Grid search"""
        logger.info("="*60)
        logger.info("Starting the parameter grid search")
        logger.info("="*60)
        
        input_files = sorted(
            list(self.input_dir.glob("*.nii")) + list(self.input_dir.glob("*.nii.gz"))
        )
        
        if not input_files:
            return None
        
        sample_files = input_files[:min(sample_size, len(input_files))]
        logger.info(f"Using {len(sample_files)} sample files")
        
        strategy_keys = list(self.strategies.keys())
        syn_keys = [k for k in strategy_keys if k.startswith('syn')]
        other_keys = [k for k in strategy_keys if not k.startswith('syn')]
        test_strategies = (syn_keys + other_keys)[:n_combinations]
        
        all_results = []
        original_results = self.results
        self.results = []
        
        for strategy_key in test_strategies:
            strategy = self.strategies[strategy_key]
            logger.info(f"\nTesting strategy: {strategy['name']} ({strategy_key})")
            
            file_results = []
            for sample_file in sample_files:
                result = self.register_single(sample_file, strategy_key, save_outputs=False, check_existing=False, use_moving_mask=True)
                if result and not result.get('skipped', False):
                    file_results.append(result)
            
            if file_results:
                avg_dice = np.mean([r['Dice'] for r in file_results])
                avg_cc = np.mean([r['CC'] for r in file_results])
                avg_time = np.mean([r['processing_time'] for r in file_results])
                
                all_results.append({
                    'strategy': strategy_key,
                    'name': strategy['name'],
                    'transform': strategy['transform'],
                    'avg_dice': avg_dice,
                    'avg_cc': avg_cc,
                    'avg_time': avg_time,
                    'samples': len(file_results)
                })
                
                logger.info(f"  mean: Dice={avg_dice:.4f}, CC={avg_cc:.4f}, time={avg_time:.1f}s")
            
            force_gc()
        
        self.results = original_results
        
        if all_results:
            df = pd.DataFrame(all_results).sort_values('avg_cc', ascending=False)
            df.to_csv(self.dirs['reports'] / 'grid_search_results.csv', index=False)
            
            best = df.iloc[0]
            logger.info(f"\n🏆 best strategy: {best['name']} (CC={best['avg_cc']:.4f})")
            return df
        return None
    
    def batch_process(self, strategy_key='syn_standard', auto_select=True, resume=True):
        """Batch processing"""
        input_files = sorted(
            list(self.input_dir.glob("*.nii")) + list(self.input_dir.glob("*.nii.gz"))
        )
        
        if not input_files:
            return
        
        if resume and len(self.completed_files) > 0:
            logger.info(f"Resume: {len(self.completed_files)}/{len(input_files)} already completed")
        
        # Automatically select the strategy (if grid-search results exist)
        if auto_select:
            grid_path = self.dirs['reports'] / 'grid_search_results.csv'
            if grid_path.exists():
                try:
                    df = pd.read_csv(grid_path)
                    best = df[df['strategy'].str.startswith('syn')].iloc[0]
                    strategy_key = best['strategy']
                    logger.info(f"Automatically selected: {self.strategies[strategy_key]['name']}")
                except:
                    pass

        strategy = self.strategies[strategy_key]
        logger.info(f"Strategy: {strategy['name']} | pipeline: N4 → quantile histogram matching → {strategy['transform']}")
        logger.info(f"Parameters: flow_sigma=3")

        # [AD_CNN_code change] this used to be an interactive submenu; the paper pipeline defaults to:
        # registration + automatic saving of the warp field (save_warp=True), without computing the Jacobian.
        do_register = True
        do_warp = True
        do_jacobian = bool(getattr(self, "cli_do_jacobian", False))

        success = 0
        skipped = 0
        t0 = time.time()

        for i, fp in enumerate(input_files, 1):
            logger.info(f"\n[{i}/{len(input_files)}] {fp.name}")
            result = self.register_single(fp, strategy_key, save_outputs=do_register, check_existing=resume, save_warp=do_warp, save_jacobian=do_jacobian, use_moving_mask=True)

            if result:
                if result.get('skipped'):
                    skipped += 1
                else:
                    success += 1
                    if success % 10 == 0:
                        self._save_intermediate_results()

        elapsed = time.time() - t0
        self._generate_report(strategy_key, success, skipped, len(input_files), elapsed)

        logger.info(f"\nDone: {success} succeeded | {skipped} skipped | {len(self.failed_files)} failed")
        logger.info(f"Elapsed: {elapsed/60:.1f} minutes")

        return success, self.results

    def post_process(self, resume=True):
        """Post-processing: additionally save the warp field or compute the Jacobian for already-registered files (supports batches and resuming)"""
        # Collect the already-registered files
        registered_files = sorted(self.dirs['registered'].glob("*.nii*"))
        if not registered_files:
            print("No registered files found")
            return

        # Find the files that are missing warp or jacobian
        need_warp = []
        need_jacobian = []
        for f in registered_files:
            file_id = Path(f).stem.replace('.nii', '')
            _, missing = check_output_exists(self.output_dir, file_id, required_files=['registered','warpfield','jacobian','overlay'])
            # If both warpfield and jacobian are missing, both have to be added
            if 'warpfield' in missing:
                need_warp.append(file_id)
            if 'jacobian' in missing:
                # If warp exists but jacobian is missing, or both are missing, the jacobian must also be computed
                need_jacobian.append(file_id)

        print(f"Warp fields to save: {len(need_warp)}; Jacobians to compute: {len(need_jacobian)}")
        print("\nPlease choose the operation(s) to perform (multiple allowed):")
        print("[1] � compute Jacobian (for files that have a warp but no Jacobian)")
        print("[2] 📦 additionally save the warp field (for files that are registered but have no warp)")
        print("[3] 🚀 select all and execute")
        print("[4] 🔙 return to the previous menu")
        sel = input("\nEnter your choice (e.g. 1, or 1,2, or 3): ").strip()
        if not sel:
            print("Nothing selected; cancelled")
            return
        choices = set([s.strip() for s in sel.split(',') if s.strip().isdigit()])
        if '4' in choices:
            return
        do_jacobian = '3' in choices or '1' in choices
        do_warp = '3' in choices or '2' in choices

        # Try to read the previous batch details to obtain the strategy information
        strategy_map = {}
        details_path = self.dirs['reports'] / 'batch_details.csv'
        if details_path.exists():
            try:
                df = pd.read_csv(details_path)
                for _, row in df.iterrows():
                    fid = str(row.get('file_id') or row.get('file'))
                    strategy_map[fid] = row.get('strategy')
            except:
                pass

        total = len(need_warp) + len(need_jacobian)
        processed = 0
        for fid in need_warp:
            if resume:
                ok, miss = check_output_exists(self.output_dir, fid, required_files=['warpfield'])
                if ok:
                    continue
            # Find the original moving file
            mp = None
            for ext in ['.nii', '.nii.gz']:
                p = self.input_dir / f"{fid}{ext}"
                if p.exists():
                    mp = p
                    break
            if mp is None:
                logger.warning(f"Original input file not found, cannot rebuild the warp: {fid}")
                continue
            strat = strategy_map.get(fid, 'syn_standard')
            metrics, reg, warped, moving_mask = self.perform_registration(mp, strat, use_moving_mask=True)
            if reg is None:
                continue
            # When the user also selects adding the Jacobian, pass save_jacobian=do_jacobian
            self._save_deformation_fields(reg, fid, moving_mask, save_warp=do_warp, save_jacobian=do_jacobian)
            processed += 1

        for fid in need_jacobian:
            if resume:
                ok, miss = check_output_exists(self.output_dir, fid, required_files=['jacobian'])
                if ok:
                    continue
            # path of the warp file
            warp_path = self.dirs['warpfields'] / f"{fid}_warp.nii.gz"
            if not warp_path.exists():
                logger.warning(f"Warp file missing, cannot compute the Jacobian: {fid}")
                continue
            # Call the jacobian branch of _save_deformation_fields directly
            fake_reg = {'fwdtransforms': [str(warp_path)]}
            # Use the template brain mask to compute the Jacobian, so that non-brain regions do not contaminate the statistics
            self._save_deformation_fields(fake_reg, fid, mask=self.template_mask, save_warp=False, save_jacobian=do_jacobian)
            processed += 1

        print(f"Post-processing complete: processed {processed}/{total} files")
    
    def _save_intermediate_results(self):
        try:
            path = self.dirs['reports'] / f'intermediate_{self.process_count}.json'
            with open(path, 'w') as f:
                json.dump({
                    'completed': list(self.completed_files),
                    'failed': self.failed_files,
                    'count': self.process_count,
                    'time': datetime.now().isoformat()
                }, f, indent=2)
        except:
            pass
    
    def _generate_report(self, strategy_key, success, skipped, total, total_time):
        """Generate the report"""
        actual = [r for r in self.results if not r.get('skipped')]
        if not actual:
            return
        
        df = pd.DataFrame(actual)
        
        summary = {
            'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Strategy': self.strategies[strategy_key]['name'],
            'Pipeline': 'N4 correction → quantile histogram matching → SyN registration (flow_sigma=3)',
            'Total files': total,
            'Succeeded': success,
            'Skipped': skipped,
            'Failed': len(self.failed_files),
            'Elapsed (minutes)': round(total_time/60, 1),
            'Mean time (s)': round(df['processing_time'].mean(), 1),
            'Mean Dice': round(df['Dice'].mean(), 4),
            'Mean CC': round(df['CC'].mean(), 4),
            'CC≥0.85': f"{np.mean(df['CC']>=0.85)*100:.1f}%",
            'CC≥0.80': f"{np.mean(df['CC']>=0.80)*100:.1f}%",
            'CC<0.75': f"{np.mean(df['CC']<0.75)*100:.1f}%",
        }
        
        with open(self.dirs['reports'] / 'batch_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        df.to_csv(self.dirs['reports'] / 'batch_details.csv', index=False)
        
        if self.failed_files:
            with open(self.dirs['reports'] / 'failed_files.json', 'w') as f:
                json.dump(self.failed_files, f, indent=2, ensure_ascii=False)
        
        if self.jacobian_stats:
            pd.DataFrame(self.jacobian_stats).T.to_csv(self.dirs['reports'] / 'jacobian_stats.csv')
        
        print("\n" + "="*60)
        print("📊 Batch-processing report")
        print("="*60)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("="*60)

# ============================ Main program ============================
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(
        description="ADNI/MIRIAD brain MRI batch registration (paper pipeline: N4 → quantile histogram "
                    "matching → ANTs SyN-fast; outputs the registered image + warp field; supports resume)")
    ap.add_argument("--input", required=True, help="input NIfTI directory (skull-stripped images, *.nii / *.nii.gz)")
    ap.add_argument("--template", required=True, help="MNI152 skull-stripped template (mni.nii.gz)")
    ap.add_argument("--output", required=True, help="output root directory (registered/ warpfields/ overlays/ reports/ are created automatically)")
    ap.add_argument("--strategy", default="syn_fast",
                    choices=["rigid", "affine", "syn_fast", "syn_standard", "syn_precise", "syn_mi"],
                    help="registration strategy (the paper uses syn_fast, the default)")
    ap.add_argument("--no-resume", action="store_true", help="disable resume (enabled by default)")
    ap.add_argument("--save-jacobian", action="store_true",
                    help="additionally save the Jacobian determinant image (the paper pipeline does not generate it by default)")
    ap.add_argument("--single", metavar="SUBSTR", help="process only the samples whose file name contains this substring (smoke test)")
    ap.add_argument("--max-files", type=int, default=None, help="limit the number of files processed (smoke test)")
    return ap.parse_args()


def main():
    args = parse_args()
    import sys as _sys
    input_dir = Path(args.input)
    template_path = Path(args.template)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: the input directory does not exist: {input_dir}")
        _sys.exit(2)
    if not template_path.exists():
        print(f"Error: the template file does not exist: {template_path}")
        _sys.exit(2)

    pipeline = ADNIRegistrationPipeline(input_dir, template_path, output_dir)
    pipeline.cli_do_jacobian = args.save_jacobian

    if not pipeline.load_template():
        _sys.exit(2)

    input_files = sorted(
        list(input_dir.glob("*.nii")) + list(input_dir.glob("*.nii.gz"))
    )
    print(f"\nFound {len(input_files)} input files | strategy: {args.strategy} | resume: {not args.no_resume}")

    if not input_files:
        return

    if args.single:
        input_files = [f for f in input_files if args.single in f.name]
        if not input_files:
            print(f"Error: no sample was found whose file name contains '{args.single}'")
            _sys.exit(2)
        print(f"Smoke mode (--single): {len(input_files)} files -> {[f.name for f in input_files]}")

    if args.max_files is not None:
        input_files = input_files[:args.max_files]
        print(f"Smoke mode (--max-files): limited to {len(input_files)} files")

    resume = not args.no_resume
    pipeline.batch_process(strategy_key=args.strategy, auto_select=False, resume=resume)


if __name__ == "__main__":
    main()
