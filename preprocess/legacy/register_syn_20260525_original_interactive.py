# ================================================================
# ADNI brain MRI batch registration system - final version v3.2
# Fixed: use quantile histogram matching (preserves pathological information)
# Date: 2024-06-25 revised the histogram-matching strategy, extending the quantile bounds lower_bound[2, 98]-[0.5, 99.5],
# increased the sampling points n_quantiles,200->1024, improving the matching quality, which particularly suits the characteristics of ADNI data (atrophic brain + pathological signal)               
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
class Config:
    """Global configuration"""
    # Path configuration
    INPUT_DIR = Path(r"<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized_monailabel_mask_process_20260523\1")
    TEMPLATE_PATH = Path(r"<AUTHOR_DATA_ROOT>/datasets\MNI_152\mni_icbm152_nlin_asym_09a_nifti\mni_icbm152_nlin_asym_09a\mni.nii.gz")
    OUTPUT_DIR = Path(r"<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized_monailabel_mask_process_20260525\1")
    
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
    """Check whether the output files already exist (resumable)"""
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
        logger.warning(f"Brain mask creation failed: {e}")
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
    """Compute the Jacobian determinant of the deformation (warp) field"""
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
    """Create the red-green overlay figure"""
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
        logger.warning(f"Overlay figure creation failed: {e}")

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
            logger.info(f"Resumable: found {len(self.completed_files)} completed files")
    
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
                'note': 'Fast registration, about 400s'
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
                'note': 'Standard registration, recommended'
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
                'note': 'Precise registration, higher quality'
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
                'note': 'Mutual-information metric, suitable for images with differing contrast'
            }
        }
    
    def load_template(self):
        logger.info(f"Loading template: {self.template_path}")
        try:
            self.template_img = ants.image_read(str(self.template_path))
            self.template_mask = create_brain_mask(self.template_img)
            logger.info(f"Template loaded: shape={self.template_img.shape}")
            return True
        except Exception as e:
            logger.error(f"Template load failed: {e}")
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

            # At the preprocessing stage, use_moving_mask decides whether the moving-image mask is applied
            moving_processed = self.preprocess_moving_image(moving_raw, moving_mask if use_moving_mask else None)

            start_time = time.time()
            # The fixed-image mask (self.template_mask) is passed in. Note: ANTsPy's ants.registration does
            # not support a parameter named `moving_mask` (it raises unexpected keyword argument),
            # so the moving-image mask is applied at the preprocessing stage (create_brain_mask / preprocess_moving_image).
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
        Quantile-based Histogram Matching
        
        Principle:
        - Match only the mid-tones (GM/WM), ignoring extreme values
        - The background (0 values) is left unchanged
        - Bright regions (pathology/vessels) keep their relative relationship
        - Particularly suited to ADNI data (preserves the pathological information of the atrophic brain)
        
        Parameters:
            source_img: image to be matched
            reference_img: reference image (template)
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
                    logger.debug("  Template and moving image shapes differ; using all valid brain voxels of the template")
                    ref_valid = ref_data[ref_data > 0]
            else:
                source_valid = source_data[source_data > 0]
                ref_valid = ref_data[ref_data > 0]
        else:
            source_valid = source_data[source_data > 0]
            ref_valid = ref_data[ref_data > 0]
        
        if len(source_valid) < 100 or len(ref_valid) < 100:
            logger.warning("  Insufficient valid voxels; skipping histogram matching")
            return source_img
        
        # ============================================================
        # Quantile-matching strategy
        # ============================================================
        # Define the mapping range: match only the 2%–98% of voxels
        # Keep the darkest 1%–2% (background/CSF edge)
        # Keep the brightest 98%–99% (vessels/pathological hyperintensity)
        # ============================================================
        
        lower_bound = 0.5   # ignore the darkest 2%
        upper_bound = 99.5  # ignore the brightest 2%
        
        # Sample quantile points uniformly over the [2%, 98%] interval
        n_quantiles = 1024  # number of quantile points (more = smoother)
        quantiles = np.linspace(lower_bound, upper_bound, n_quantiles)
        
        # Compute the source and reference quantiles
        source_quantiles = np.percentile(source_valid, quantiles)
        ref_quantiles = np.percentile(ref_valid, quantiles)
        
        # Ensure monotonic increase (remove duplicate values)
        diff = np.diff(source_quantiles)
        valid_idx = np.concatenate([[True], diff > 1e-10])
        
        if valid_idx.sum() < 10:
            logger.warning("  Too few quantile points; using simple rescaling")
            # Simple rescaling to the same range
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
            
            # Build the piecewise-linear interpolation map
            f_map = interp1d(
                s_q, r_q,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'  # extrapolate for values outside the range
            )
            
            # Apply the map
            matched_data = f_map(source_data)
            
            # Handle NaN and Inf
            matched_data = np.nan_to_num(matched_data, nan=0.0, posinf=ref_valid.max(), neginf=0.0)
            
            # Ensure non-negativity
            matched_data = np.maximum(matched_data, 0)
        
        # ============================================================
        # Key point: keep the background 0 values unchanged
        # ============================================================
        background_mask = source_data == 0
        matched_data[background_mask] = 0
        
        # Report statistics before and after matching
        if logger.isEnabledFor(logging.DEBUG):
            old_p50 = np.percentile(source_valid, 50)
            new_p50 = np.percentile(matched_data[matched_data > 0], 50)
            ref_p50 = np.percentile(ref_valid, 50)
            logger.debug(f"  Histogram matching: Source median={old_p50:.1f} → "
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
        
        # Step 1.5: if a moving-image mask is provided, zero the voxels outside the mask so that they do not affect the histogram matching or the similarity metric during registration
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
                        logger.debug("  Moving-image mask applied (voxels outside the mask zeroed)")
                except Exception:
                    logger.debug("  Failed to apply the moving mask; skipping mask application")
        except Exception as e:
            logger.warning(f"  Failed to apply the moving mask: {e}; skipping")
        
        # Step 2: quantile histogram matching
        try:
            # Prefer the ANTs built-in functions
            if hasattr(ants, 'histogram_match'):
                moving_matched = ants.histogram_match(moving_n4, self.template_img)
                logger.debug("  Histogram matching complete (ANTS built-in)")
            elif hasattr(ants, 'iMath_histogram_matching'):
                moving_matched = ants.iMath_histogram_matching(
                    moving_n4, self.template_img, mask=moving_mask
                )
                logger.debug("  Histogram matching complete (iMath)")
            else:
                # Use our own quantile method
                moving_matched = self._quantile_histogram_matching(
                    moving_n4, self.template_img, moving_mask
                )
                logger.debug("  Histogram matching complete (quantile method)")
        except Exception as e:
            logger.warning(f"  ANTS histogram matching failed: {e}; using the quantile method")
            try:
                moving_matched = self._quantile_histogram_matching(
                    moving_n4, self.template_img, moving_mask
                )
                logger.debug("  Histogram matching complete (quantile fallback)")
            except Exception as e2:
                logger.warning(f"  Quantile matching also failed: {e2}; using the N4 result")
                moving_matched = moving_n4
        
        return moving_matched
    
    def register_single(self, moving_path, strategy_key, save_outputs=True, check_existing=True, save_warp=None, save_jacobian=None, use_moving_mask=True):
        """Register a single image: always runs the registration and saves 'registered'; warp and jacobian are optional

        save_warp/save_jacobian: None means decide automatically from the strategy (the SyN family saves them)
        """
        file_id = Path(moving_path).stem.replace('.nii', '')
        strategy = self.strategies.get(strategy_key)

        if strategy is None:
            logger.error(f"Unknown strategy: {strategy_key}")
            return None

        # Decide the default behaviour: if outputs are to be saved, the warp field is saved automatically by default
        if save_warp is None:
            save_warp = save_outputs
        if save_jacobian is None:
            save_jacobian = False

        # Resumable check
        if check_existing and save_outputs:
            if file_id in self.completed_files:
                logger.info(f"⏭️ Skipping completed: {file_id}")
                return {'file_id': file_id, 'skipped': True, 'reason': 'already_completed'}

            required_files = ['registered', 'overlay']
            if save_warp:
                required_files.append('warpfield')
            if save_jacobian:
                required_files.append('jacobian')

            all_exist, missing = check_output_exists(self.output_dir, file_id, required_files)
            if all_exist:
                self._save_completed_file(file_id)
                logger.info(f"⏭️ Skipping (files already exist): {file_id}")
                return {'file_id': file_id, 'skipped': True, 'reason': 'files_exist'}

        logger.info(f"Registration: {file_id} | strategy: {strategy['name']}")

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
                quality = 'excellent'
            elif metrics['CC'] >= 0.80:
                quality = 'good'
            elif metrics['CC'] >= 0.75:
                quality = 'acceptable'
            else:
                quality = 'needs review'

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
        """Save or compute the deformation (warp) field and the Jacobian, with behaviour controlled by the flags"""
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
            logger.warning(f"Warp-field/Jacobian handling failed {file_id}: {e}")
        finally:
            safe_del(warp_field, jacobian)
    
    def grid_search(self, sample_size=2, n_combinations=10):
        """Grid search"""
        logger.info("="*60)
        logger.info("Starting parameter grid search")
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
                
                logger.info(f"  Mean: Dice={avg_dice:.4f}, CC={avg_cc:.4f}, time={avg_time:.1f}s")
            
            force_gc()
        
        self.results = original_results
        
        if all_results:
            df = pd.DataFrame(all_results).sort_values('avg_cc', ascending=False)
            df.to_csv(self.dirs['reports'] / 'grid_search_results.csv', index=False)
            
            best = df.iloc[0]
            logger.info(f"\n🏆 Best strategy: {best['name']} (CC={best['avg_cc']:.4f})")
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
            logger.info(f"Resumable: {len(self.completed_files)}/{len(input_files)} completed")
        
        # Automatic strategy selection (if grid-search results are available)
        if auto_select:
            grid_path = self.dirs['reports'] / 'grid_search_results.csv'
            if grid_path.exists():
                try:
                    df = pd.read_csv(grid_path)
                    best = df[df['strategy'].str.startswith('syn')].iloc[0]
                    strategy_key = best['strategy']
                    logger.info(f"Auto-selected: {self.strategies[strategy_key]['name']}")
                except:
                    pass

        strategy = self.strategies[strategy_key]
        logger.info(f"Strategy: {strategy['name']} | pipeline: N4 → quantile histogram matching → {strategy['transform']}")
        logger.info(f"Parameters: flow_sigma=3")

        # Sub-menu: choose the operations to run (multiple selections allowed)
        print("\nChoose the operations to run (multiple selections allowed):")
        print("[1] ✅ Registration + automatic warp-field saving (required)")
        print("[2] 📐 Compute the Jacobian determinant (optional, requires a warp field)")
        print("[3] 🚀 Select all and run")
        print("[4] 🔙 Return to the previous menu")
        sel = input("\nEnter your choice (e.g. 1, or 1,2, or 3): ").strip()
        if not sel:
            print("Nothing selected; cancelling batch processing")
            return 0, []

        choices = set([s.strip() for s in sel.split(',') if s.strip().isdigit()])
        if '4' in choices:
            print("Returning to the previous menu")
            return 0, []

        # True multi-select: 1 = registration (and save warp), 2 = compute Jacobian (requires an existing warp), 3 = all
        do_register = ('1' in choices) or ('3' in choices)
        do_warp = ('1' in choices) or ('3' in choices)
        do_jacobian = ('2' in choices) or ('3' in choices)

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

        logger.info(f"\nDone: {success} ok | {skipped} skipped | {len(self.failed_files)} failed")
        logger.info(f"Elapsed: {elapsed/60:.1f} minutes")

        return success, self.results

    def post_process(self, resume=True):
        """Post-processing: additionally save the warp field or compute the Jacobian for already-registered files (supports batch operation and resumption)"""
        # Collect the already-registered files
        registered_files = sorted(self.dirs['registered'].glob("*.nii*"))
        if not registered_files:
            print("No registered files found")
            return

        # Find the files missing a warp or a jacobian
        need_warp = []
        need_jacobian = []
        for f in registered_files:
            file_id = Path(f).stem.replace('.nii', '')
            _, missing = check_output_exists(self.output_dir, file_id, required_files=['registered','warpfield','jacobian','overlay'])
            # If both warpfield and jacobian are missing, both need to be filled in
            if 'warpfield' in missing:
                need_warp.append(file_id)
            if 'jacobian' in missing:
                # If the warp exists but the jacobian is missing, or both are missing, the jacobian must also be computed
                need_jacobian.append(file_id)

        print(f"Warp fields to save: {len(need_warp)}; Jacobians to compute: {len(need_jacobian)}")
        print("\nChoose the operations to run (multiple selections allowed):")
        print("[1] � Compute Jacobian (for files that have a warp but no Jacobian)")
        print("[2] 📦 Additionally save the warp field (for registered files without a warp)")
        print("[3] 🚀 Select all and run")
        print("[4] 🔙 Return to the previous menu")
        sel = input("\nEnter your choice (e.g. 1, or 1,2, or 3): ").strip()
        if not sel:
            print("Nothing selected; cancelling")
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
            # Locate the original moving file
            mp = None
            for ext in ['.nii', '.nii.gz']:
                p = self.input_dir / f"{fid}{ext}"
                if p.exists():
                    mp = p
                    break
            if mp is None:
                logger.warning(f"Original input file not found; cannot rebuild the warp: {fid}")
                continue
            strat = strategy_map.get(fid, 'syn_standard')
            metrics, reg, warped, moving_mask = self.perform_registration(mp, strat, use_moving_mask=True)
            if reg is None:
                continue
            # When the user also chose to fill in the Jacobian, pass save_jacobian=do_jacobian
            self._save_deformation_fields(reg, fid, moving_mask, save_warp=do_warp, save_jacobian=do_jacobian)
            processed += 1

        for fid in need_jacobian:
            if resume:
                ok, miss = check_output_exists(self.output_dir, fid, required_files=['jacobian'])
                if ok:
                    continue
            # Warp file path
            warp_path = self.dirs['warpfields'] / f"{fid}_warp.nii.gz"
            if not warp_path.exists():
                logger.warning(f"Warp file missing; cannot compute the Jacobian: {fid}")
                continue
            # Directly invoke the jacobian branch of _save_deformation_fields
            fake_reg = {'fwdtransforms': [str(warp_path)]}
            # Compute the Jacobian using the template brain mask so that non-brain regions do not contaminate the statistics
            self._save_deformation_fields(fake_reg, fid, mask=self.template_mask, save_warp=False, save_jacobian=do_jacobian)
            processed += 1

        print(f"Post-processing complete: {processed}/{total} files processed")
    
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
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy': self.strategies[strategy_key]['name'],
            'pipeline': 'N4 correction → quantile histogram matching → SyN registration (flow_sigma=3)',
            'total_files': total,
            'ok': success,
            'skipped': skipped,
            'failed': len(self.failed_files),
            'minutes': round(total_time/60, 1),
            'mean_seconds': round(df['processing_time'].mean(), 1),
            'mean_dice': round(df['Dice'].mean(), 4),
            'mean_cc': round(df['CC'].mean(), 4),
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
        print("📊 Batch processing report")
        print("="*60)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("="*60)

# ============================ Main program ============================
def main():
    print("="*60)
    print("ADNI brain MRI batch registration system v3.2")
    print("Pipeline: N4 correction → quantile histogram matching → SyN registration (flow_sigma=3)")
    print("Features: preserves the [2%,98%] quantile band, protecting pathological information")
    print("="*60)
    
    if not Config.INPUT_DIR.exists():
        print(f"Error: input directory does not exist")
        return
    if not Config.TEMPLATE_PATH.exists():
        print(f"Error: template file does not exist")
        return
    
    pipeline = ADNIRegistrationPipeline(Config.INPUT_DIR, Config.TEMPLATE_PATH, Config.OUTPUT_DIR)
    
    if not pipeline.load_template():
        return
    
    input_files = sorted(
        list(Config.INPUT_DIR.glob("*.nii")) + list(Config.INPUT_DIR.glob("*.nii.gz"))
    )
    print(f"\nFound {len(input_files)} input files")
    
    if not input_files:
        return
    
    while True:
        print("\n" + "-"*50)
        print("1. 🔍 Parameter grid search")
        print("2. 🚀 Batch registration (opens a sub-menu)")
        print("3. 📊 View available strategies")
        print("4. 🧪 Single-file test")
        print("5. 📈 View completion status")
        print("6. 🔧 Post-processing (fill in warp field/Jacobian)")
        print("7. 🚪 Exit")
        print("-"*50)

        choice = input("Select (1-7): ").strip()
        
        if choice == '1':
            samples = int(input(f"Number of samples (1-3): ") or "2")
            combos = int(input(f"Number of strategies (1-6): ") or "4")
            pipeline.grid_search(min(samples, len(input_files)), min(combos, 6))
        
        elif choice == '2':
            grid = pipeline.dirs['reports'] / 'grid_search_results.csv'
            sk = None
            if grid.exists():
                try:
                    df = pd.read_csv(grid)
                    best = df.iloc[0]
                    print(f"\nBest from grid search: {best['name']} (CC={best['avg_cc']:.4f})")
                    if input("Use the best? (y/n): ").strip().lower() != 'n':
                        sk = best['strategy']
                except:
                    pass

            if sk is None:
                for i, (k, s) in enumerate(pipeline.strategies.items(), 1):
                    print(f"  {i}. {s['name']}")
                try:
                    idx = int(input(f"Select (1-{len(pipeline.strategies)}, default 2): ") or "2") - 1
                    sk = list(pipeline.strategies.keys())[idx]
                except:
                    sk = 'syn_standard'

            resume = input("Enable resumption? (y/n): ").strip().lower() != 'n'
            if input("Confirm processing? (y/n): ").strip().lower() == 'y':
                # The user selected the strategy manually in the interactive session, so automatic override is disabled (auto_select=False)
                pipeline.batch_process(strategy_key=sk, resume=resume, auto_select=False)
        
        elif choice == '3':
            for k, s in pipeline.strategies.items():
                print(f"\n  {s['name']} ({k})")
                print(f"    Parameters: {s['params']}")
        
        elif choice == '4':
            for i, f in enumerate(input_files[:10], 1):
                print(f"  {i}. {f.name}")
            try:
                idx = int(input("Select a file (1-10): ")) - 1
                tf = input_files[idx]
            except:
                tf = input_files[0]
            
            for i, (k, s) in enumerate(pipeline.strategies.items(), 1):
                print(f"  {i}. {s['name']}")
            try:
                idx = int(input(f"Select a strategy (1-{len(pipeline.strategies)}): ") or "2") - 1
                sk = list(pipeline.strategies.keys())[idx]
            except:
                sk = 'syn_standard'
            
            result = pipeline.register_single(tf, sk, save_outputs=True, check_existing=False)
            if result and not result.get('skipped'):
                print(f"\n  Dice={result['Dice']:.4f}, CC={result['CC']:.4f}, time={result['processing_time']:.1f}s")
        
        elif choice == '5':
            print(f"\nCompleted: {len(pipeline.completed_files)}")
            print(f"Failed: {len(pipeline.failed_files)}")
            if pipeline.results:
                actual = [r for r in pipeline.results if not r.get('skipped')]
                if actual:
                    df = pd.DataFrame(actual)
                    print(f"Mean Dice: {df['Dice'].mean():.4f}")
                    print(f"Mean CC: {df['CC'].mean():.4f}")

        elif choice == '6':
            # Post-processing: fill in the warp field and/or the Jacobian
            resume = input("Enable resumption? (y/n): ").strip().lower() != 'n'
            pipeline.post_process(resume=resume)

        elif choice == '7':
            break

if __name__ == "__main__":
    main()