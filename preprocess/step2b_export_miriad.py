# [Source] 01_data_and_preprocessing_paper2.1-2.2_S1/preprocessing_code/1.1.1batch_export_segmented_MR_for_miriad.py  (translated from the authors' archive path)
# [Paper correspondence] Segmentation export for the 69 MIRIAD cases (MIRIAD version of step2)
# [Note] This file is copied unchanged and its logic is untouched; before running, change the paths in the following lines to local paths:
#        Line numbers: [12]  (line numbers refer to the original Chinese version)

import os
import glob
import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, ornt_transform, apply_orientation
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

# Set the paths
BASE = r"<AUTHOR_DATA_ROOT>/datasets/miriad\miriad_lastly_classified\1"
LBL_DIR = os.path.join(BASE, "labels", "final")
OUT_DIR = os.path.join(BASE, "segmented")
os.makedirs(OUT_DIR, exist_ok=True)

def fix_z_flip_if_needed(label_data, label_img, raw_img):
    """
    Fix the Z-axis flip problem specifically
    """
    # Get the orientation information
    raw_ornt = io_orientation(raw_img.affine)
    label_ornt = io_orientation(label_img.affine)
    
    raw_axes = nib.orientations.ornt2axcodes(raw_ornt)
    label_axes = nib.orientations.ornt2axcodes(label_ornt)
    
    # Check whether the Z axis is flipped
    if raw_axes[2] != label_axes[2]:
        if (raw_axes[2] == 'Z' and label_axes[2] == 'Z-') or \
           (raw_axes[2] == 'Z-' and label_axes[2] == 'Z'):
            print(f"    Z-axis flip detected: {label_axes} -> {raw_axes}")
            # Flip the Z axis
            label_data = np.flip(label_data, axis=2)
            return label_data, True
    
    # Check whether a full reorientation is needed
    if not np.allclose(raw_ornt, label_ornt):
        # Use the full transform
        ornt_trans = ornt_transform(label_ornt, raw_ornt)
        label_data = apply_orientation(label_data, ornt_trans)
        return label_data, True
    
    return label_data, False

def safe_extract_data(img):
    """
    Safely extract the data, handling the 4D case
    """
    data = img.get_fdata()
    
    # If 4D, take the first time point
    if data.ndim == 4:
        if data.shape[-1] == 1:
            data = data[..., 0]
        else:
            print(f"    Warning: the 4D data has {data.shape[-1]} time points; only the first is taken")
            data = data[..., 0]
    
    return data

def process_image(raw_path, label_path, output_path):
    """
    Process a single image pair
    """
    try:
        # Load the images
        raw_img = nib.load(raw_path)
        label_img = nib.load(label_path)
        
        # Extract the data (use float32 to save memory)
        raw_data = safe_extract_data(raw_img).astype(np.float32)
        label_data = safe_extract_data(label_img).astype(np.float32)
        
        # Check the shapes
        if raw_data.shape != label_data.shape:
            print(f"    Shape mismatch: raw {raw_data.shape} vs label {label_data.shape}")
            return False, 'shape_mismatch'
        
        # Fix the Z-axis flip
        label_data, was_flipped = fix_z_flip_if_needed(label_data, label_img, raw_img)
        
        # Validate the label values
        unique_vals = np.unique(label_data)
        valid_vals = [0, 1]
        if not np.all(np.isin(unique_vals, valid_vals)):
            print(f"    Warning: the label contains values other than {0,1}: {unique_vals}")
            # Set values that are not 0 to 1 (assuming they are all brain tissue)
            label_data = (label_data > 0).astype(np.float32)
        
        # Check whether any brain tissue is present
        mask = (label_data == 1)
        if not np.any(mask):
            print(f"    Warning: the label has no voxel with value 1")
            return False, 'no_brain'
        
        # Apply the mask (memory-optimised)
        masked_data = np.zeros_like(raw_data)
        masked_data[mask] = raw_data[mask]
        
        # Handle invalid values
        if np.any(np.isnan(masked_data)) or np.any(np.isinf(masked_data)):
            print(f"    Warning: the data contains NaN/Inf; replacing with 0")
            masked_data = np.nan_to_num(masked_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Keep the original data type
        if raw_img.get_data_dtype() in [np.int16, np.int32, np.int64]:
            # Check the data range
            dtype_info = np.iinfo(raw_img.get_data_dtype())
            masked_data = np.clip(masked_data, dtype_info.min, dtype_info.max)
            masked_data = masked_data.astype(raw_img.get_data_dtype())
        else:
            masked_data = masked_data.astype(raw_img.get_data_dtype())
        
        # Create the new image (using the original affine matrix)
        new_img = nib.Nifti1Image(masked_data, raw_img.affine, raw_img.header)
        
        # Update the header information
        new_header = new_img.header.copy()
        non_zero = masked_data[masked_data != 0]
        if len(non_zero) > 0:
            new_header['cal_min'] = np.float64(np.min(non_zero))
            new_header['cal_max'] = np.float64(np.max(non_zero))
        
        # Save
        nib.save(new_img, output_path)
        return True, 'success'
        
    except Exception as e:
        print(f"    Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 'error'

def main():
    # Get the file list
    img_files = sorted(glob.glob(os.path.join(BASE, "*.nii.gz")))
    
    # Statistics
    stats = {
        'total': len(img_files),
        'matched': 0,
        'processed': 0,
        'skipped_no_label': 0,
        'skipped_shape_mismatch': [],
        'skipped_no_brain': [],
        'flipped_z': [],
        'reoriented': [],
        'errors': [],
        'bad_labels': []
    }
    
    print(f"Found {stats['total']} raw images")
    print("="*60)
    
    # Debug information (only the first file is shown)
    debug_done = False
    
    for i, img_path in enumerate(img_files, 1):
        name = os.path.basename(img_path)
        label_path = os.path.join(LBL_DIR, name)
        output_path = os.path.join(OUT_DIR, name)
        
        print(f"\n[{i}/{stats['total']}] Processing: {name}")
        
        # Check whether the label exists
        if not os.path.exists(label_path):
            stats['skipped_no_label'] += 1
            print("  skipped: the label file does not exist")
            continue
        
        stats['matched'] += 1
        
        # Check whether the output already exists (optional)
        if os.path.exists(output_path):
            print("  skipped: the output file already exists")
            continue
        
        # Process
        success, reason = process_image(img_path, label_path, output_path)
        
        if success:
            stats['processed'] += 1
            print(f"  ✓ saved successfully")
        else:
            if reason == 'shape_mismatch':
                stats['skipped_shape_mismatch'].append(name)
            elif reason == 'no_brain':
                stats['skipped_no_brain'].append(name)
            else:
                stats['errors'].append(name)
        
        # Debug information
        if not debug_done and os.path.exists(label_path):
            debug_done = True
            try:
                raw_img = nib.load(img_path)
                label_img = nib.load(label_path)
                raw_ornt = io_orientation(raw_img.affine)
                label_ornt = io_orientation(label_img.affine)
                raw_axes = nib.orientations.ornt2axcodes(raw_ornt)
                label_axes = nib.orientations.ornt2axcodes(label_ornt)
                print("\n[DEBUG] orientation information of the first file:")
                print(f"  raw image axes: {raw_axes}")
                print(f"  label axes: {label_axes}")
                print(f"  raw shape: {raw_img.shape}")
                print(f"  label shape: {label_img.shape}")
                print(f"  Z axis flipped: {raw_axes[2] != label_axes[2]}")
                print()
            except:
                pass
        
        # Periodically free memory
        if i % 10 == 0:
            gc.collect()
    
    # Print the statistics
    print("\n" + "="*60)
    print("Processing complete! Statistics:")
    print("="*60)
    print(f"Total files: {stats['total']}")
    print(f"Files with a label: {stats['matched']}")
    print(f"Processed successfully: {stats['processed']}")
    print(f"Skipped (no label): {stats['skipped_no_label']}")
    print(f"Shape mismatch: {len(stats['skipped_shape_mismatch'])}")
    print(f"No brain tissue: {len(stats['skipped_no_brain'])}")
    print(f"Processing errors: {len(stats['errors'])}")
    
    # Save the detailed report
    report_path = os.path.join(OUT_DIR, "processing_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Processing report\n")
        f.write("="*60 + "\n")
        f.write(f"Total files: {stats['total']}\n")
        f.write(f"Files with a label: {stats['matched']}\n")
        f.write(f"Processed successfully: {stats['processed']}\n")
        f.write(f"Skipped (no label): {stats['skipped_no_label']}\n\n")
        
        if stats['skipped_shape_mismatch']:
            f.write("Files with a shape mismatch:\n")
            for name in stats['skipped_shape_mismatch']:
                f.write(f"  {name}\n")
        
        if stats['skipped_no_brain']:
            f.write("\nFiles with no brain tissue:\n")
            for name in stats['skipped_no_brain']:
                f.write(f"  {name}\n")
        
        if stats['errors']:
            f.write("\nFiles that failed processing:\n")
            for name in stats['errors']:
                f.write(f"  {name}\n")
    
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    main()