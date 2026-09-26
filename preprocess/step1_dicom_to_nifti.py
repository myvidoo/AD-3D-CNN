# [Source] 01_data_and_preprocessing_paper2.1-2.2_S1/preprocessing_code/0.0.2DICOM_to_NIfTI.py  (translated from the authors' archive path)
# [Paper correspondence] 2.1/S1.1-S1.2 DICOM retrieval, series selection and dcm2niix conversion
# [Note] This file is copied unchanged and its logic is untouched; before running, change the paths in the following lines to local paths:
#        Line numbers: [29, 30, 31, 32, 42, 867, 868, 869, 872]  (line numbers refer to the original Chinese version)

import os
import sys
import json
import shutil
import subprocess
import hashlib
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import time

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration parameters
# ============================================================


class ConversionConfig:
    """Conversion configuration"""
    # Input and output paths
    data_root = r"<AUTHOR_DATA_ROOT>/datasets/ADNI\AD_MCI_CN_MR_3T_T1_ALL_dataset\ADNI"
    excel_file = r"<AUTHOR_DATA_ROOT>/MONAILabel/AD\selected_subjects_AD_MCI_CN_MR_3T_T1_ALL_9_30_2025.xlsx"
    output_root = r"<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized"
    dcm2niix_path = r"<AUTHOR_DATA_ROOT>/MONAILabel/AD\dcm2niix.exe"
    
    # Group mapping (for deep learning)
    GROUP_MAPPING = {
        'AD': '1',      # Alzheimer's disease
        'MCI': '0.5',   # mild cognitive impairment
        'CN': '0'       # cognitively normal
    }
    
    # dcm2niix path (specify the full path if it is not on PATH)
    dcm2niix_path = "dcm2niix"  # or r"C:\Program Files\dcm2niix\dcm2niix.exe"
    
    # dcm2niix conversion arguments
    dcm2niix_args = {
        'compress': 'y',      # compressed output (.nii.gz)
        'merge_slices': 'y',  # merge 2D slices into 3D
        'bids_json': 'y',     # save BIDS-format JSON metadata
        'verbose': 'n',       # verbose output
        'crop': 'n',          # do not crop
        'lossless': 'n',      # lossless scaling
    }
    
    # Processing control
    max_subjects: int = None  # limit the number processed; None means all
    skip_existing: bool = True  # skip existing output
    prefer_t1: bool = True   # prefer T1 series
    timeout_per_subject: int = 300  # timeout per subject (seconds)
    
    # T1-series recognition keywords
    T1_KEYWORDS = [
        'T1', 'T1W', 'T1-W', 'T1_', 'MPRAGE', 'SPGR', 'FSPGR', 
        'IR-SPGR', 'BRAVO', '3D TFE', 'VIBE', 'FLASH', 'TFE',
        'MPR', 'IR-FSPGR', 'T1W_3D'
    ]
    
    # Excluded keywords (non-T1 series)
    EXCLUDE_KEYWORDS = [
        'T2', 'T2W', 'T2-W', 'FLAIR', 'STIR', 'PD', 'DTI', 'DWI',
        'ADC', 'FA', 'MD', 'SWI', 'BOLD', 'ASL', 'Perfusion',
        'localizer', 'scout', 'calibration', 'reference'
    ]


# ============================================================
# Excel data loader
# ============================================================

class ExcelDataLoader:
    """Load and manage the subject information in the Excel file"""
    
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.df = None
        self.records = []  # store all records
        self.id_to_record = {}
        
    def load(self) -> bool:
        """Load the Excel file"""
        if not os.path.exists(self.excel_path):
            print(f"❌ Excel file does not exist: {self.excel_path}")
            return False
        
        try:
            self.df = pd.read_excel(self.excel_path)
            print(f"\n📋 Loading Excel file: {self.excel_path}")
            print(f"   Column names: {list(self.df.columns)}")
            print(f"   Total rows: {len(self.df)}")
            
            # Check the required columns
            required_cols = ['Image Data ID', 'Subject', 'Group']
            missing_cols = [c for c in required_cols if c not in self.df.columns]
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                return False
            
            # Parse every row
            for idx, row in self.df.iterrows():
                image_id = str(row['Image Data ID']).strip()
                if image_id and image_id.lower() != 'nan':
                    record = {
                        'Image Data ID': image_id,
                        'Subject': str(row.get('Subject', '')).strip(),
                        'Group': str(row.get('Group', '')).strip(),
                        'Sex': str(row.get('Sex', '')).strip(),
                        'Description': str(row.get('Description', '')).strip()
                    }
                    self.records.append(record)
                    self.id_to_record[image_id] = record
            
            print(f"   Valid records: {len(self.records)}")
            
            # Count the number in each group
            group_counts = Counter(r['Group'] for r in self.records)
            print(f"   Distribution by Group: {dict(group_counts)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load the Excel file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_record(self, image_id: str) -> Optional[Dict]:
        """Get the record for the given Image Data ID"""
        return self.id_to_record.get(image_id)
    
    def get_all_ids(self) -> List[str]:
        """Get all Image Data IDs"""
        return [r['Image Data ID'] for r in self.records]


# ============================================================
# Folder locator
# ============================================================

class FolderLocator:
    """Locate the DICOM folder corresponding to an Image Data ID"""
    
    def __init__(self, data_root: str, excel_loader: ExcelDataLoader):
        self.data_root = data_root
        self.excel_loader = excel_loader
        self.folder_index = {}  # folder name -> full path
        self.folder_mapping = {}  # Image Data ID -> folder path
        
    def build_index(self):
        """Build the folder index"""
        print("\n🔍 Building the folder index...")
        
        for root, dirs, files in os.walk(self.data_root):
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                self.folder_index[dir_name] = full_path
                
                # Also index the parent-directory + current-directory combination
                parent = os.path.basename(root)
                if parent:
                    combined = f"{parent}/{dir_name}"
                    self.folder_index[combined] = full_path
        
        print(f"   Indexed {len(self.folder_index)} folders")
    
    def locate_all(self) -> Dict[str, str]:
        """Locate the folder for every Image Data ID"""
        print("\n📍 Locating the folder for each Image Data ID...")
        
        image_ids = self.excel_loader.get_all_ids()
        
        for image_id in tqdm(image_ids, desc="locating folders"):
            found = False
            
            # Strategy 1: exact match
            if image_id in self.folder_index:
                self.folder_mapping[image_id] = self.folder_index[image_id]
                found = True
            
            # Strategy 2: containment match
            if not found:
                for dir_name, full_path in self.folder_index.items():
                    if image_id in dir_name or dir_name in image_id:
                        self.folder_mapping[image_id] = full_path
                        found = True
                        break
            
            # Strategy 3: path-containment match
            if not found:
                for dir_name, full_path in self.folder_index.items():
                    if image_id in full_path:
                        self.folder_mapping[image_id] = full_path
                        found = True
                        break
        
        print(f"   Found: {len(self.folder_mapping)}")
        print(f"   Not found: {len(image_ids) - len(self.folder_mapping)}")
        
        return self.folder_mapping


# ============================================================
# DICOM series selector
# ============================================================

class DICOMSeriesSelector:
    """Select the best series in a DICOM folder"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
    
    def _safe_get(self, ds, tag: str, default: Any = None) -> Any:
        """Safely read a DICOM tag"""
        try:
            if hasattr(ds, tag):
                value = getattr(ds, tag)
                return value if value is not None else default
            return default
        except:
            return default
    
    def _is_dicom(self, file_path: str) -> bool:
        """Check whether a file is a DICOM file"""
        try:
            with open(file_path, 'rb') as f:
                f.seek(128)
                return f.read(4) == b'DICM'
        except:
            return file_path.lower().endswith(('.dcm', '.dicom', '.ima', '.img'))
    
    def find_all_series(self, folder_path: str) -> Dict[str, List[str]]:
        """Find all DICOM series in the folder"""
        series_files = defaultdict(list)
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                if not self._is_dicom(file_path):
                    continue
                
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                    series_uid = self._safe_get(ds, 'SeriesInstanceUID', '')
                    if series_uid:
                        series_files[series_uid].append(file_path)
                except:
                    continue
        
        return series_files
    
    def analyze_series(self, series_files: Dict[str, List[str]]) -> Dict[str, Dict]:
        """Analyse the characteristics of each series"""
        series_info = {}
        
        for uid, files in series_files.items():
            if len(files) < 5:  # filter out series that are too small
                continue
            
            try:
                # Read the first file to obtain the metadata
                ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
                
                modality = str(self._safe_get(ds, 'Modality', ''))
                if modality != 'MR':
                    continue
                
                description = str(self._safe_get(ds, 'SeriesDescription', '')).upper()
                series_number = int(self._safe_get(ds, 'SeriesNumber', 9999) or 9999)
                
                # Check whether it is an excluded series
                is_excluded = any(kw in description for kw in self.config.EXCLUDE_KEYWORDS)
                
                # Check whether it is T1
                is_t1 = any(kw in description for kw in self.config.T1_KEYWORDS)
                
                # Get extra information
                manufacturer = str(self._safe_get(ds, 'Manufacturer', ''))
                field_strength = float(self._safe_get(ds, 'MagneticFieldStrength', 0) or 0)
                
                # Compute the score
                score = 0
                if is_t1:
                    score += 100
                if not is_excluded:
                    score += 50
                score += len(files) // 10  # bonus for a larger number of files
                if 'MPRAGE' in description or 'SPGR' in description:
                    score += 30
                if field_strength >= 3.0:
                    score += 20
                
                series_info[uid] = {
                    'uid': uid,
                    'description': description,
                    'modality': modality,
                    'is_t1': is_t1,
                    'is_excluded': is_excluded,
                    'num_files': len(files),
                    'series_number': series_number,
                    'manufacturer': manufacturer,
                    'field_strength': field_strength,
                    'score': score,
                    'sample_file': files[0]
                }
                
            except Exception as e:
                continue
        
        return series_info
    
    def select_best_series(self, folder_path: str) -> Optional[Tuple[str, List[str], Dict]]:
        """Select the best series for conversion"""
        series_files = self.find_all_series(folder_path)
        
        if not series_files:
            return None
        
        series_info = self.analyze_series(series_files)
        
        if not series_info:
            return None
        
        # Sort by score
        sorted_series = sorted(series_info.items(), 
                              key=lambda x: (x[1]['score'], x[1]['num_files']), 
                              reverse=True)
        
        if sorted_series:
            best_uid, best_info = sorted_series[0]
            return best_uid, series_files[best_uid], best_info
        
        return None


# ============================================================
# dcm2niix converter
# ============================================================

class DCM2NIIXConverter:
    """High-quality conversion using dcm2niix"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.dcm2niix_path = self._find_dcm2niix()
        
    def _find_dcm2niix(self) -> Optional[str]:
        """Locate the dcm2niix executable"""
        if self.config.dcm2niix_path:
            # Check the specified path
            if os.path.exists(self.config.dcm2niix_path):
                return self.config.dcm2niix_path
            if shutil.which(self.config.dcm2niix_path):
                return self.config.dcm2niix_path
        
        # Search the common paths
        common_paths = [
            r"C:\Program Files\dcm2niix\dcm2niix.exe",
            r"<TOOL_DIR>/dcm2niix\dcm2niix.exe",
            r"<TOOL_DIR>/dcm2niix\dcm2niix.exe",
            r"<TOOL_DIR>/dcm2niix\dcm2niix.exe",
            "/usr/bin/dcm2niix",
            "/usr/local/bin/dcm2niix",
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # Check PATH
        path_in_env = shutil.which("dcm2niix")
        if path_in_env:
            return path_in_env
        
        return None
    
    def is_available(self) -> bool:
        """Check whether dcm2niix is available"""
        if self.dcm2niix_path is None:
            print("❌ dcm2niix not found!")
            print("   Please download and install it: https://github.com/rordenlab/dcm2niix/releases")
            return False
        
        print(f"✅ found dcm2niix: {self.dcm2niix_path}")
        return True
    
    def convert(self, input_folder: str, output_folder: str, 
                output_basename: str) -> Tuple[bool, str, Optional[str]]:
        """
        Convert a DICOM folder with dcm2niix
        
        Returns:
            (success, message, output_file_path)
        """
        if not self.is_available():
            return False, "dcm2niix not available", None
        
        # Ensure that the output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Build the command
        cmd = [
            self.dcm2niix_path,
            "-z", self.config.dcm2niix_args['compress'],      # compressed output
            "-m", self.config.dcm2niix_args['merge_slices'],  # merge slices
            "-b", self.config.dcm2niix_args['bids_json'],     # BIDS JSON
            "-v", self.config.dcm2niix_args['verbose'],       # verbose output
            "-x", self.config.dcm2niix_args['crop'],          # do not crop
            "-f", output_basename,                            # output file name
            "-o", output_folder,                              # output directory
            input_folder                                       # input directory
        ]
        
        try:
            # Run the conversion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_per_subject,
                cwd=output_folder
            )
            
            if result.returncode == 0:
                # Find the generated files
                expected_file = os.path.join(output_folder, f"{output_basename}.nii.gz")
                if os.path.exists(expected_file):
                    return True, "Success", expected_file
                else:
                    # Try to find any generated .nii.gz file
                    for f in os.listdir(output_folder):
                        if f.endswith('.nii.gz') and output_basename in f:
                            return True, "Success", os.path.join(output_folder, f)
                    return False, f"Output file not found: {expected_file}", None
            else:
                return False, result.stderr, None
                
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {self.config.timeout_per_subject}s", None
        except Exception as e:
            return False, str(e), None


# ============================================================
# Batch converter
# ============================================================

class BatchOrganizedConverter:
    """Batch conversion with the output organised by Group"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.excel_loader = ExcelDataLoader(config.excel_file)
        self.folder_locator = None
        self.series_selector = DICOMSeriesSelector(config)
        self.converter = DCM2NIIXConverter(config)
        
        self.results = []
        self.failed = []
        self.skipped = []
        
    def _get_output_info(self, record: Dict) -> Tuple[str, str]:
        """
        Get the output directory and file name
        
        Directory structure: output_root/{group_label}/
        File name: {ImageID}_{Subject}_{Group}_{Sex}_{Description}.nii.gz
        """
        group = record['Group']
        group_label = self.config.GROUP_MAPPING.get(group, group)
        
        # Build the output directory
        output_dir = os.path.join(self.config.output_root, group_label)
        
        # Build the file name (cleaning illegal characters)
        def clean_filename(s: str) -> str:
            if not s:
                return ""
            # Replace illegal characters
            for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
                s = s.replace(char, '_')
            return s
        
        parts = [
            clean_filename(record['Image Data ID']),
            clean_filename(record['Subject']),
            clean_filename(record['Group']),
            clean_filename(record['Sex']),
            clean_filename(record['Description'])
        ]
        
        filename = "_".join(p for p in parts if p)
        
        return output_dir, filename
    
    def _process_folder(self, image_id: str, folder_path: str, 
                        record: Dict) -> Dict:
        """Process a single folder"""
        result = {
            'image_id': image_id,
            'folder_path': folder_path,
            'subject': record['Subject'],
            'group': record['Group'],
            'group_label': self.config.GROUP_MAPPING.get(record['Group'], record['Group']),
            'status': 'pending',
            'output_file': None,
            'error': None
        }
        
        # Get the output information
        output_dir, filename = self._get_output_info(record)
        output_path = os.path.join(output_dir, f"{filename}.nii.gz")
        result['output_file'] = output_path
        
        # Check whether it already exists
        if self.config.skip_existing and os.path.exists(output_path):
            result['status'] = 'skipped'
            return result
        
        # Select the best series
        best_series = self.series_selector.select_best_series(folder_path)
        
        if best_series is None:
            result['status'] = 'failed'
            result['error'] = 'No valid series found'
            return result
        
        series_uid, dcm_files, series_info = best_series
        result['series_info'] = series_info
        
        # Ensure that the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine the input directory (the directory holding the DICOM files)
        if dcm_files:
            input_dir = os.path.dirname(dcm_files[0])
        else:
            input_dir = folder_path
        
        # Run the conversion
        success, message, actual_output = self.converter.convert(
            input_dir, output_dir, filename
        )
        
        if success:
            result['status'] = 'success'
            result['actual_output'] = actual_output
            
            # Save the metadata
            self._save_metadata(output_dir, filename, record, series_info)
        else:
            result['status'] = 'failed'
            result['error'] = message
        
        return result
    
    def _save_metadata(self, output_dir: str, filename: str, 
                       record: Dict, series_info: Dict):
        """Save the conversion metadata"""
        metadata = {
            'conversion_date': datetime.now().isoformat(),
            'source': record,
            'series_info': {
                'description': series_info.get('description'),
                'num_files': series_info.get('num_files'),
                'manufacturer': series_info.get('manufacturer'),
                'field_strength': series_info.get('field_strength'),
                'is_t1': series_info.get('is_t1')
            },
            'output_filename': filename,
            'group_label': self.config.GROUP_MAPPING.get(record['Group'])
        }
        
        meta_path = os.path.join(output_dir, f"{filename}_meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
    
    def run(self):
        """Run the batch conversion"""
        print("\n" + "="*70)
        print("🔄 DICOM to NIfTI batch conversion (dcm2niix)")
        print("="*70)
        
        # Check dcm2niix
        if not self.converter.is_available():
            print("\nPlease install dcm2niix first:")
            print("  1. Download: https://github.com/rordenlab/dcm2niix/releases")
            print("  2. After unzipping, add the directory containing dcm2niix.exe to the system PATH")
            print("  3. Or specify the full path in the code")
            return
        
        # Load the Excel file
        print("\n📋 Step 1/6: loading the Excel file...")
        if not self.excel_loader.load():
            return
        
        # Build the folder index
        print("\n📁 Step 2/6: building the folder index...")
        self.folder_locator = FolderLocator(self.config.data_root, self.excel_loader)
        self.folder_locator.build_index()
        
        # Locate the folders
        print("\n📍 Step 3/6: locating the folders...")
        folder_mapping = self.folder_locator.locate_all()
        
        if not folder_mapping:
            print("❌ No folder was found")
            return
        
        # Prepare the output directory
        print("\n📂 Step 4/6: creating the output directory structure...")
        for group_label in set(self.config.GROUP_MAPPING.values()):
            os.makedirs(os.path.join(self.config.output_root, group_label), exist_ok=True)
        print(f"   Created {len(self.config.GROUP_MAPPING)} group directories")
        for group, label in self.config.GROUP_MAPPING.items():
            print(f"     {group} -> {label}")
        
        # Run the conversion
        print("\n🔄 Step 5/6: running the conversion...")
        
        items_to_process = list(folder_mapping.items())
        if self.config.max_subjects:
            items_to_process = items_to_process[:self.config.max_subjects]
        
        print(f"   {len(items_to_process)} subjects will be processed")
        
        for image_id, folder_path in tqdm(items_to_process, desc="conversion progress"):
            record = self.excel_loader.get_record(image_id)
            if record is None:
                record = {
                    'Image Data ID': image_id,
                    'Subject': 'Unknown',
                    'Group': 'Unknown',
                    'Sex': '',
                    'Description': ''
                }
            
            result = self._process_folder(image_id, folder_path, record)
            self.results.append(result)
            
            if result['status'] == 'failed':
                self.failed.append(result)
            elif result['status'] == 'skipped':
                self.skipped.append(result)
        
        # Generate the report
        print("\n📊 Step 6/6: generating the report...")
        self._generate_report()
        
        # Print the summary
        self._print_summary()
    
    def _generate_report(self):
        """Generate the conversion report"""
        report_dir = os.path.join(self.config.output_root, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        # Prepare the DataFrame
        report_data = []
        for r in self.results:
            report_data.append({
                'Image ID': r['image_id'],
                'Subject': r['subject'],
                'Group': r['group'],
                'Group Label': r['group_label'],
                'Status': r['status'],
                'Output File': r.get('output_file', ''),
                'Error': r.get('error', '')
            })
        
        df = pd.DataFrame(report_data)
        
        # Save the Excel report
        excel_path = os.path.join(report_dir, f"conversion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # All results
            df.to_excel(writer, sheet_name='All Results', index=False)
            
            # Summary by status
            summary_by_status = df.groupby('Status').size().reset_index(name='Count')
            summary_by_status.to_excel(writer, sheet_name='By Status', index=False)
            
            # Summary by Group
            summary_by_group = df.groupby(['Group', 'Group Label', 'Status']).size().unstack(fill_value=0)
            summary_by_group.to_excel(writer, sheet_name='By Group')
            
            # Failed records
            failed_df = df[df['Status'] == 'failed']
            if not failed_df.empty:
                failed_df.to_excel(writer, sheet_name='Failed', index=False)
        
        print(f"\n📄 Report saved: {excel_path}")
        
        # Save the detailed JSON
        json_path = os.path.join(report_dir, f"conversion_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'data_root': self.config.data_root,
                    'output_root': self.config.output_root,
                    'group_mapping': self.config.GROUP_MAPPING
                },
                'summary': {
                    'total': len(self.results),
                    'success': sum(1 for r in self.results if r['status'] == 'success'),
                    'failed': len(self.failed),
                    'skipped': len(self.skipped)
                },
                'results': self.results
            }, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"📄 JSON details saved: {json_path}")
    
    def _print_summary(self):
        """Print the conversion summary"""
        success = sum(1 for r in self.results if r['status'] == 'success')
        failed = len(self.failed)
        skipped = len(self.skipped)
        total = len(self.results)
        
        print("\n" + "="*70)
        print("📊 Conversion-complete statistics")
        print("="*70)
        print(f"  ✅ succeeded: {success}")
        print(f"  ❌ failed: {failed}")
        print(f"  ⏭️ skipped: {skipped}")
        print(f"  📁 total: {total}")
        
        # Statistics by Group
        print("\n📈 Statistics by Group:")
        group_stats = defaultdict(lambda: {'success': 0, 'failed': 0, 'skipped': 0})
        for r in self.results:
            group_stats[r['group']][r['status']] += 1
        
        for group, stats in sorted(group_stats.items()):
            label = self.config.GROUP_MAPPING.get(group, group)
            print(f"  {group} ({label}): succeeded={stats['success']}, failed={stats['failed']}, skipped={stats['skipped']}")
        
        # Output directory structure
        print(f"\n📁 Output directory structure:")
        print(f"  {self.config.output_root}/")
        for label in sorted(set(self.config.GROUP_MAPPING.values())):
            dir_path = os.path.join(self.config.output_root, label)
            if os.path.exists(dir_path):
                file_count = len([f for f in os.listdir(dir_path) if f.endswith('.nii.gz')])
                print(f"    ├── {label}/ ({file_count} files)")
        print(f"    └── reports/")
        
        # Failure details
        if self.failed:
            print(f"\n⚠️ Failure details (first 5):")
            for r in self.failed[:5]:
                print(f"    - {r['image_id']}: {r.get('error', 'Unknown error')[:50]}")


# ============================================================
# Validation utility
# ============================================================

class OutputValidator:
    """Validate the conversion output"""
    
    @staticmethod
    def validate_nifti(file_path: str) -> Dict:
        """Validate a NIfTI file"""
        try:
            import nibabel as nib
            
            img = nib.load(file_path)
            
            return {
                'valid': True,
                'shape': img.shape,
                'voxel_size': img.header.get_zooms()[:3],
                'data_type': img.get_data_dtype().name,
                'affine_valid': not np.allclose(img.affine, np.eye(4))
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    @staticmethod
    def validate_all_outputs(output_root: str, group_mapping: Dict) -> pd.DataFrame:
        """Validate all output files"""
        results = []
        
        for group, label in group_mapping.items():
            dir_path = os.path.join(output_root, label)
            if not os.path.exists(dir_path):
                continue
            
            for f in os.listdir(dir_path):
                if f.endswith('.nii.gz'):
                    file_path = os.path.join(dir_path, f)
                    validation = OutputValidator.validate_nifti(file_path)
                    validation['file'] = f
                    validation['group'] = group
                    validation['label'] = label
                    results.append(validation)
        
        return pd.DataFrame(results)


# ============================================================
# Main program
# ============================================================

def main():
    """Main function"""
    config = ConversionConfig()
    
    print("="*70)
    print("🔄 DICOM to NIfTI batch conversion tool (dcm2niix)")
    print("="*70)
    print(f"Data root: {config.data_root}")
    print(f"Excel file: {config.excel_file}")
    print(f"Output root: {config.output_root}")
    print(f"Group mapping: {config.GROUP_MAPPING}")
    print("="*70)
    
    # Run the conversion
    converter = BatchOrganizedConverter(config)
    converter.run()
    
    # Optional: validate the output
    print("\n🔍 Validating the output files...")
    df_validation = OutputValidator.validate_all_outputs(
        config.output_root, 
        config.GROUP_MAPPING
    )
    
    if not df_validation.empty:
        valid_count = df_validation['valid'].sum()
        total_count = len(df_validation)
        print(f"   Validation result: {valid_count}/{total_count} files valid")
        
        if valid_count < total_count:
            invalid_files = df_validation[~df_validation['valid']]
            print(f"   ⚠️ invalid files: {len(invalid_files)}")
            for _, row in invalid_files.head(5).iterrows():
                print(f"      - {row['file']}: {row.get('error', 'Unknown')}")
    
    print("\n✅ Conversion complete!")


# ============================================================
# Usage example
# ============================================================

def run_example():
    """Run the example"""
    # Create the configuration
    config = ConversionConfig()
    
    # Change the paths to your actual paths
    config.data_root = r"<AUTHOR_DATA_ROOT>/datasets/ADNI\AD_MCI_CN_MR_3T_T1_ALL_dataset\ADNI"
    config.excel_file = r"<AUTHOR_DATA_ROOT>/MONAILabel/AD\selected_subjects_AD_MCI_CN_MR_3T_T1_ALL_9_30_2025.xlsx"
    config.output_root = r"<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized"
    
    # If dcm2niix is not on PATH, specify the full path
    # config.dcm2niix_path = r"C:\Program Files\dcm2niix\dcm2niix.exe"
    
    # Limit the number processed (for testing)
    config.max_subjects = 10
    
    # Run the conversion
    converter = BatchOrganizedConverter(config)
    converter.run()
    
    return converter.results


if __name__ == "__main__":
    main()
    # Or run the example
    # results = run_example()