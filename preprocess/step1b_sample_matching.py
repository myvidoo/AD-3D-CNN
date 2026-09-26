# [Source] 01_data_and_preprocessing_paper2.1-2.2_S1/preprocessing_code/0.0.4.0match_three_group_samples_by_AD_group_revised.py  (translated from the authors' archive path)
# [Paper correspondence] 2.1/S1.4 greedy matching with a hard sex constraint and soft age optimisation (318 cases per group)
# [Note] This file is copied unchanged and its logic is untouched; before running, change the paths in the following lines to local paths:
#        Line numbers: [20, 21]  (line numbers refer to the original Chinese version)

# Revised version that allows duplicate files to be submitted; it adds a smarter file-saving mechanism and an improved matching algorithm, so that results are still saved successfully when a file is in use, and clear error messages and suggestions are given when the sample pool is insufficient.
'''
✅ _fallback_fill has been fixed:

The Level 3 duplicate-filling logic np.random.choice(..., replace=True) was removed
When the candidate pool is exhausted and a shortfall remains, a ValueError carrying
diagnostic information is raised
Level 1 / Level 2 are kept, filling only from the unused sample pool
The class interface, the match method, the file-saving functions and the main flow are unchanged
'''
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
import time
warnings.filterwarnings('ignore')

# Set the paths
main_folder = r"<AUTHOR_DATA_ROOT>/datasets/ADNI/NIfTI_organized_monailabel_test_sample_selection_review"
csv_path = r"<AUTHOR_DATA_ROOT>/datasets/ADNI\AD_MCI_CN_MR_3T_T1_ALL_dataset\AD_MCI_CN_MR_3T_T1_ALL_9_30_2025.csv"

def safe_save_excel(df, filepath, max_retries=3):
    """Safely save the Excel file, handling permission errors"""
    for attempt in range(max_retries):
        try:
            # Check whether the file is in use
            if os.path.exists(filepath):
                try:
                    # Try renaming the file to check whether it is locked
                    temp_name = filepath + '.temp'
                    os.rename(filepath, temp_name)
                    os.rename(temp_name, filepath)
                except PermissionError:
                    print(f"  ⚠ file is in use: {filepath}")
                    if attempt < max_retries - 1:
                        print(f"  retrying after 3 seconds (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(3)
                        continue
                    else:
                        # Last resort: use a fallback file name
                        base, ext = os.path.splitext(filepath)
                        backup_path = f"{base}_new{ext}"
                        print(f"  using a fallback file name: {backup_path}")
                        df.to_excel(backup_path, index=False)
                        return backup_path
            
            # Try to save
            df.to_excel(filepath, index=False)
            print(f"  ✓ saved: {filepath}")
            return filepath
            
        except PermissionError as e:
            print(f"  ✗ permission error: {e}")
            if attempt < max_retries - 1:
                print(f"  retrying after 3 seconds (attempt {attempt + 1}/{max_retries})...")
                time.sleep(3)
            else:
                # Create a new file name using a timestamp
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                base, ext = os.path.splitext(filepath)
                backup_path = f"{base}_{timestamp}{ext}"
                print(f"  using a timestamped file name: {backup_path}")
                try:
                    df.to_excel(backup_path, index=False)
                    print(f"  ✓ saved to the fallback location: {backup_path}")
                    return backup_path
                except:
                    print(f"  ✗ could not save the file; close the Excel file manually and try again")
                    return None

def scan_target_files(main_folder):
    """Scan the target folder and extract the Image Data IDs"""
    target_folders = ['0', '0.5', '1']
    image_data_ids = []
    file_info = []
    
    for folder_name in target_folders:
        folder_path = os.path.join(main_folder, folder_name)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.nii.gz'):
                    file_path = os.path.join(folder_path, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    if file_size > 2:
                        # Extract the Image Data ID from the file name (the part before the first underscore)
                        image_data_id = file.split('_')[0]
                        image_data_ids.append(image_data_id)
                        file_info.append({
                            'file_name': file,
                            'file_path': file_path,
                            'file_size_mb': round(file_size, 2),
                            'image_data_id': image_data_id,
                            'source_folder': folder_name
                        })
    
    print(f"Found {len(image_data_ids)} qualifying .nii.gz files (>2MB)")
    if image_data_ids:
        print(f"Example Image Data ID: {image_data_ids[:3]}")
    return image_data_ids, file_info

def create_summary1(image_data_ids, csv_path, main_folder):
    """Search the CSV for the Image Data IDs and create subject_index_20260531"""
    
    # Read the CSV file
    df = pd.read_csv(csv_path, encoding='latin1')
    
    # Clean the column names - remove quotes and whitespace
    df.columns = [col.strip().strip('"').strip("'") for col in df.columns]
    
    print(f"\nCSV file information:")
    print(f"Total rows: {len(df)}")
    print(f"Column names: {list(df.columns)}")
    
    # Find the Image Data ID column
    id_column = None
    for col in df.columns:
        if 'Image Data ID' in col or col == 'Image Data ID':
            id_column = col
            break
    
    if id_column is None:
        print(f"Warning: the 'Image Data ID' column was not found; using the first column: {df.columns[0]}")
        id_column = df.columns[0]
    else:
        print(f"Found the Image Data ID column: '{id_column}'")
    
    # Clean the ID column
    df[id_column] = df[id_column].astype(str).str.strip()
    image_data_ids = [str(id).strip() for id in image_data_ids]
    
    # Filter the matching rows
    matched_rows = df[df[id_column].isin(image_data_ids)]
    
    print(f"Exact matches: {len(matched_rows)} records")
    
    if len(matched_rows) > 0:
        # Look at the Group distribution
        if 'Group' in matched_rows.columns:
            print(f"Group distribution of the matches:")
            print(matched_rows['Group'].value_counts())
    
    # Save subject_index_20260531 - using the safe-save function
    output_path = os.path.join(main_folder, 'subject_index_20260531.xlsx')
    saved_path = safe_save_excel(matched_rows, output_path)
    
    if saved_path:
        print(f"\nsaved subject_index_20260531: {saved_path}")
    else:
        print(f"\n⚠ saving subject_index_20260531 failed, but the data is in memory and processing will continue")
    
    return matched_rows

class ImprovedGreedyMatcherWithFallback:
    """Improved greedy matcher (with a forced-fill mechanism)"""
    
    def __init__(self, ad_group, target_group, sex_col, age_col, desc_col, group_name=""):
        self.ad_group = ad_group.copy()
        self.target_group = target_group.copy()
        self.sex_col = sex_col
        self.age_col = age_col
        self.desc_col = desc_col
        self.group_name = group_name
        
        # Clean the data
        self._clean_data()
        
        # Analyse the sex distribution of the AD group
        if len(self.ad_group) > 0 and self.sex_col:
            self.ad_sex_distribution = self.ad_group[self.sex_col].value_counts()
            self.target_count = len(self.ad_group)
            print(f"  [{self.group_name}] AD-group sex distribution: {self.ad_sex_distribution.to_dict()}")
        else:
            self.ad_sex_distribution = pd.Series()
            self.target_count = 0
    
    def _clean_data(self):
        """Clean and standardise the data"""
        # Standardise the sex values
        if self.sex_col:
            sex_mapping = {
                'M': 'M', 'Male': 'M', 'male': 'M', '1': 'M', 1: 'M',
                'F': 'F', 'Female': 'F', 'female': 'F', '0': 'F', 0: 'F'
            }
            
            self.ad_group[self.sex_col] = self.ad_group[self.sex_col].astype(str).str.strip().map(sex_mapping)
            self.target_group[self.sex_col] = self.target_group[self.sex_col].astype(str).str.strip().map(sex_mapping)
            
            # Drop invalid sex values
            self.ad_group = self.ad_group[self.ad_group[self.sex_col].notna()]
            self.target_group = self.target_group[self.target_group[self.sex_col].notna()]
        
        # Prepare the age data
        if self.age_col:
            self.ad_group[self.age_col] = pd.to_numeric(self.ad_group[self.age_col], errors='coerce')
            self.target_group[self.age_col] = pd.to_numeric(self.target_group[self.age_col], errors='coerce')
            
            # Drop invalid age values
            self.ad_group = self.ad_group[self.ad_group[self.age_col].notna()]
            self.target_group = self.target_group[self.target_group[self.age_col].notna()]
    
    def _calculate_age_score(self, target_ages, ad_ages):
        """Compute the age-matching score"""
        if len(target_ages) == 0 or len(ad_ages) == 0:
            return 0
        
        try:
            scores = []
            
            # Mean-difference score
            mean_diff = abs(np.mean(target_ages) - np.mean(ad_ages))
            mean_score = np.exp(-mean_diff / 5)
            scores.append(mean_score * 0.4)
            
            # SD-difference score
            std_diff = abs(np.std(target_ages) - np.std(ad_ages))
            std_score = np.exp(-std_diff / 3)
            scores.append(std_score * 0.2)
            
            # Distribution-similarity score
            if len(target_ages) > 1 and len(ad_ages) > 1:
                ks_stat, ks_pvalue = stats.ks_2samp(target_ages, ad_ages)
                scores.append(ks_pvalue * 0.4)
            else:
                scores.append(0.2)
            
            return sum(scores)
        except:
            return 0.5
    
    def _evaluate_combination(self, candidate_indices):
        """Evaluate the overall score of one candidate combination"""
        if len(candidate_indices) == 0:
            return 0
            
        candidates = self.target_group.loc[candidate_indices]
        
        if self.age_col and self.age_col in candidates.columns:
            ad_ages = self.ad_group[self.age_col].values
            target_ages = candidates[self.age_col].values
            age_score = self._calculate_age_score(target_ages, ad_ages)
        else:
            age_score = 0.5
        
        total_score = age_score * 0.7 + 0.3
        return total_score
    
    def _fallback_fill(self, matched_indices, required_count, preferred_sex=None):
        """Forced-fill mechanism"""
        current_count = len(matched_indices)
        shortfall = required_count - current_count
        
        if shortfall <= 0:
            return matched_indices
        
        print(f"  [{self.group_name}] ⚠ forced fill triggered: {shortfall} samples need to be added")
        
        remaining_pool = self.target_group[~self.target_group.index.isin(matched_indices)]
        
        if len(remaining_pool) == 0:
            message = (
                f"[{self.group_name}] no candidate samples remain; filling cannot be completed."
                f" target={required_count}, matched={current_count}, shortfall={shortfall}."
                f" Suggestion: check the target-group sample size, or adjust the matching strategy/relax the matching constraints."
            )
            raise ValueError(message)
        
        fill_indices = []
        
        # Level 1: sex matching
        if preferred_sex and self.sex_col:
            sex_matched_pool = remaining_pool[remaining_pool[self.sex_col] == preferred_sex]
            
            if len(sex_matched_pool) > 0:
                n_fill = min(shortfall, len(sex_matched_pool))
                
                if self.age_col and len(sex_matched_pool) > n_fill:
                    ad_mean_age = self.ad_group[self.age_col].mean()
                    sex_matched_pool = sex_matched_pool.copy()
                    sex_matched_pool['age_distance'] = abs(
                        sex_matched_pool[self.age_col] - ad_mean_age
                    )
                    fills = sex_matched_pool.nsmallest(n_fill, 'age_distance')
                else:
                    fills = sex_matched_pool.sample(n=n_fill, random_state=42)
                
                fill_indices.extend(fills.index.tolist())
                shortfall -= n_fill
                remaining_pool = remaining_pool[~remaining_pool.index.isin(fills.index)]
                print(f"  [{self.group_name}] Level 1: filled {n_fill} by sex matching")
        
        # Level 2: ignore sex
        if shortfall > 0:
            remaining_pool = remaining_pool[~remaining_pool.index.isin(fill_indices)]
            if len(remaining_pool) == 0:
                message = (
                    f"[{self.group_name}] no candidate samples remain after Level 2; filling cannot be completed."
                    f" target={required_count}, matched={current_count + len(fill_indices)}, shortfall={shortfall}."
                    f" Suggestion: check the target-group sample size, or adjust the matching strategy/relax the matching constraints."
                )
                raise ValueError(message)
            
            n_fill = min(shortfall, len(remaining_pool))
            
            if self.age_col and len(remaining_pool) > n_fill:
                ad_mean_age = self.ad_group[self.age_col].mean()
                remaining_pool = remaining_pool.copy()
                remaining_pool['age_distance'] = abs(
                    remaining_pool[self.age_col] - ad_mean_age
                )
                fills = remaining_pool.nsmallest(n_fill, 'age_distance')
            else:
                fills = remaining_pool.sample(n=n_fill, random_state=42)
            
            fill_indices.extend(fills.index.tolist())
            shortfall -= n_fill
            print(f"  [{self.group_name}] Level 2: downgraded fill of {n_fill} (sex constraint ignored)")
        
        if shortfall > 0:
            message = (
                f"[{self.group_name}] the matching candidate pool is insufficient; filling cannot be completed."
                f" target={required_count}, matched={current_count + len(fill_indices)}, shortfall={shortfall}."
                f" Suggestion: increase the number of samples in the target group or relax the matching constraints."
            )
            raise ValueError(message)
        
        matched_indices.extend(fill_indices)
        return matched_indices
    
    def match(self):
        """Run the improved greedy matching (with forced fill)"""
        
        if self.target_count == 0:
            print(f"\n[{self.group_name}] the AD group is empty; skipping matching")
            return pd.DataFrame()
        
        if len(self.target_group) == 0:
            print(f"\n[{self.group_name}] the target group is empty; matching is impossible")
            return pd.DataFrame()
        
        print(f"\n[{self.group_name}] starting matching:")
        print(f"  AD-group sample size: {self.target_count}")
        print(f"  Available samples in the target group: {len(self.target_group)}")
        
        matched_indices = []
        
        # Get the sex requirements of the AD group
        if self.sex_col and self.sex_col in self.ad_group.columns:
            ad_sex_counts = self.ad_group[self.sex_col].value_counts()
            
            for sex in ['M', 'F']:
                ad_sex_count = ad_sex_counts.get(sex, 0)
                if ad_sex_count == 0:
                    continue
                
                target_sex_pool = self.target_group[self.target_group[self.sex_col] == sex]
                
                print(f"  sex {sex}: {ad_sex_count} needed, {len(target_sex_pool)} available")
                
                sex_matched = []
                
                if len(target_sex_pool) >= ad_sex_count:
                    if len(target_sex_pool) == ad_sex_count:
                        sex_matched = target_sex_pool.index.tolist()
                        print(f"    exactly matched {len(sex_matched)}")
                    else:
                        # Greedy search for the best combination
                        best_score = -1
                        best_combination = None
                        
                        n_attempts = min(1000, len(target_sex_pool) * 10)
                        for _ in range(n_attempts):
                            candidates = np.random.choice(target_sex_pool.index, 
                                                        size=ad_sex_count, 
                                                        replace=False)
                            score = self._evaluate_combination(candidates)
                            
                            if score > best_score:
                                best_score = score
                                best_combination = candidates
                        
                        sex_matched = best_combination.tolist() if best_combination is not None else []
                        print(f"    greedy selection of {len(sex_matched)}, score: {best_score:.3f}")
                else:
                    # Not enough: use up all of them first
                    sex_matched = target_sex_pool.index.tolist()
                    print(f"    insufficient available; using up {len(sex_matched)} first")
                    # Forced fill
                    sex_matched = self._fallback_fill(sex_matched, ad_sex_count, preferred_sex=sex)
                
                matched_indices.extend(sex_matched)
        else:
            # No sex column: select at random
            print(f"  no sex column found; selecting {self.target_count} samples at random")
            if len(self.target_group) >= self.target_count:
                matched_indices = np.random.choice(self.target_group.index, 
                                                  size=self.target_count, 
                                                  replace=False).tolist()
            else:
                matched_indices = self.target_group.index.tolist()
                matched_indices = self._fallback_fill(matched_indices, self.target_count)
        
        # Make sure the count is correct
        current_count = len(matched_indices)
        if current_count < self.target_count:
            print(f"  ⚠ insufficient count ({current_count}/{self.target_count}); forced fill...")
            matched_indices = self._fallback_fill(matched_indices, self.target_count)
        elif current_count > self.target_count:
            matched_indices = matched_indices[:self.target_count]
        
        matched_df = self.target_group.loc[matched_indices]
        
        print(f"  ✓ matching result: {len(matched_df)} subjects")
        if self.sex_col and self.sex_col in matched_df.columns:
            print(f"  sex distribution: {matched_df[self.sex_col].value_counts().to_dict()}")
        if self.age_col and self.age_col in matched_df.columns:
            ages = pd.to_numeric(matched_df[self.age_col], errors='coerce')
            ad_ages = pd.to_numeric(self.ad_group[self.age_col], errors='coerce')
            if len(ages) > 0 and len(ad_ages) > 0:
                print(f"  age: {ages.mean():.1f}±{ages.std():.1f} (AD: {ad_ages.mean():.1f}±{ad_ages.std():.1f})")
        
        return matched_df

def create_summary2_with_fallback(summary1_df, main_folder):
    """Create subject_summary_20260531 with the improved greedy algorithm with forced fill"""
    
    df = summary1_df.copy()
    
    # Clean the column names
    df.columns = [col.strip().strip('"').strip("'") for col in df.columns]
    
    # Find the column names
    group_col = 'Group'
    sex_col = 'Sex'
    age_col = 'Age'
    desc_col = 'Description'
    
    # Verify that the columns exist
    for col_name, col_var in [('Group', group_col), ('Sex', sex_col), 
                               ('Age', age_col), ('Description', desc_col)]:
        if col_var not in df.columns:
            print(f"⚠ Warning: the '{col_name}' column was not found")
            # Try fuzzy matching
            for col in df.columns:
                if col_name.lower() in col.lower():
                    print(f"  using column: '{col}' as '{col_name}'")
                    if col_name == 'Group':
                        group_col = col
                    elif col_name == 'Sex':
                        sex_col = col
                    elif col_name == 'Age':
                        age_col = col
                    elif col_name == 'Description':
                        desc_col = col
    
    # Clean the Group values
    df[group_col] = df[group_col].astype(str).str.strip().str.upper()
    
    print(f"\nGroup value distribution:")
    print(df[group_col].value_counts())
    
    # Group by Group
    ad_group = df[df[group_col] == 'AD']
    cn_group = df[df[group_col] == 'CN']
    mci_group = df[df[group_col] == 'MCI']
    
    print(f"\nGrouping result:")
    print(f"AD group: {len(ad_group)} subjects")
    print(f"CN group: {len(cn_group)} subjects")
    print(f"MCI group: {len(mci_group)} subjects")
    
    if len(ad_group) == 0:
        print("\n❌ Error: the AD group is empty! Matching cannot be performed.")
        return None
    
    # Show the details of the AD group
    if sex_col in ad_group.columns:
        print(f"\nAD-group sex distribution:")
        print(ad_group[sex_col].value_counts())
    if age_col in ad_group.columns:
        ad_ages = pd.to_numeric(ad_group[age_col], errors='coerce')
        print(f"AD-group age: {ad_ages.mean():.1f} ± {ad_ages.std():.1f}")
    
    # Match the CN group
    print("\n" + "="*50)
    cn_matcher = ImprovedGreedyMatcherWithFallback(
        ad_group, cn_group, sex_col, age_col, desc_col, group_name="CN"
    )
    matched_cn = cn_matcher.match()
    
    # Match the MCI group
    print("\n" + "="*50)
    mci_matcher = ImprovedGreedyMatcherWithFallback(
        ad_group, mci_group, sex_col, age_col, desc_col, group_name="MCI"
    )
    matched_mci = mci_matcher.match()
    
    # Merge the results
    final_df = pd.concat([ad_group, matched_cn, matched_mci], ignore_index=True)
    
    # Save subject_summary_20260531
    output_path = os.path.join(main_folder, 'subject_summary_20260531.xlsx')
    saved_path = safe_save_excel(final_df, output_path)
    
    print(f"\n{'='*50}")
    print(f"✅ Final result:")
    print(f"  AD group: {len(ad_group)} subjects")
    print(f"  CN group: {len(matched_cn)} subjects")
    print(f"  MCI group: {len(matched_mci)} subjects")
    print(f"  Total: {len(final_df)} subjects")
    
    # Generate the detailed report
    report_path = os.path.join(main_folder, 'matching_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ADNI data matching report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Generated at: {pd.Timestamp.now()}\n\n")
        
        f.write("Sample counts:\n")
        f.write(f"  AD group:  {len(ad_group)} subjects\n")
        f.write(f"  CN group:  {len(matched_cn)} subjects\n")
        f.write(f"  MCI group: {len(matched_mci)} subjects\n")
        f.write(f"  Total:  {len(final_df)} subjects\n\n")
        
        # Sex distribution
        if sex_col in final_df.columns:
            f.write("Sex distribution:\n")
            for group_name, group_df in [('AD', ad_group), ('CN', matched_cn), ('MCI', matched_mci)]:
                sex_dist = group_df[sex_col].value_counts()
                f.write(f"  {group_name} group: ")
                for sex, count in sex_dist.items():
                    f.write(f"{sex}={count} ")
                f.write("\n")
            f.write("\n")
        
        # Age statistics
        if age_col in final_df.columns:
            f.write("Age statistics:\n")
            for group_name, group_df in [('AD', ad_group), ('CN', matched_cn), ('MCI', matched_mci)]:
                ages = pd.to_numeric(group_df[age_col], errors='coerce')
                if len(ages) > 0:
                    f.write(f"  {group_name} group: {ages.mean():.2f} ± {ages.std():.2f} years")
                    f.write(f" (range: {ages.min():.0f}-{ages.max():.0f})\n")
            f.write("\n")
        
        # Matching description
        f.write("Matching strategy:\n")
        f.write("  1. Use the AD group as the baseline and make the CN and MCI group sizes equal to it\n")
        f.write("  2. Match the sex distribution as a priority (hard constraint)\n")
        f.write("  3. Given the sex match, optimise the age distribution (soft optimisation)\n")
        f.write("  4. When good-quality samples are insufficient, use a graded filling strategy\n")
    
    print(f"\nMatching report saved: {report_path}")
    
    return final_df

def main():
    print("="*60)
    print("ADNI data matching script")
    print("="*60)
    print(f"\nNote: make sure the output files are not open (subject_index_20260531.xlsx, subject_summary_20260531.xlsx)")
    print("If a file is in use, the script automatically uses a fallback file name\n")
    
    # Step 1: scan the files
    print("Step 1: scanning the target files...")
    image_data_ids, file_info = scan_target_files(main_folder)
    
    if len(image_data_ids) == 0:
        print("❌ Error: no qualifying .nii.gz files (>2MB) were found")
        return
    
    # Step 2: create subject_index_20260531
    print("\nStep 2: creating subject_index_20260531...")
    summary1_df = create_summary1(image_data_ids, csv_path, main_folder)
    
    if len(summary1_df) == 0:
        print("❌ Error: subject_index_20260531 is empty; cannot continue")
        return
    
    # Step 3: match and create subject_summary_20260531
    print("\nStep 3: performing the matching...")
    final_df = create_summary2_with_fallback(summary1_df, main_folder)
    
    if final_df is not None:
        print("\n" + "="*60)
        print("✅ All processing complete!")
        print(f"Output file location: {main_folder}")
        print(f"  - subject_index_20260531.xlsx (or subject_index_20260531_new.xlsx)")
        print(f"  - subject_summary_20260531.xlsx (or subject_summary_20260531_new.xlsx)")
        print(f"  - matching_report.txt")

if __name__ == "__main__":
    main()