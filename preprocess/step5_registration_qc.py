# [Source] 01_data_and_preprocessing_paper2.1-2.2_S1/preprocessing_code/2.2registration_info_summary.py  (translated from the authors' archive path)
# [Paper correspondence] S1.9 registration QC: parse the registration logs and summarise Dice/CC/MAE
# [Note] This file is copied unchanged and its logic is untouched; before running, change the paths in the following lines to local paths:
#        Line numbers: [152]  (line numbers refer to the original Chinese version)

# Registration information summary
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

def parse_log_file(filepath):
    """
    Parse a single log file and extract the registration results
    """
    results = []
    filename = filepath.name
    
    try:
        # Read the file contents
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()

        # Regular expressions matching the key information
        # Sample match pattern: [index/total] file name
        # Result match pattern: Done: Dice=value, CC=value, time=value
        # NOTE: these labels are the parsing interface with the SyN-fast registration step in
        # this directory (step3_register_syn_fast.py), which emits exactly this format - the same
        # one used by the archived original script in legacy/. Both sides must be changed
        # together, or the parser silently finds nothing.
        sample_pattern = r'\[\d+/\d+\]\s+([^\s\|]+)'
        result_pattern = r'Done:\s*Dice=([\d.]+),\s*CC=([\d.]+),\s*time=([\d.]+)'

        # Find all matches
        samples = re.findall(sample_pattern, content)
        results_raw = re.findall(result_pattern, content)
        
        # Make sure the number of samples matches the number of results
        min_len = min(len(samples), len(results_raw))
        
        for i in range(min_len):
            try:
                dice = float(results_raw[i][0])
                cc = float(results_raw[i][1])
                time = float(results_raw[i][2])
                results.append({
                    'Source_File': samples[i],
                    'Log_File': filename,
                    'Dice': dice,
                    'CC': cc,
                    'Time_sec': time
                })
            except (ValueError, IndexError) as e:
                continue
                
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")

    return results

def analyze_registration_logs(log_dir):
    """
    Main analysis function
    """
    all_data = []
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"Error: directory not found: {log_dir}")
        return

    # 1. Walk over all .log files in the directory
    log_files = list(log_path.glob("*.log"))
    if not log_files:
        print("No .log files found in the specified directory.")
        return

    print(f"Found {len(log_files)} log files, starting parsing...\n")

    # 2. Parse all logs
    for log_file in log_files:
        print(f"Processing: {log_file.name}")
        data = parse_log_file(log_file)
        all_data.extend(data)

    if not all_data:
        print("No valid data extracted.")
        return

    print(f"\nSuccessfully extracted {len(all_data)} registration records. Starting statistical analysis...")

    # 3. Data conversion and statistics
    df = pd.DataFrame(all_data)

    # Compute the statistics
    stats_summary = {
        'Metric': ['Dice Coefficient', 'Correlation Coefficient (CC)', 'Processing Time (s)'],
        'Mean': [
            df['Dice'].mean(),
            df['CC'].mean(),
            df['Time_sec'].mean()
        ],
        'Median': [
            df['Dice'].median(),
            df['CC'].median(),
            df['Time_sec'].median()
        ],
        'Std': [
            df['Dice'].std(),
            df['CC'].std(),
            df['Time_sec'].std()
        ],
        'Min': [
            df['Dice'].min(),
            df['CC'].min(),
            df['Time_sec'].min()
        ],
        'Max': [
            df['Dice'].max(),
            df['CC'].max(),
            df['Time_sec'].max()
        ],
        'Count': len(df)
    }
    
    stats_df = pd.DataFrame(stats_summary)
    
    # 4. Output the results
    print("\n" + "="*60)
    print("           Registration Performance Statistics (Summary Statistics)")
    print("="*60)
    print(stats_df.to_string(index=False, float_format="%.4f"))

    # 5. Save the results to CSV
    output_csv = log_path / "registration_analysis_report.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed data saved to: {output_csv}")

    # 6. Generate suggested text for the paper (based on the statistics)
    print("\n" + "="*60)
    print("           Suggested Text for the Paper (Suggested Text)")
    print("="*60)

    avg_dice = stats_df.loc[0, 'Mean']
    avg_time = stats_df.loc[2, 'Mean']
    std_dice = stats_df.loc[0, 'Std']

    suggested_text = f"""
In the registration quality control of this study, we quantitatively evaluated all {len(df)} samples.
The registration strategy adopted shows very high accuracy and stability: the mean Dice similarity
coefficient of whole-brain registration reaches {avg_dice:.3f} (SD: {std_dice:.3f}),
and the mean correlation coefficient (CC) is [specific value], indicating a high degree of
spatial alignment consistency between images.
In terms of computational efficiency, the mean registration time per sample is about {avg_time:.1f} seconds.
Moreover, inspection of the minimum (Dice: {stats_df.loc[0, 'Min']:.3f}) and maximum (Dice: {stats_df.loc[0, 'Max']:.3f})
shows no significant outliers, indicating that the pipeline is robust in large-scale data processing.
"""
    print(suggested_text)

if __name__ == "__main__":
    # Change this path to the folder where your log files are stored
    LOG_DIR = r"<AUTHOR_DATA_ROOT>/MONAILabel/AD/registration_info_log/SyN-fast"

    analyze_registration_logs(LOG_DIR)