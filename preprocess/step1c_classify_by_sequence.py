# [Source] 01_data_and_preprocessing_paper2.1-2.2_S1/preprocessing_code/0.0.8classify_by_sequence_for_segmentation.py  (translated from the authors' archive path)
# [Paper correspondence] S1.1 series classification (grouping the examination types before skull stripping)
# [Note] This file is copied unchanged and its logic is untouched; before running, change the paths in the following lines to local paths:
#        Line numbers: [501]  (line numbers refer to the original Chinese version)

import os
import re
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Fix the environment-variable problem
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

def extract_description(filename):
    """Extract the Description from the file name"""
    if filename.endswith('.nii.gz'):
        name_without_ext = filename[:-7]
    else:
        name_without_ext = filename
    
    match = re.search(r'_(F|M)_(.+)$', name_without_ext)
    if match:
        return match.group(2), match.group(1)
    return None, None

def analyze_nifti_file(filepath):
    """Analyse the pixel-value distribution of a NIfTI file"""
    try:
        img = nib.load(str(filepath))
        data = img.get_fdata()
        
        # Analyse only non-zero pixels (excluding the background)
        non_zero_data = data[data != 0]
        
        if len(non_zero_data) == 0:
            return None
        
        # Downsample to reduce the computational cost
        if len(non_zero_data) > 50000:
            np.random.seed(42)
            indices = np.random.choice(len(non_zero_data), 50000, replace=False)
            sampled_data = non_zero_data[indices]
        else:
            sampled_data = non_zero_data
        
        # Compute various statistical features
        features = {
            'mean': np.mean(sampled_data),
            'std': np.std(sampled_data),
            'median': np.median(sampled_data),
            'min': np.min(sampled_data),
            'max': np.max(sampled_data),
            'skewness': stats.skew(sampled_data),
            'kurtosis': stats.kurtosis(sampled_data),
            'percentile_5': np.percentile(sampled_data, 5),
            'percentile_25': np.percentile(sampled_data, 25),
            'percentile_75': np.percentile(sampled_data, 75),
            'percentile_95': np.percentile(sampled_data, 95),
            'total_voxels': len(non_zero_data),
        }
        
        return features, sampled_data
    except Exception as e:
        print(f"    Failed to read file {filepath.name}: {e}")
        return None

def simple_kmeans(X, n_clusters, max_iter=300):
    """A simple K-means implementation"""
    np.random.seed(42)
    
    # Standardise the data
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    X_norm = (X - mean) / std
    
    # Initialise the cluster centres (k-means++)
    n_samples = X_norm.shape[0]
    centers = []
    
    first_center_idx = np.random.randint(n_samples)
    centers.append(X_norm[first_center_idx])
    
    for _ in range(1, n_clusters):
        distances = np.array([min([np.sum((x - c) ** 2) for c in centers]) for x in X_norm])
        probs = distances / distances.sum()
        cumulative_probs = np.cumsum(probs)
        r = np.random.random()
        
        for i, cum_prob in enumerate(cumulative_probs):
            if r < cum_prob:
                centers.append(X_norm[i])
                break
    
    centers = np.array(centers)
    
    # Iterative optimisation
    for iteration in range(max_iter):
        distances = np.zeros((n_samples, n_clusters))
        for i, center in enumerate(centers):
            distances[:, i] = np.sum((X_norm - center) ** 2, axis=1)
        
        labels = np.argmin(distances, axis=1)
        
        new_centers = np.zeros_like(centers)
        for i in range(n_clusters):
            cluster_points = X_norm[labels == i]
            if len(cluster_points) > 0:
                new_centers[i] = cluster_points.mean(axis=0)
            else:
                new_centers[i] = X_norm[np.random.randint(n_samples)]
        
        if np.allclose(centers, new_centers):
            break
        
        centers = new_centers
    
    # Compute the inertia
    final_distances = np.zeros((n_samples, n_clusters))
    for i, center in enumerate(centers):
        final_distances[:, i] = np.sum((X_norm - center) ** 2, axis=1)
    inertia = np.sum(np.min(final_distances, axis=1))
    
    return labels, inertia

def calculate_silhouette_score(X, labels):
    """Compute the silhouette score"""
    n_samples = X.shape[0]
    
    if len(np.unique(labels)) < 2:
        return -1
    
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    X_norm = (X - mean) / std
    
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        distances[i] = np.sqrt(np.sum((X_norm - X_norm[i]) ** 2, axis=1))
    
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        same_cluster = labels == labels[i]
        same_cluster[i] = False
        
        if np.sum(same_cluster) == 0:
            silhouette_vals[i] = 0
            continue
        
        a_i = np.mean(distances[i][same_cluster])
        
        b_i = np.inf
        for label in np.unique(labels):
            if label != labels[i]:
                other_cluster = labels == label
                mean_dist = np.mean(distances[i][other_cluster])
                if mean_dist < b_i:
                    b_i = mean_dist
        
        if max(a_i, b_i) == 0:
            silhouette_vals[i] = 0
        else:
            silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
    
    return np.mean(silhouette_vals)

def perform_clustering(base_path, original_groups):
    """Run the clustering analysis"""
    print("\n" + "="*80)
    print("Clustering analysis stage")
    print("="*80)
    
    # Extract representative files for analysis
    print("\nExtracting a representative file from each category for analysis...")
    
    sample_features = {}
    all_descriptions = sorted(original_groups.keys())
    
    for i, desc in enumerate(all_descriptions, 1):
        representative_file = original_groups[desc][0]
        print(f"  [{i}/{len(all_descriptions)}] analysing: {desc}")
        
        result = analyze_nifti_file(representative_file)
        if result:
            features, data = result
            sample_features[desc] = {
                'features': features,
                'data': data,
                'file': representative_file.name,
                'count': len(original_groups[desc])
            }
    
    if len(sample_features) < 2:
        print("Insufficient number of samples; clustering analysis cannot be performed")
        return None
    
    # Prepare the clustering features
    print("\nPreparing the clustering features...")
    
    feature_names = ['mean', 'std', 'median', 'skewness', 'kurtosis', 
                    'percentile_5', 'percentile_25', 'percentile_75', 'percentile_95']
    
    X = []
    desc_list = []
    for desc in sorted(sample_features.keys()):
        features = sample_features[desc]['features']
        feature_vector = [features[name] for name in feature_names]
        X.append(feature_vector)
        desc_list.append(desc)
    
    X = np.array(X)
    
    # Test different numbers of clusters
    max_clusters = min(12, len(desc_list) - 1)
    
    print(f"\nTesting a cluster-count range from 2 to {max_clusters}")
    print("-"*60)
    print(f"{'Clusters':<10} {'Inertia':<15} {'Silhouette':<15}")
    print("-"*60)
    
    inertias = []
    silhouette_scores_list = []
    
    for n_clusters in range(2, max_clusters + 1):
        labels, inertia = simple_kmeans(X, n_clusters)
        sil_score = calculate_silhouette_score(X, labels)
        
        inertias.append(inertia)
        silhouette_scores_list.append(sil_score)
        
        print(f"{n_clusters:<10} {inertia:<15.2f} {sil_score:<15.4f}")
    
    # Find the optimal number of clusters
    valid_scores = [s for s in silhouette_scores_list if s > -1]
    if valid_scores:
        optimal_idx = silhouette_scores_list.index(max(valid_scores))
        optimal_clusters = optimal_idx + 2
        print("-"*60)
        print(f"\nOptimal number of clusters: {optimal_clusters} (silhouette score: {silhouette_scores_list[optimal_idx]:.4f})")
    else:
        optimal_clusters = 2
        print("\nThe optimal number of clusters could not be determined; using the default value 2")
    
    # Run the final clustering
    print(f"\nUsing {optimal_clusters} clusters for the final classification...")
    final_labels, final_inertia = simple_kmeans(X, optimal_clusters)
    
    # Organise the clustering results
    clusters = defaultdict(list)
    for desc, label in zip(desc_list, final_labels):
        clusters[label].append({
            'description': desc,
            'count': sample_features[desc]['count'],
            'features': sample_features[desc]['features']
        })
    
    # Show the clustering results
    print("\nClustering result:")
    print("="*80)
    
    merge_map = {}
    
    for cluster_id in sorted(clusters.keys()):
        members = clusters[cluster_id]
        total_files = sum(m['count'] for m in members)
        
        # Compute the average features
        avg_features = {}
        for feat_name in feature_names:
            avg_features[feat_name] = np.mean([m['features'][feat_name] for m in members])
        
        # Build the cluster name
        main_desc = max(members, key=lambda x: x['count'])['description']
        cluster_name = f"Cluster_{cluster_id+1}_{main_desc[:40]}"
        # Clean special characters out of the folder name
        cluster_name = cluster_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')
        
        print(f"\nCluster {cluster_id + 1}: {cluster_name}")
        print(f"  Total files: {total_files}")
        print(f"  Number of categories included: {len(members)}")
        print(f"  Mean pixel mean: {avg_features['mean']:.2f}")
        print(f"  Mean pixel SD: {avg_features['std']:.2f}")
        print(f"  Descriptions included:")
        
        for member in sorted(members, key=lambda x: x['description']):
            desc = member['description']
            count = member['count']
            merge_map[desc] = cluster_name
            print(f"    - {desc} ({count} files)")
    
    # Generate the visualisation
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow-method plot
        ax1 = axes[0]
        ax1.plot(range(2, len(inertias) + 2), inertias, 'bo-')
        ax1.set_xlabel('Number of clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        ax1.axvline(x=optimal_clusters, color='r', linestyle='--', alpha=0.5)
        
        # Silhouette-score plot
        ax2 = axes[1]
        valid_indices = [i for i, s in enumerate(silhouette_scores_list) if s > -1]
        valid_clusters_list = [i + 2 for i in valid_indices]
        valid_scores_clean = [silhouette_scores_list[i] for i in valid_indices]
        
        ax2.plot(valid_clusters_list, valid_scores_clean, 'ro-')
        ax2.set_xlabel('Number of clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        ax2.axvline(x=optimal_clusters, color='g', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        output_path = base_path / 'clustering_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nClustering-analysis figure saved to: {output_path}")
        plt.show()
    except Exception as e:
        print(f"\nFailed to generate the visualisation: {e}")
    
    return merge_map, optimal_clusters

def safe_copy(src, dst):
    """Safely copy a file"""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        if dst.exists():
            if dst.stat().st_size == src.stat().st_size:
                return True, "skipped"
        
        shutil.copy2(src, dst)
        return True, "copied"
    except Exception as e:
        return False, str(e)

def process_files(base_path, merge_map):
    """Process the files according to the merge map"""
    print("\n" + "="*80)
    print("File processing and copying stage")
    print("="*80)
    
    # Collect all qualifying files
    print("\nCollecting files...")
    all_files = []
    for file in base_path.iterdir():
        if file.is_file() and file.name.endswith('.nii.gz') and file.stat().st_size > 2 * 1024 * 1024:
            all_files.append(file)
    
    print(f"Found {len(all_files)} files in total")
    
    # Group by the new clusters
    cluster_groups = defaultdict(list)
    unclassified_files = []
    
    for file_path in all_files:
        desc, sex = extract_description(file_path.name)
        if desc and desc in merge_map:
            cluster_name = merge_map[desc]
            cluster_groups[cluster_name].append(file_path)
        else:
            unclassified_files.append(file_path)
    
    if unclassified_files:
        print(f"\nWarning: {len(unclassified_files)} files could not be classified:")
        for f in unclassified_files[:5]:
            print(f"  - {f.name}")
        if len(unclassified_files) > 5:
            print(f"  ... {len(unclassified_files)-5} more files")
    
    print(f"\nFile grouping complete: {len(cluster_groups)} groups:")
    for cluster_name in sorted(cluster_groups.keys()):
        print(f"  {cluster_name}: {len(cluster_groups[cluster_name])} files")
    
    # Confirm whether to continue
    response = input("\nStart copying the files? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled by the user")
        return
    
    # Process each cluster group
    stats = {
        'total_files': 0,
        'copied_main': 0,
        'copied_labels': 0,
        'skipped_main': 0,
        'skipped_labels': 0,
        'errors': []
    }
    
    for cluster_name in sorted(cluster_groups.keys()):
        files = cluster_groups[cluster_name]
        print(f"\n{'='*60}")
        print(f"Processing: {cluster_name}")
        print(f"Number of files: {len(files)}")
        
        # Create the target folder
        target_folder = base_path / f"1_{cluster_name}"
        target_folder.mkdir(exist_ok=True)
        
        # Copy the main files
        copied_filenames = []
        for file_path in files:
            dest_path = target_folder / file_path.name
            success, status = safe_copy(file_path, dest_path)
            
            if success:
                if status == "copied":
                    stats['copied_main'] += 1
                else:
                    stats['skipped_main'] += 1
                copied_filenames.append(file_path.name)
            else:
                stats['errors'].append(f"Copy failed {file_path.name}: {status}")
        
        stats['total_files'] += len(files)
        print(f"Main files: copied {stats['copied_main']}, skipped {stats['skipped_main']}")
        
        # Process the labels
        if copied_filenames:
            labels_final_folder = target_folder / "labels" / "final"
            labels_final_folder.mkdir(parents=True, exist_ok=True)
            
            source_labels_final = base_path / "labels" / "final"
            
            if source_labels_final.exists():
                matched = 0
                for filename in copied_filenames:
                    source_label = source_labels_final / filename
                    if source_label.exists() and source_label.stat().st_size < 1 * 1024 * 1024:
                        dest_label = labels_final_folder / filename
                        success, status = safe_copy(source_label, dest_label)
                        if success:
                            if status == "copied":
                                stats['copied_labels'] += 1
                            else:
                                stats['skipped_labels'] += 1
                            matched += 1
                
                print(f"Labels: matched {matched}/{len(copied_filenames)}")
            else:
                print(f"Warning: the labels/final folder does not exist")
    
    # Final verification
    print("\n" + "="*80)
    print("Final verification")
    print("="*80)
    
    verification_passed = True
    
    # Check all original files
    all_original_names = {f.name for f in all_files}
    all_copied_names = set()
    
    for cluster_name in cluster_groups.keys():
        target_folder = base_path / f"1_{cluster_name}"
        if target_folder.exists():
            for f in target_folder.iterdir():
                if f.is_file() and f.name.endswith('.nii.gz'):
                    all_copied_names.add(f.name)
    
    missing_files = all_original_names - all_copied_names
    extra_files = all_copied_names - all_original_names
    
    if missing_files:
        print(f"❌ missing files ({len(missing_files)}):")
        for f in sorted(missing_files)[:10]:
            print(f"  - {f}")
        verification_passed = False
    
    if extra_files:
        print(f"⚠️ extra files ({len(extra_files)}):")
        for f in sorted(extra_files)[:10]:
            print(f"  - {f}")
    
    if verification_passed:
        print("✅ verification passed! All files were processed correctly")
    else:
        print("❌ verification found problems; please check the information above")
    
    # Print the statistics
    print(f"\nProcessing statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Main files copied: {stats['copied_main']}")
    print(f"  Main files skipped: {stats['skipped_main']}")
    print(f"  labels copied: {stats['copied_labels']}")
    print(f"  labels skipped: {stats['skipped_labels']}")
    if stats['errors']:
        print(f"  Number of errors: {len(stats['errors'])}")

def main():
    base_path = Path(r"<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized_monailabel_test\0.5")
    
    if not base_path.exists():
        print(f"Error: the path {base_path} does not exist")
        return
    
    print("="*80)
    print("Medical-image file intelligent classification system")
    print("="*80)
    print(f"Working directory: {base_path}")
    
    # Stage one: collect and parse the files
    print("\nStage one: file scanning and parsing")
    print("-"*40)
    
    target_files = []
    for file in base_path.iterdir():
        if file.is_file() and file.name.endswith('.nii.gz') and file.stat().st_size > 2 * 1024 * 1024:
            target_files.append(file)
    
    print(f"Found {len(target_files)} qualifying .nii.gz files (>2MB)")
    
    # Group by the original Description
    original_groups = defaultdict(list)
    unparsed_files = []
    
    for file_path in target_files:
        desc, sex = extract_description(file_path.name)
        if desc:
            original_groups[desc].append(file_path)
        else:
            unparsed_files.append(file_path)
    
    print(f"Parsed {len(original_groups)} distinct Description categories")
    
    if unparsed_files:
        print(f"Warning: {len(unparsed_files)} files could not be parsed")
    
    # Show an overview of the original categories
    print("\nOverview of the original categories:")
    for desc in sorted(original_groups.keys()):
        print(f"  {desc}: {len(original_groups[desc])} files")
    
    # Stage two: clustering analysis
    print("\n" + "="*80)
    print("Stage two: intelligent clustering analysis")
    print("="*80)
    
    response = input("\nRun the clustering analysis to merge similar categories? (y/n, default y): ").strip().lower()
    if response == 'n':
        print("Skipping the clustering analysis; keeping the original categories")
        # Create a simple 1-to-1 mapping
        merge_map = {desc: desc for desc in original_groups.keys()}
    else:
        # Run the clustering analysis
        merge_map, n_clusters = perform_clustering(base_path, original_groups)
        
        if merge_map is None:
            print("Clustering analysis failed; the original categories will be used")
            merge_map = {desc: desc for desc in original_groups.keys()}
        else:
            print("\n" + "="*80)
            print("Clustering analysis complete; please confirm the merge scheme")
            print("="*80)
            
            # Show the merge map
            final_clusters = defaultdict(list)
            for desc, cluster in merge_map.items():
                final_clusters[cluster].append(desc)
            
            for cluster_name in sorted(final_clusters.keys()):
                print(f"\n{cluster_name}:")
                for desc in final_clusters[cluster_name]:
                    print(f"  - {desc}")
            
            # Ask whether to accept
            print("\nOptions:")
            print("1. accept this clustering scheme and continue")
            print("2. reject it and use the original 30 categories")
            print("3. exit the program")
            
            choice = input("Please choose (1/2/3, default 1): ").strip()
            
            if choice == '2':
                print("Using the original classification scheme")
                merge_map = {desc: desc for desc in original_groups.keys()}
            elif choice == '3':
                print("Exiting the program")
                return
            else:
                print("Continuing with the clustering scheme")
    
    # Stage three: file processing
    print("\n" + "="*80)
    print("Stage three: file processing and copying")
    print("="*80)
    
    process_files(base_path, merge_map)
    
    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by the user")
    except Exception as e:
        print(f"\nError while running the program: {e}")
        import traceback
        traceback.print_exc()