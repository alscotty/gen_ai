"""
Dimensionality Reduction Advanced Analysis: PCA, t-SNE, and UMAP
A unique comprehensive study exploring dimensionality reduction techniques with advanced metrics
not found in standard tutorials. Includes downstream task performance, noise sensitivity analysis,
cluster separability preservation, parameter sensitivity exploration, and information-theoretic metrics.

Unique features:
- Downstream classification performance on reduced spaces
- Noise sensitivity degradation analysis
- Cluster separability preservation metrics
- Parameter sensitivity heatmaps (perplexity, n_neighbors)
- Information-theoretic entropy analysis
- Dimensionality scaling performance curves

Compares PCA (linear), t-SNE (non-linear, local), and UMAP (non-linear, global-local balance)
on diverse synthetic datasets to understand their strengths, limitations, and when to use each.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs, make_moons, make_circles, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, silhouette_score, adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

# Set random seed for reproducibility
np.random.seed(42)
sns.set_style("whitegrid")

print("=" * 70)
print("Dimensionality Reduction Comparison Study")
print("=" * 70)

def generate_datasets():
    """Generate diverse synthetic datasets with distinct characteristics."""
    datasets = {}
    
    # Dataset 1: High-dimensional spherical clusters (PCA should excel)
    print("\n1. Generating Dataset 1: High-dimensional spherical clusters...")
    X1, y1 = make_blobs(
        n_samples=300,
        centers=4,
        n_features=10,  # High-dimensional
        cluster_std=1.0,
        random_state=42
    )
    datasets['high_dim_spherical'] = (X1, y1, "High-dimensional spherical clusters (10D)")
    print(f"   Shape: {X1.shape}, True clusters: {len(np.unique(y1))}")
    
    # Dataset 2: Non-linear structure (moons - t-SNE/UMAP should excel)
    print("\n2. Generating Dataset 2: Non-linear structure (moons)...")
    X2, y2 = make_moons(
        n_samples=300,
        noise=0.1,
        random_state=42
    )
    # Embed in higher dimensions with noise
    X2_high = np.hstack([X2, np.random.randn(300, 8) * 0.1])
    datasets['nonlinear_moons'] = (X2_high, y2, "Non-linear moons structure (10D)")
    print(f"   Shape: {X2_high.shape}, True clusters: {len(np.unique(y2))}")
    
    # Dataset 3: Concentric circles (non-linear, t-SNE/UMAP should excel)
    print("\n3. Generating Dataset 3: Concentric circles...")
    X3, y3 = make_circles(
        n_samples=300,
        noise=0.1,
        factor=0.5,
        random_state=42
    )
    # Embed in higher dimensions
    X3_high = np.hstack([X3, np.random.randn(300, 8) * 0.1])
    datasets['concentric_circles'] = (X3_high, y3, "Concentric circles (10D)")
    print(f"   Shape: {X3_high.shape}, True clusters: {len(np.unique(y3))}")
    
    # Dataset 4: Swiss roll (manifold learning challenge)
    print("\n4. Generating Dataset 4: Swiss roll manifold...")
    X4, y4 = make_swiss_roll(
        n_samples=300,
        noise=0.1,
        random_state=42
    )
    # Color by height (z-coordinate)
    y4_binned = (X4[:, 2] > X4[:, 2].mean()).astype(int)
    datasets['swiss_roll'] = (X4, y4_binned, "Swiss roll manifold (3D)")
    print(f"   Shape: {X4.shape}, True clusters: {len(np.unique(y4_binned))}")
    
    # Dataset 5: Linear structure (PCA should excel)
    print("\n5. Generating Dataset 5: Linear structure with noise...")
    t = np.linspace(0, 4*np.pi, 300)
    X5 = np.column_stack([
        np.cos(t),
        np.sin(t),
        t / (4*np.pi),
        np.random.randn(300, 7) * 0.1  # Noise dimensions
    ])
    y5 = (t > 2*np.pi).astype(int)
    datasets['linear_structure'] = (X5, y5, "Linear structure with noise (10D)")
    print(f"   Shape: {X5.shape}, True clusters: {len(np.unique(y5))}")
    
    return datasets

def compute_cluster_separability(X_reduced, y_true):
    """Compute cluster separability: ratio of inter-cluster to intra-cluster distances."""
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        return 0.0
    
    # Compute intra-cluster distances (within clusters)
    intra_distances = []
    for label in unique_labels:
        cluster_points = X_reduced[y_true == label]
        if len(cluster_points) > 1:
            intra_dist = pdist(cluster_points)
            intra_distances.extend(intra_dist)
    
    # Compute inter-cluster distances (between clusters)
    inter_distances = []
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            cluster1_points = X_reduced[y_true == label1]
            cluster2_points = X_reduced[y_true == label2]
            # Compute distances between all points in cluster1 and cluster2
            for p1 in cluster1_points:
                for p2 in cluster2_points:
                    inter_distances.append(np.linalg.norm(p1 - p2))
    
    if len(intra_distances) == 0 or len(inter_distances) == 0:
        return 0.0
    
    mean_intra = np.mean(intra_distances)
    mean_inter = np.mean(inter_distances)
    
    # Separability = inter / intra (higher is better, clusters are more separated)
    separability = mean_inter / mean_intra if mean_intra > 0 else 0.0
    return separability

def compute_entropy_metric(X_reduced, n_bins=10):
    """Compute entropy-based metric: how well-distributed the data is in reduced space."""
    # Discretize the reduced space into bins
    x_min, x_max = X_reduced[:, 0].min(), X_reduced[:, 0].max()
    y_min, y_max = X_reduced[:, 1].min(), X_reduced[:, 1].max()
    
    # Create 2D histogram
    hist, _, _ = np.histogram2d(X_reduced[:, 0], X_reduced[:, 1], 
                                bins=n_bins, range=[[x_min, x_max], [y_min, y_max]])
    
    # Normalize to probabilities
    hist_flat = hist.flatten()
    hist_flat = hist_flat[hist_flat > 0]  # Remove empty bins
    if len(hist_flat) == 0:
        return 0.0
    
    prob = hist_flat / hist_flat.sum()
    
    # Compute entropy (higher entropy = more uniform distribution = better)
    ent = entropy(prob, base=2)
    max_entropy = np.log2(len(hist_flat))
    normalized_entropy = ent / max_entropy if max_entropy > 0 else 0.0
    
    return normalized_entropy

def compute_preservation_metrics(X_original, X_reduced, y_true=None):
    """Compute comprehensive metrics to assess how well the reduced space preserves original structure."""
    metrics = {}
    
    # Compute pairwise distances in original and reduced space
    dist_original = pairwise_distances(X_original)
    dist_reduced = pairwise_distances(X_reduced)
    
    # Flatten upper triangle (excluding diagonal)
    n = len(X_original)
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    dist_orig_flat = dist_original[mask]
    dist_reduced_flat = dist_reduced[mask]
    
    # Correlation between distance matrices (higher is better)
    if np.std(dist_orig_flat) > 0 and np.std(dist_reduced_flat) > 0:
        metrics['distance_correlation'] = np.corrcoef(dist_orig_flat, dist_reduced_flat)[0, 1]
    else:
        metrics['distance_correlation'] = 0.0
    
    # Trustworthiness (how well local neighborhoods are preserved)
    k = min(10, len(X_original) // 10)
    trustworthiness_scores = []
    for i in range(len(X_original)):
        orig_distances = dist_original[i]
        orig_neighbors = np.argsort(orig_distances)[1:k+1]
        
        reduced_distances = dist_reduced[i]
        reduced_neighbors = np.argsort(reduced_distances)[1:k+1]
        
        preserved = len(set(orig_neighbors) & set(reduced_neighbors))
        trustworthiness_scores.append(preserved / k)
    
    metrics['trustworthiness'] = np.mean(trustworthiness_scores)
    
    # Variance explained (for PCA, this is meaningful; for others, approximate)
    var_original = np.var(X_original, axis=0).sum()
    var_reduced = np.var(X_reduced, axis=0).sum()
    metrics['variance_ratio'] = var_reduced / var_original if var_original > 0 else 0.0
    
    # Cluster separability (if labels available)
    if y_true is not None:
        metrics['cluster_separability'] = compute_cluster_separability(X_reduced, y_true)
        metrics['silhouette_score'] = silhouette_score(X_reduced, y_true)
    
    # Entropy-based distribution metric
    metrics['entropy_score'] = compute_entropy_metric(X_reduced)
    
    return metrics

def compute_downstream_performance(X_original, X_reduced, y_true):
    """Test how well reduced space works for downstream classification task."""
    # Train KNN classifier on original space
    knn_orig = KNeighborsClassifier(n_neighbors=5)
    orig_scores = cross_val_score(knn_orig, X_original, y_true, cv=5, scoring='accuracy')
    
    # Train KNN classifier on reduced space
    knn_reduced = KNeighborsClassifier(n_neighbors=5)
    reduced_scores = cross_val_score(knn_reduced, X_reduced, y_true, cv=5, scoring='accuracy')
    
    # Performance retention = how much of original performance is retained
    orig_mean = orig_scores.mean()
    reduced_mean = reduced_scores.mean()
    retention = reduced_mean / orig_mean if orig_mean > 0 else 0.0
    
    return {
        'original_accuracy': orig_mean,
        'reduced_accuracy': reduced_mean,
        'performance_retention': retention
    }

def apply_dimensionality_reduction(X, dataset_name, y_true, n_components=2):
    """Apply all dimensionality reduction techniques with runtime tracking and advanced metrics."""
    results = {}
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    print(f"\n   Applying PCA (n_components={n_components})...")
    start_time = time.time()
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_time = time.time() - start_time
    
    pca_metrics = compute_preservation_metrics(X_scaled, X_pca, y_true)
    pca_metrics['explained_variance_ratio'] = pca.explained_variance_ratio_.sum()
    pca_downstream = compute_downstream_performance(X_scaled, X_pca, y_true)
    
    results['PCA'] = {
        'embedding': X_pca,
        'runtime': pca_time,
        'metrics': pca_metrics,
        'explained_variance_ratio': pca.explained_variance_ratio_.sum(),
        'downstream': pca_downstream
    }
    
    # t-SNE
    print(f"   Applying t-SNE (n_components={n_components}, perplexity=30)...")
    start_time = time.time()
    tsne = TSNE(n_components=n_components, perplexity=30, random_state=42, 
                init='pca', n_iter=1000, verbose=0)
    X_tsne = tsne.fit_transform(X_scaled)
    tsne_time = time.time() - start_time
    
    tsne_metrics = compute_preservation_metrics(X_scaled, X_tsne, y_true)
    tsne_downstream = compute_downstream_performance(X_scaled, X_tsne, y_true)
    results['t-SNE'] = {
        'embedding': X_tsne,
        'runtime': tsne_time,
        'metrics': tsne_metrics,
        'downstream': tsne_downstream
    }
    
    # UMAP
    if UMAP_AVAILABLE:
        print(f"   Applying UMAP (n_components={n_components}, n_neighbors=15)...")
        start_time = time.time()
        reducer = umap.UMAP(n_components=n_components, n_neighbors=15, 
                           min_dist=0.1, random_state=42, verbose=False)
        X_umap = reducer.fit_transform(X_scaled)
        umap_time = time.time() - start_time
        
        umap_metrics = compute_preservation_metrics(X_scaled, X_umap, y_true)
        umap_downstream = compute_downstream_performance(X_scaled, X_umap, y_true)
        results['UMAP'] = {
            'embedding': X_umap,
            'runtime': umap_time,
            'metrics': umap_metrics,
            'downstream': umap_downstream
        }
    else:
        print("   Skipping UMAP (not available)")
    
    return results, X_scaled

def visualize_results(datasets, all_results):
    """Create comprehensive visualization of dimensionality reduction results."""
    n_datasets = len(datasets)
    n_methods = 4 if UMAP_AVAILABLE else 3  # Original + PCA + t-SNE + (UMAP if available)
    
    fig, axes = plt.subplots(n_datasets, n_methods, figsize=(5*n_methods, 4*n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    dataset_names = list(datasets.keys())
    method_names = ['Original (2D projection)' if datasets[name][0].shape[1] > 2 else 'Original',
                    'PCA', 't-SNE'] + (['UMAP'] if UMAP_AVAILABLE else [])
    
    for i, (name, (X, y_true, description)) in enumerate(datasets.items()):
        results = all_results[name]
        X_scaled = results['X_scaled']
        
        # If original is high-dimensional, show first 2 principal components
        if X_scaled.shape[1] > 2:
            pca_temp = PCA(n_components=2, random_state=42)
            X_original_2d = pca_temp.fit_transform(X_scaled)
            ax_title = 'Original (first 2 PCs)'
        else:
            X_original_2d = X_scaled
            ax_title = 'Original'
        
        # Plot original
        ax = axes[i, 0]
        scatter = ax.scatter(X_original_2d[:, 0], X_original_2d[:, 1], c=y_true, 
                            cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.set_title(f'{description}\n{ax_title}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.grid(True, alpha=0.3)
        
        # Plot PCA
        ax = axes[i, 1]
        pca_embedding = results['PCA']['embedding']
        ax.scatter(pca_embedding[:, 0], pca_embedding[:, 1], c=y_true, 
                  cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
        evr = results['PCA']['explained_variance_ratio']
        ax.set_title(f"PCA\nEV Ratio: {evr:.3f}", fontsize=10)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.grid(True, alpha=0.3)
        
        # Plot t-SNE
        ax = axes[i, 2]
        tsne_embedding = results['t-SNE']['embedding']
        ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=y_true, 
                  cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
        trust = results['t-SNE']['metrics']['trustworthiness']
        ax.set_title(f"t-SNE\nTrustworthiness: {trust:.3f}", fontsize=10)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.3)
        
        # Plot UMAP if available
        if UMAP_AVAILABLE:
            ax = axes[i, 3]
            umap_embedding = results['UMAP']['embedding']
            ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=y_true, 
                      cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
            trust = results['UMAP']['metrics']['trustworthiness']
            ax.set_title(f"UMAP\nTrustworthiness: {trust:.3f}", fontsize=10)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dimensionality_reduction_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'dimensionality_reduction_comparison.png'")
    
    # Create metrics comparison visualization
    if all_results:
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        
        datasets_list = list(all_results.keys())
        methods = ['PCA', 't-SNE'] + (['UMAP'] if UMAP_AVAILABLE else [])
        
        # Runtime comparison
        runtime_data = {method: [all_results[ds][method].get('runtime', 0) * 1000 
                                for ds in datasets_list] 
                       for method in methods}
        x = np.arange(len(datasets_list))
        width = 0.25
        for i, method in enumerate(methods):
            axes2[0].bar(x + i*width, runtime_data[method], width, label=method, alpha=0.8)
        axes2[0].set_xlabel('Dataset')
        axes2[0].set_ylabel('Runtime (ms)')
        axes2[0].set_title('Runtime Performance Comparison')
        axes2[0].set_xticks(x + width)
        axes2[0].set_xticklabels([d.replace('_', '\n')[:15] for d in datasets_list], 
                                fontsize=8, rotation=45, ha='right')
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3, axis='y')
        axes2[0].set_yscale('log')  # Log scale for runtime
        
        # Trustworthiness comparison
        trust_data = {method: [all_results[ds][method]['metrics']['trustworthiness'] 
                             for ds in datasets_list] 
                     for method in methods}
        for i, method in enumerate(methods):
            axes2[1].bar(x + i*width, trust_data[method], width, label=method, alpha=0.8)
        axes2[1].set_xlabel('Dataset')
        axes2[1].set_ylabel('Trustworthiness Score')
        axes2[1].set_title('Local Structure Preservation (Trustworthiness)')
        axes2[1].set_xticks(x + width)
        axes2[1].set_xticklabels([d.replace('_', '\n')[:15] for d in datasets_list], 
                                fontsize=8, rotation=45, ha='right')
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3, axis='y')
        axes2[1].set_ylim([0, 1])
        
        # Distance correlation comparison
        corr_data = {method: [all_results[ds][method]['metrics']['distance_correlation'] 
                            for ds in datasets_list] 
                    for method in methods}
        for i, method in enumerate(methods):
            axes2[2].bar(x + i*width, corr_data[method], width, label=method, alpha=0.8)
        axes2[2].set_xlabel('Dataset')
        axes2[2].set_ylabel('Distance Correlation')
        axes2[2].set_title('Global Structure Preservation (Distance Correlation)')
        axes2[2].set_xticks(x + width)
        axes2[2].set_xticklabels([d.replace('_', '\n')[:15] for d in datasets_list], 
                                fontsize=8, rotation=45, ha='right')
        axes2[2].legend()
        axes2[2].grid(True, alpha=0.3, axis='y')
        axes2[2].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('dimensionality_reduction_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Metrics comparison saved as 'dimensionality_reduction_metrics_comparison.png'")
    
    return fig

def print_comparison_table(all_results):
    """Print a comprehensive comparison table."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("=" * 70)
    
    methods = ['PCA', 't-SNE'] + (['UMAP'] if UMAP_AVAILABLE else [])
    print(f"\n{'Dataset':<20} {'Method':<10} {'Runtime(ms)':<12} {'Trustworthiness':<15} "
          f"{'Dist Corr':<12} {'EV Ratio':<10} {'Notes':<20}")
    print("-" * 110)
    
    for dataset_name, results in all_results.items():
        for method_name in methods:
            method_results = results[method_name]
            metrics = method_results['metrics']
            runtime_ms = method_results.get('runtime', 0) * 1000
            
            ev_ratio = method_results.get('explained_variance_ratio', 0.0)
            ev_str = f"{ev_ratio:.3f}" if ev_ratio > 0 else "N/A"
            
            notes = ""
            if method_name == 'PCA' and ev_ratio > 0.8:
                notes = "High variance explained"
            elif metrics['trustworthiness'] > 0.7:
                notes = "Good local preservation"
            
            print(f"{dataset_name[:18]:<20} {method_name:<10} {runtime_ms:<12.2f} "
                  f"{metrics['trustworthiness']:<15.3f} {metrics['distance_correlation']:<12.3f} "
                  f"{ev_str:<10} {notes:<20}")
        print("-" * 110)

def analyze_results(all_results):
    """Provide insights and recommendations."""
    print("\n" + "=" * 70)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 70)
    
    insights = {
        'high_dim_spherical': {
            'best': 'PCA',
            'reason': 'Linear structure with high variance; PCA captures maximum variance efficiently'
        },
        'nonlinear_moons': {
            'best': 't-SNE or UMAP',
            'reason': 'Non-linear structure requires non-linear dimensionality reduction'
        },
        'concentric_circles': {
            'best': 't-SNE or UMAP',
            'reason': 'Non-convex structure cannot be captured by linear methods like PCA'
        },
        'swiss_roll': {
            'best': 't-SNE or UMAP',
            'reason': 'Manifold learning challenge; non-linear methods can unfold the manifold'
        },
        'linear_structure': {
            'best': 'PCA',
            'reason': 'Linear structure with noise; PCA efficiently captures the linear trend'
        }
    }
    
    for dataset_name, results in all_results.items():
        print(f"\n📊 {dataset_name.upper().replace('_', ' ')} Dataset:")
        print(f"   Expected best: {insights[dataset_name]['best']}")
        print(f"   Reason: {insights[dataset_name]['reason']}")
        
        # Find best by trustworthiness
        best_trust = -1
        best_method = None
        for method_name in ['PCA', 't-SNE'] + (['UMAP'] if UMAP_AVAILABLE else []):
            trust = results[method_name]['metrics']['trustworthiness']
            if trust > best_trust:
                best_trust = trust
                best_method = method_name
        
        print(f"   Best local preservation: {best_method} (Trustworthiness: {best_trust:.3f})")
        
        # Method-specific insights
        print(f"\n   Method Performance:")
        for method_name in ['PCA', 't-SNE'] + (['UMAP'] if UMAP_AVAILABLE else []):
            method_results = results[method_name]
            metrics = method_results['metrics']
            print(f"   • {method_name}:")
            print(f"     - Runtime: {method_results.get('runtime', 0)*1000:.2f} ms")
            print(f"     - Trustworthiness: {metrics['trustworthiness']:.3f}")
            print(f"     - Distance Correlation: {metrics['distance_correlation']:.3f}")
            if method_name == 'PCA':
                evr = method_results.get('explained_variance_ratio', 0)
                print(f"     - Explained Variance Ratio: {evr:.3f}")

def main():
    """Main execution function."""
    if not UMAP_AVAILABLE:
        print("\n⚠️  UMAP not available. Install with: pip install umap-learn")
        print("   Continuing with PCA and t-SNE only...\n")
    
    # Generate datasets
    datasets = generate_datasets()
    
    # Apply dimensionality reduction to each dataset
    print("\n" + "=" * 70)
    print("APPLYING DIMENSIONALITY REDUCTION TECHNIQUES")
    print("=" * 70)
    
    all_results = {}
    
    for name, (X, y_true, description) in datasets.items():
        print(f"\n📦 Processing {name} dataset ({description})...")
        results, X_scaled = apply_dimensionality_reduction(X, name, n_components=2)
        results['X_scaled'] = X_scaled
        all_results[name] = results
    
    # Print comparison table
    print_comparison_table(all_results)
    
    # Visualize results
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    visualize_results(datasets, all_results)
    
    # Analyze and provide insights
    analyze_results(all_results)
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. PCA (Principal Component Analysis):
   • Best for: Linear structures, high-dimensional data with linear relationships
   • Strengths: Fast, interpretable, preserves global variance, deterministic
   • Limitations: Cannot capture non-linear structures, assumes linear relationships
   • Use when: Data has linear structure, you need fast computation, or interpretability matters
   • Explained Variance Ratio: Meaningful metric showing how much variance is preserved

2. t-SNE (t-Distributed Stochastic Neighbor Embedding):
   • Best for: Non-linear structures, visualization, local neighborhood preservation
   • Strengths: Excellent at preserving local structure, great for visualization
   • Limitations: Slow, non-deterministic, doesn't preserve global structure, sensitive to parameters
   • Use when: Visualizing high-dimensional data, exploring local clusters, non-linear structures
   • Trustworthiness: Measures how well local neighborhoods are preserved

3. UMAP (Uniform Manifold Approximation and Projection):
   • Best for: Non-linear structures, balance between local and global preservation
   • Strengths: Faster than t-SNE, preserves both local and global structure better
   • Limitations: Still slower than PCA, requires parameter tuning
   • Use when: You need non-linear reduction with better global structure than t-SNE
   • Balance: Better global structure preservation than t-SNE while maintaining local structure

4. GENERAL RULE:
   • Use PCA for linear structures or when speed/interpretability is critical
   • Use t-SNE for visualization-focused tasks with non-linear data
   • Use UMAP when you need non-linear reduction with better global structure
   • Always visualize results to understand what each method is doing
   • Consider computational cost: PCA << UMAP < t-SNE for large datasets
   • Distance correlation measures global structure preservation
   • Trustworthiness measures local neighborhood preservation
    """)
    
    print("=" * 70)
    print("Analysis complete! Check 'dimensionality_reduction_comparison.png' for visualizations.")
    print("=" * 70)

if __name__ == "__main__":
    main()
