"""
Clustering Algorithms Comparison Study: Parameter Sensitivity & Robustness Analysis
A comprehensive study exploring how different clustering algorithms perform across various data 
distributions, with emphasis on parameter sensitivity, runtime performance, and robustness to noise.
Compares K-means, DBSCAN, and Agglomerative clustering with systematic parameter sweeps.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("Clustering Algorithms Comparison Study")
print("=" * 70)

def generate_datasets():
    """Generate three different synthetic datasets with distinct characteristics."""
    datasets = {}
    
    # Dataset 1: Well-separated spherical clusters (K-means should excel)
    print("\n1. Generating Dataset 1: Well-separated spherical clusters...")
    X1, y1 = make_blobs(
        n_samples=300,
        centers=4,
        n_features=2,
        cluster_std=0.8,
        random_state=42
    )
    datasets['spherical'] = (X1, y1)
    print(f"   Shape: {X1.shape}, True clusters: {len(np.unique(y1))}")
    
    # Dataset 2: Non-spherical clusters (moons - DBSCAN should excel)
    print("\n2. Generating Dataset 2: Non-spherical clusters (moons)...")
    X2, y2 = make_moons(
        n_samples=300,
        noise=0.1,
        random_state=42
    )
    datasets['moons'] = (X2, y2)
    print(f"   Shape: {X2.shape}, True clusters: {len(np.unique(y2))}")
    
    # Dataset 3: Concentric circles (DBSCAN should excel)
    print("\n3. Generating Dataset 3: Concentric circles...")
    X3, y3 = make_circles(
        n_samples=300,
        noise=0.1,
        factor=0.5,
        random_state=42
    )
    datasets['circles'] = (X3, y3)
    print(f"   Shape: {X3.shape}, True clusters: {len(np.unique(y3))}")
    
    # Dataset 4: Varying density clusters (challenging for all)
    print("\n4. Generating Dataset 4: Varying density clusters...")
    X4_1, y4_1 = make_blobs(n_samples=100, centers=[[-2, -2]], cluster_std=0.3, random_state=42)
    X4_2, y4_2 = make_blobs(n_samples=200, centers=[[2, 2]], cluster_std=1.2, random_state=43)
    X4 = np.vstack([X4_1, X4_2])
    y4 = np.hstack([np.zeros(100), np.ones(200)])
    datasets['varying_density'] = (X4, y4)
    print(f"   Shape: {X4.shape}, True clusters: {len(np.unique(y4))}")
    
    # Dataset 5: Anisotropic clusters (elongated, not spherical) - unique challenge
    print("\n5. Generating Dataset 5: Anisotropic (elongated) clusters...")
    X5, y5 = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=1.0, random_state=42)
    # Apply transformation to create elongated clusters
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X5 = np.dot(X5, transformation)
    datasets['anisotropic'] = (X5, y5)
    print(f"   Shape: {X5.shape}, True clusters: {len(np.unique(y5))}")
    
    return datasets

def apply_clustering(X, dataset_name, true_labels):
    """Apply all three clustering algorithms with runtime tracking and return results."""
    results = {}
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_clusters = len(np.unique(true_labels))
    
    # K-means clustering with runtime
    print(f"\n   Applying K-means (k={n_clusters})...")
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_time = time.time() - start_time
    results['K-means'] = {
        'labels': kmeans_labels,
        'n_clusters': len(np.unique(kmeans_labels)),
        'silhouette': silhouette_score(X_scaled, kmeans_labels),
        'ari': adjusted_rand_score(true_labels, kmeans_labels),
        'davies_bouldin': davies_bouldin_score(X_scaled, kmeans_labels),
        'runtime': kmeans_time
    }
    
    # DBSCAN clustering with runtime
    print(f"   Applying DBSCAN (eps=0.3, min_samples=5)...")
    start_time = time.time()
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    dbscan_time = time.time() - start_time
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    results['DBSCAN'] = {
        'labels': dbscan_labels,
        'n_clusters': n_clusters_dbscan,
        'n_noise': list(dbscan_labels).count(-1),
        'silhouette': silhouette_score(X_scaled, dbscan_labels) if n_clusters_dbscan > 1 else -1,
        'ari': adjusted_rand_score(true_labels, dbscan_labels) if n_clusters_dbscan > 0 else 0,
        'davies_bouldin': davies_bouldin_score(X_scaled, dbscan_labels) if n_clusters_dbscan > 1 else float('inf'),
        'runtime': dbscan_time
    }
    
    # Agglomerative clustering with runtime
    print(f"   Applying Agglomerative Clustering (n_clusters={n_clusters})...")
    start_time = time.time()
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    agg_labels = agg.fit_predict(X_scaled)
    agg_time = time.time() - start_time
    results['Agglomerative'] = {
        'labels': agg_labels,
        'n_clusters': len(np.unique(agg_labels)),
        'silhouette': silhouette_score(X_scaled, agg_labels),
        'ari': adjusted_rand_score(true_labels, agg_labels),
        'davies_bouldin': davies_bouldin_score(X_scaled, agg_labels),
        'runtime': agg_time
    }
    
    return results, X_scaled

def parameter_sensitivity_analysis(X_scaled, true_labels, dataset_name):
    """Perform parameter sensitivity analysis for DBSCAN."""
    print(f"\n   🔬 Running parameter sensitivity analysis for DBSCAN on {dataset_name}...")
    
    eps_values = np.linspace(0.1, 0.5, 9)
    min_samples_values = [3, 5, 7, 10]
    
    best_ari = -1
    best_params = None
    sensitivity_results = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters > 1:
                ari = adjusted_rand_score(true_labels, labels)
                n_noise = list(labels).count(-1)
                sensitivity_results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'ari': ari,
                    'n_noise': n_noise
                })
                
                if ari > best_ari:
                    best_ari = ari
                    best_params = (eps, min_samples, n_clusters, n_noise)
    
    if best_params:
        print(f"      Best DBSCAN params: eps={best_params[0]:.2f}, min_samples={best_params[1]}, "
              f"clusters={best_params[2]}, noise={best_params[3]}, ARI={best_ari:.3f}")
    
    return sensitivity_results, best_params

def visualize_results(datasets, all_results, sensitivity_data=None):
    """Create comprehensive visualization of clustering results."""
    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets, 4, figsize=(16, 4 * n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    dataset_names = list(datasets.keys())
    algorithm_names = ['True Labels', 'K-means', 'DBSCAN', 'Agglomerative']
    
    for i, (name, (X, y_true)) in enumerate(datasets.items()):
        results = all_results[name]
        X_scaled = results['X_scaled']
        
        # Plot true labels
        ax = axes[i, 0]
        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.6)
        ax.set_title(f'{name.capitalize()}\nTrue Labels', fontsize=10, fontweight='bold')
        ax.set_xlabel('Feature 1 (scaled)')
        ax.set_ylabel('Feature 2 (scaled)')
        ax.grid(True, alpha=0.3)
        
        # Plot K-means
        ax = axes[i, 1]
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=results['K-means']['labels'], 
                  cmap='viridis', s=50, alpha=0.6)
        ax.set_title(f"K-means\nARI: {results['K-means']['ari']:.3f}, Sil: {results['K-means']['silhouette']:.3f}", 
                    fontsize=10)
        ax.set_xlabel('Feature 1 (scaled)')
        ax.set_ylabel('Feature 2 (scaled)')
        ax.grid(True, alpha=0.3)
        
        # Plot DBSCAN
        ax = axes[i, 2]
        dbscan_labels = results['DBSCAN']['labels']
        # Color noise points differently
        noise_mask = dbscan_labels == -1
        if np.any(noise_mask):
            ax.scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1], 
                      c='red', marker='x', s=100, alpha=0.8, label='Noise')
        ax.scatter(X_scaled[~noise_mask, 0], X_scaled[~noise_mask, 1], 
                  c=dbscan_labels[~noise_mask], cmap='viridis', s=50, alpha=0.6)
        ax.set_title(f"DBSCAN\nARI: {results['DBSCAN']['ari']:.3f}, Sil: {results['DBSCAN']['silhouette']:.3f}\nNoise: {results['DBSCAN']['n_noise']}", 
                    fontsize=10)
        ax.set_xlabel('Feature 1 (scaled)')
        ax.set_ylabel('Feature 2 (scaled)')
        ax.grid(True, alpha=0.3)
        if np.any(noise_mask):
            ax.legend()
        
        # Plot Agglomerative
        ax = axes[i, 3]
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=results['Agglomerative']['labels'], 
                  cmap='viridis', s=50, alpha=0.6)
        ax.set_title(f"Agglomerative\nARI: {results['Agglomerative']['ari']:.3f}, Sil: {results['Agglomerative']['silhouette']:.3f}", 
                    fontsize=10)
        ax.set_xlabel('Feature 1 (scaled)')
        ax.set_ylabel('Feature 2 (scaled)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'clustering_comparison.png'")
    
    # Create additional visualization for runtime and metric comparison
    if all_results:
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        
        datasets_list = list(all_results.keys())
        algorithms = ['K-means', 'DBSCAN', 'Agglomerative']
        
        # Runtime comparison
        runtime_data = {algo: [all_results[ds][algo].get('runtime', 0) * 1000 for ds in datasets_list] 
                       for algo in algorithms}
        x = np.arange(len(datasets_list))
        width = 0.25
        for i, algo in enumerate(algorithms):
            axes2[0].bar(x + i*width, runtime_data[algo], width, label=algo, alpha=0.8)
        axes2[0].set_xlabel('Dataset')
        axes2[0].set_ylabel('Runtime (ms)')
        axes2[0].set_title('Runtime Performance Comparison')
        axes2[0].set_xticks(x + width)
        axes2[0].set_xticklabels([d.replace('_', '\n') for d in datasets_list], fontsize=8)
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3, axis='y')
        
        # ARI comparison
        ari_data = {algo: [all_results[ds][algo]['ari'] for ds in datasets_list] 
                   for algo in algorithms}
        for i, algo in enumerate(algorithms):
            axes2[1].bar(x + i*width, ari_data[algo], width, label=algo, alpha=0.8)
        axes2[1].set_xlabel('Dataset')
        axes2[1].set_ylabel('Adjusted Rand Index')
        axes2[1].set_title('ARI Performance Comparison')
        axes2[1].set_xticks(x + width)
        axes2[1].set_xticklabels([d.replace('_', '\n') for d in datasets_list], fontsize=8)
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3, axis='y')
        axes2[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        # Silhouette comparison
        sil_data = {algo: [all_results[ds][algo]['silhouette'] for ds in datasets_list] 
                   for algo in algorithms}
        for i, algo in enumerate(algorithms):
            axes2[2].bar(x + i*width, sil_data[algo], width, label=algo, alpha=0.8)
        axes2[2].set_xlabel('Dataset')
        axes2[2].set_ylabel('Silhouette Score')
        axes2[2].set_title('Silhouette Score Comparison')
        axes2[2].set_xticks(x + width)
        axes2[2].set_xticklabels([d.replace('_', '\n') for d in datasets_list], fontsize=8)
        axes2[2].legend()
        axes2[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('clustering_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Metrics comparison saved as 'clustering_metrics_comparison.png'")
    
    return fig

def print_comparison_table(all_results):
    """Print a comprehensive comparison table with runtime and multiple metrics."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("=" * 70)
    
    # Header with additional metrics
    print(f"\n{'Dataset':<18} {'Algorithm':<15} {'Clusters':<10} {'ARI':<8} {'Silhouette':<12} {'DB Score':<10} {'Runtime(ms)':<12} {'Notes':<15}")
    print("-" * 110)
    
    for dataset_name, results in all_results.items():
        for algo_name in ['K-means', 'DBSCAN', 'Agglomerative']:
            algo_results = results[algo_name]
            notes = ""
            if algo_name == 'DBSCAN' and algo_results.get('n_noise', 0) > 0:
                notes = f"Noise: {algo_results['n_noise']}"
            
            db_score = algo_results.get('davies_bouldin', float('inf'))
            db_str = f"{db_score:.3f}" if db_score != float('inf') else "N/A"
            runtime_ms = algo_results.get('runtime', 0) * 1000
            
            print(f"{dataset_name:<18} {algo_name:<15} {algo_results['n_clusters']:<10} "
                  f"{algo_results['ari']:<8.3f} {algo_results['silhouette']:<12.3f} "
                  f"{db_str:<10} {runtime_ms:<12.2f} {notes:<15}")
        print("-" * 110)

def analyze_results(all_results):
    """Provide insights and recommendations."""
    print("\n" + "=" * 70)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 70)
    
    insights = {
        'spherical': {
            'best': 'K-means',
            'reason': 'Spherical clusters are ideal for K-means which assumes spherical cluster shapes'
        },
        'moons': {
            'best': 'DBSCAN',
            'reason': 'Non-spherical, non-convex clusters require density-based methods like DBSCAN'
        },
        'circles': {
            'best': 'DBSCAN',
            'reason': 'Concentric structures cannot be separated by distance-based methods; DBSCAN excels'
        },
        'varying_density': {
            'best': 'DBSCAN or Agglomerative',
            'reason': 'Varying densities challenge all algorithms; DBSCAN can handle it with proper tuning'
        },
        'anisotropic': {
            'best': 'Agglomerative or DBSCAN',
            'reason': 'Elongated clusters violate K-means spherical assumption; hierarchical methods can adapt better'
        }
    }
    
    for dataset_name, results in all_results.items():
        print(f"\n📊 {dataset_name.upper().replace('_', ' ')} Dataset:")
        print(f"   Expected best: {insights[dataset_name]['best']}")
        print(f"   Reason: {insights[dataset_name]['reason']}")
        
        # Find actual best by ARI
        best_ari = -1
        best_algo = None
        for algo_name in ['K-means', 'DBSCAN', 'Agglomerative']:
            ari = results[algo_name]['ari']
            if ari > best_ari:
                best_ari = ari
                best_algo = algo_name
        
        print(f"   Actual best (by ARI): {best_algo} (ARI: {best_ari:.3f})")
        
        # Algorithm-specific insights
        print(f"\n   Algorithm Performance:")
        for algo_name in ['K-means', 'DBSCAN', 'Agglomerative']:
            algo_results = results[algo_name]
            print(f"   • {algo_name}:")
            print(f"     - Clusters found: {algo_results['n_clusters']}")
            print(f"     - ARI: {algo_results['ari']:.3f}")
            print(f"     - Silhouette: {algo_results['silhouette']:.3f}")
            db_val = algo_results.get('davies_bouldin', float('inf'))
            if db_val != float('inf'):
                print(f"     - Davies-Bouldin: {db_val:.3f}")
            else:
                print("     - Davies-Bouldin: N/A")
            print(f"     - Runtime: {algo_results.get('runtime', 0)*1000:.2f} ms")
            if algo_name == 'DBSCAN' and algo_results.get('n_noise', 0) > 0:
                print(f"     - Noise points: {algo_results['n_noise']}")

def main():
    """Main execution function."""
    # Generate datasets
    datasets = generate_datasets()
    
    # Apply clustering to each dataset
    print("\n" + "=" * 70)
    print("APPLYING CLUSTERING ALGORITHMS")
    print("=" * 70)
    
    all_results = {}
    sensitivity_analysis = {}
    
    for name, (X, y_true) in datasets.items():
        print(f"\n📦 Processing {name} dataset...")
        results, X_scaled = apply_clustering(X, name, y_true)
        results['X_scaled'] = X_scaled
        all_results[name] = results
        
        # Perform parameter sensitivity analysis for DBSCAN
        if name in ['moons', 'circles', 'varying_density']:  # Most interesting cases
            sens_results, best_params = parameter_sensitivity_analysis(X_scaled, y_true, name)
            sensitivity_analysis[name] = {'results': sens_results, 'best': best_params}
    
    # Print comparison table
    print_comparison_table(all_results)
    
    # Visualize results
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    visualize_results(datasets, all_results, sensitivity_analysis)
    
    # Print parameter sensitivity insights
    if sensitivity_analysis:
        print("\n" + "=" * 70)
        print("PARAMETER SENSITIVITY INSIGHTS")
        print("=" * 70)
        for dataset_name, sens_data in sensitivity_analysis.items():
            if sens_data['best']:
                print(f"\n📊 {dataset_name.upper().replace('_', ' ')} Dataset:")
                print(f"   DBSCAN is highly sensitive to parameters. Best found:")
                print(f"   eps={sens_data['best'][0]:.2f}, min_samples={sens_data['best'][1]}")
                print(f"   This demonstrates the importance of parameter tuning for DBSCAN.")
    
    # Analyze and provide insights
    analyze_results(all_results)
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. K-MEANS:
   • Best for: Spherical, well-separated clusters
   • Assumes: Equal cluster sizes, spherical shapes
   • Limitations: Cannot handle non-convex or varying density clusters
   • Use when: You know the number of clusters and data is spherical

2. DBSCAN:
   • Best for: Non-spherical, varying density, noise detection
   • Assumes: Clusters are dense regions separated by sparse areas
   • Limitations: Sensitive to eps and min_samples parameters
   • Use when: Unknown number of clusters, non-spherical shapes, or noise present

3. AGGLOMERATIVE CLUSTERING:
   • Best for: Hierarchical structure, when you need cluster hierarchy
   • Assumes: Can work with various shapes but prefers compact clusters
   • Limitations: Computationally expensive for large datasets
   • Use when: You need hierarchical relationships or have small-medium datasets

4. GENERAL RULE:
   • No single algorithm works best for all data types
   • Always visualize your data first to understand its structure
   • Use multiple algorithms and compare results
   • Consider your domain knowledge when choosing algorithms
    """)
    
    print("=" * 70)
    print("Analysis complete! Check 'clustering_comparison.png' for visualizations.")
    print("=" * 70)

if __name__ == "__main__":
    main()
