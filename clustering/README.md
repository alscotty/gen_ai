# Clustering Algorithms Comparison

This directory contains a comprehensive comparison study of three fundamental clustering algorithms: K-means, DBSCAN, and Agglomerative Clustering. Unlike standard tutorials that demonstrate a single algorithm, this study systematically compares how different algorithms perform on various data distributions to understand their strengths, limitations, and when to use each.

## Files

- **clustering_comparison.py**: Comprehensive Clustering Algorithm Comparison
  - Compares K-means, DBSCAN, and Agglomerative Clustering on four different synthetic datasets
  - Tests algorithms on: spherical clusters, non-spherical clusters (moons), concentric circles, and varying density clusters
  - Evaluates performance using Adjusted Rand Index (ARI) and Silhouette Score
  - Generates detailed visualizations showing how each algorithm handles different cluster shapes
  - Provides actionable insights on when to use each algorithm

## Key Concepts Demonstrated

- **K-means Clustering**: Centroid-based clustering, assumes spherical clusters
- **DBSCAN**: Density-based clustering, handles non-spherical shapes and noise
- **Agglomerative Clustering**: Hierarchical clustering using linkage methods
- **Cluster Evaluation Metrics**: Adjusted Rand Index (ARI), Silhouette Score, and Davies-Bouldin Index for comprehensive evaluation
- **Runtime Performance Analysis**: Comparative timing analysis across algorithms
- **Parameter Sensitivity Analysis**: Systematic parameter sweep for DBSCAN to understand tuning requirements
- **Data Preprocessing**: Feature standardization for clustering algorithms
- **Synthetic Data Generation**: Creating controlled datasets with specific characteristics (spherical, non-spherical, concentric, varying density, anisotropic)
- **Comparative Analysis**: Systematic methodology for choosing the right clustering algorithm

## Datasets Used

1. **Spherical Clusters**: Well-separated, equal-sized spherical clusters (ideal for K-means)
2. **Moons Dataset**: Two crescent-shaped clusters (challenges K-means, ideal for DBSCAN)
3. **Concentric Circles**: Two nested circular clusters (impossible for K-means, ideal for DBSCAN)
4. **Varying Density**: Clusters with different densities and sizes (challenges all algorithms)
5. **Anisotropic Clusters**: Elongated, non-spherical clusters created via transformation (tests algorithm robustness)

## Setup

1. Create a virtual environment:
```bash
python3 -m venv path/to/venv
source path/to/venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the comparison:
```bash
python clustering_comparison.py
```

## Output

The script will:
- Display detailed comparison tables with multiple metrics (ARI, Silhouette, Davies-Bouldin, Runtime) for each algorithm
- Save comprehensive visualization plots:
  - `clustering_comparison.png`: Side-by-side comparison of all algorithms on all datasets
  - `clustering_metrics_comparison.png`: Bar charts comparing runtime, ARI, and Silhouette scores across datasets
- Perform parameter sensitivity analysis for DBSCAN on challenging datasets
- Provide analysis and recommendations on when to use each algorithm
- Show noise detection capabilities of DBSCAN
- Highlight the limitations and strengths of each approach
- Demonstrate runtime performance differences between algorithms

## Algorithm Characteristics

### K-means
- **Best for**: Spherical, well-separated clusters with similar sizes
- **Assumptions**: Clusters are spherical and have similar variance
- **Limitations**: Cannot handle non-convex shapes, varying densities, or unknown number of clusters
- **Use when**: You know the number of clusters and data has spherical structure

### DBSCAN
- **Best for**: Non-spherical clusters, varying densities, noise detection
- **Assumptions**: Clusters are dense regions separated by sparse areas
- **Limitations**: Sensitive to `eps` and `min_samples` parameters, struggles with varying densities
- **Use when**: Unknown number of clusters, non-spherical shapes, or when noise is present

### Agglomerative Clustering
- **Best for**: Hierarchical structure, when you need cluster relationships
- **Assumptions**: Can work with various shapes but prefers compact clusters
- **Limitations**: Computationally expensive for large datasets, requires number of clusters
- **Use when**: You need hierarchical relationships or have small-medium datasets

## Unique Features

Unlike standard clustering tutorials that show a single algorithm, this script:
- **Systematically compares multiple algorithms** to understand trade-offs
- **Uses diverse synthetic datasets** including anisotropic (elongated) clusters not commonly tested
- **Performs parameter sensitivity analysis** - systematically tests DBSCAN across parameter ranges to demonstrate tuning importance
- **Includes runtime performance analysis** - compares computational efficiency across algorithms
- **Uses multiple evaluation metrics** - ARI, Silhouette Score, and Davies-Bouldin Index for comprehensive assessment
- **Focuses on decision-making** - when to use which algorithm based on data characteristics
- **Demonstrates real-world challenges** like non-spherical clusters, varying densities, and noise
- **Generates comparative visualizations** - both cluster assignments and metric comparisons
- **Provides actionable insights** based on quantitative comparisons and parameter sensitivity

## Key Insights

1. **No one-size-fits-all**: Different algorithms excel at different cluster shapes
2. **Data structure matters**: Understanding your data's structure is crucial for algorithm selection
3. **Parameter sensitivity**: DBSCAN requires careful tuning of `eps` and `min_samples`
4. **Evaluation is critical**: Use multiple metrics (ARI, Silhouette) to assess cluster quality
5. **Visualization helps**: Always visualize your data and clustering results to understand what's happening
