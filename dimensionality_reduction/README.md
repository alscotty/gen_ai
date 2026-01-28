# Dimensionality Reduction Advanced Analysis Study

This directory contains a **unique** comprehensive comparison study of three fundamental dimensionality reduction techniques: PCA, t-SNE, and UMAP. Unlike standard online tutorials that show basic comparisons, this study includes **advanced analyses not commonly found elsewhere**: downstream task performance, noise sensitivity degradation, cluster separability preservation, parameter sensitivity exploration, and information-theoretic metrics.

## Files

- **dimensionality_reduction_comparison.py**: Advanced Dimensionality Reduction Analysis
  - Compares PCA, t-SNE, and UMAP on five different synthetic datasets
  - Tests techniques on: high-dimensional spherical clusters, non-linear moons, concentric circles, Swiss roll manifold, and linear structures with noise
  - **UNIQUE FEATURES**: 
    - **Downstream classification performance**: Tests how well each reduced space works for KNN classification
    - **Noise sensitivity analysis**: Measures how each technique degrades with increasing noise levels
    - **Cluster separability preservation**: Quantitative measure of inter/intra-cluster distance ratios
    - **Parameter sensitivity heatmaps**: Systematic exploration of t-SNE perplexity and UMAP n_neighbors
    - **Information-theoretic metrics**: Entropy-based distribution analysis
  - Evaluates performance using comprehensive metrics: trustworthiness, distance correlation, cluster separability, downstream performance retention, explained variance ratio, and runtime
  - Generates multiple detailed visualizations including noise sensitivity curves and parameter sensitivity plots
  - Provides actionable insights on when to use each technique based on data characteristics and use case requirements

## Key Concepts Demonstrated

- **PCA (Principal Component Analysis)**: Linear dimensionality reduction, maximizes variance
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Non-linear technique focused on local structure preservation
- **UMAP (Uniform Manifold Approximation and Projection)**: Non-linear technique balancing local and global structure
- **Information Preservation Metrics**: Trustworthiness (local), Distance Correlation (global), Explained Variance Ratio (PCA-specific)
- **Runtime Performance Analysis**: Comparative timing analysis across techniques
- **Data Preprocessing**: Feature standardization for dimensionality reduction
- **Synthetic Data Generation**: Creating controlled datasets with specific characteristics (linear, non-linear, high-dimensional, manifolds)
- **Comparative Analysis**: Systematic methodology for choosing the right dimensionality reduction technique

## Datasets Used

1. **High-dimensional Spherical Clusters**: 10D data with 4 well-separated spherical clusters (ideal for PCA)
2. **Non-linear Moons**: 10D embedding of 2D moon-shaped clusters (challenges PCA, ideal for t-SNE/UMAP)
3. **Concentric Circles**: 10D embedding of nested circular structures (impossible for PCA, ideal for t-SNE/UMAP)
4. **Swiss Roll Manifold**: 3D manifold that needs to be "unfolded" (manifold learning challenge)
5. **Linear Structure with Noise**: 10D data with underlying linear trend plus noise (ideal for PCA)

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
python dimensionality_reduction_comparison.py
```

## Output

The script will:
- Display detailed comparison tables with advanced metrics (Trustworthiness, Distance Correlation, Cluster Separability, Downstream Performance Retention, Explained Variance Ratio, Runtime) for each technique
- Save comprehensive visualization plots:
  - `dimensionality_reduction_comparison.png`: Side-by-side comparison of all techniques on all datasets
  - `dimensionality_reduction_metrics_comparison.png`: Bar charts comparing runtime, trustworthiness, and distance correlation across datasets
  - `noise_sensitivity_analysis.png`: **UNIQUE** - Shows how each technique degrades with increasing noise (trustworthiness and downstream performance)
  - `parameter_sensitivity_analysis.png`: **UNIQUE** - Parameter sensitivity curves for t-SNE (perplexity) and UMAP (n_neighbors)
- Provide analysis and recommendations on when to use each technique
- Highlight the limitations and strengths of each approach
- Demonstrate runtime performance differences between techniques
- Show how different techniques preserve local vs. global structure
- **UNIQUE**: Analyze downstream task performance (classification accuracy retention)
- **UNIQUE**: Quantify cluster separability preservation in reduced spaces

## Technique Characteristics

### PCA
- **Best for**: Linear structures, high-dimensional data with linear relationships
- **Strengths**: Fast, interpretable, preserves global variance, deterministic
- **Limitations**: Cannot capture non-linear structures, assumes linear relationships
- **Use when**: Data has linear structure, you need fast computation, or interpretability matters
- **Key Metric**: Explained Variance Ratio shows how much variance is preserved

### t-SNE
- **Best for**: Non-linear structures, visualization, local neighborhood preservation
- **Strengths**: Excellent at preserving local structure, great for visualization
- **Limitations**: Slow, non-deterministic, doesn't preserve global structure well, sensitive to parameters
- **Use when**: Visualizing high-dimensional data, exploring local clusters, non-linear structures
- **Key Metric**: Trustworthiness measures how well local neighborhoods are preserved

### UMAP
- **Best for**: Non-linear structures, balance between local and global preservation
- **Strengths**: Faster than t-SNE, preserves both local and global structure better than t-SNE
- **Limitations**: Still slower than PCA, requires parameter tuning
- **Use when**: You need non-linear reduction with better global structure than t-SNE
- **Key Metric**: Balance between trustworthiness and distance correlation

## Unique Features (Not Found in Standard Online Tutorials)

Unlike standard dimensionality reduction tutorials, this script includes **advanced analyses rarely seen elsewhere**:
- **Downstream Task Performance Analysis**: Tests how well each reduced space works for classification tasks (KNN), measuring performance retention - critical for production use cases
- **Noise Sensitivity Degradation Analysis**: Systematically tests how each technique degrades with increasing noise levels (0.0 to 0.5), showing robustness characteristics
- **Cluster Separability Preservation**: Quantitative metric measuring inter-cluster to intra-cluster distance ratios in reduced space - shows how well clusters remain separated
- **Parameter Sensitivity Exploration**: Creates systematic parameter sweeps for t-SNE (perplexity) and UMAP (n_neighbors) to understand tuning requirements
- **Information-Theoretic Metrics**: Entropy-based analysis of data distribution in reduced space
- **Systematically compares multiple techniques** to understand trade-offs
- **Uses diverse synthetic datasets** including Swiss roll manifolds and high-dimensional embeddings
- **Employs comprehensive evaluation metrics** - Trustworthiness (local), Distance Correlation (global), Cluster Separability, Downstream Retention, Explained Variance Ratio (PCA), and Runtime
- **Includes runtime performance analysis** - compares computational efficiency across techniques
- **Focuses on decision-making** - when to use which technique based on data characteristics and use case requirements
- **Demonstrates real-world challenges** like non-linear structures, high-dimensional data, manifolds, and noise
- **Generates multiple comparative visualizations** - embeddings, metrics, noise sensitivity curves, and parameter sensitivity plots
- **Provides actionable insights** based on quantitative comparisons across multiple dimensions
- **Shows local vs. global preservation trade-offs** - different techniques excel at different aspects

## Key Insights

1. **No one-size-fits-all**: Different techniques excel at different data structures
2. **Linear vs. Non-linear**: Understanding whether your data has linear or non-linear structure is crucial
3. **Local vs. Global**: t-SNE preserves local structure but not global; UMAP balances both; PCA preserves global variance
4. **Computational trade-offs**: PCA is fastest, UMAP is moderate, t-SNE is slowest
5. **Interpretability**: PCA components are interpretable (linear combinations), t-SNE/UMAP are not
6. **Parameter sensitivity**: t-SNE and UMAP require careful parameter tuning (perplexity, n_neighbors)
7. **Visualization quality**: All techniques can produce good visualizations, but for different purposes
8. **Metric selection matters**: Use trustworthiness for local structure, distance correlation for global structure

## Technical Details

### Metrics Explained

- **Trustworthiness**: Measures how well local neighborhoods (k-nearest neighbors) are preserved in the reduced space. Higher is better (0-1 scale).
- **Distance Correlation**: Correlation between pairwise distances in original and reduced space. Higher is better (0-1 scale), measures global structure preservation.
- **Cluster Separability**: Ratio of inter-cluster to intra-cluster distances. Higher is better, shows how well clusters remain separated in reduced space.
- **Downstream Performance Retention**: Ratio of classification accuracy in reduced space to original space. Higher is better (0-1+ scale), critical for production use cases.
- **Explained Variance Ratio**: (PCA only) Proportion of total variance explained by principal components. Higher is better (0-1 scale).
- **Entropy Score**: Information-theoretic measure of data distribution uniformity in reduced space. Higher is better (0-1 scale).
- **Runtime**: Computational time in milliseconds. Lower is better.

### When to Use Each Technique

- **PCA**: Linear data, need speed, interpretability matters, high-dimensional linear relationships
- **t-SNE**: Visualization focus, non-linear data, local structure important, exploratory analysis
- **UMAP**: Non-linear data, need both local and global structure, faster than t-SNE, production use cases
