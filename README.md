# gen_ai

Gen AI Review & Practice

## Tooling
- /streamlit - example dashboards, quick ML classifier for flowers with sklearn
- /tokenization - intro to tokenization, dividing paragraphs down with nltk
    - /stemming : stemming words with nltk porterStemmer and regexpstemmer
    - /lemmatization : superiorer, always returns valid word, no partial words, maintains more of original word meaning
- /neural_networks - comparative studies of neural network design with TensorFlow/Keras
    - /basic_nn : architecture comparison study - compares 5 different network architectures (shallow, deep, wide, narrow) to understand complexity vs performance trade-offs
    - /regression_nn : loss function comparison study - compares MSE, MAE, and Huber loss functions to understand robustness to outliers and error handling
    - Uses synthetic data for controlled experiments, focuses on systematic comparisons and decision-making insights rather than single-model tutorials
- /clustering - comparative study of unsupervised clustering algorithms with scikit-learn
    - /clustering_comparison : algorithm comparison study - compares K-means, DBSCAN, and Agglomerative clustering on diverse synthetic datasets (spherical, non-spherical, concentric, varying density)
    - Evaluates performance using Adjusted Rand Index and Silhouette Score to understand when each algorithm excels
    - Demonstrates how cluster shape and data characteristics determine algorithm effectiveness, providing decision-making guidance for real-world applications
- /dimensionality_reduction - advanced comparative study of dimensionality reduction techniques with unique analyses
    - /dimensionality_reduction_comparison : comprehensive comparison study - compares PCA, t-SNE, and UMAP on diverse synthetic datasets (high-dimensional, non-linear, manifolds)
    - **Unique advanced analyses not found in standard online tutorials:**
        - **Downstream task performance**: Tests how well each reduced space works for KNN classification, measuring performance retention - critical for production use cases where reduced dimensions are used for downstream ML tasks
        - **Noise sensitivity degradation**: Systematically tests how each technique degrades with increasing noise levels (0.0 to 0.5), showing robustness characteristics and failure modes
        - **Cluster separability preservation**: Quantitative metric measuring inter-cluster to intra-cluster distance ratios in reduced space - shows how well clusters remain separated after reduction
        - **Parameter sensitivity exploration**: Creates systematic parameter sweeps for t-SNE (perplexity) and UMAP (n_neighbors) to understand tuning requirements and sensitivity
        - **Information-theoretic metrics**: Entropy-based analysis of data distribution uniformity in reduced space
    - Evaluates performance using comprehensive metrics: trustworthiness (local structure), distance correlation (global structure), cluster separability, downstream performance retention, explained variance ratio (PCA), entropy score, and runtime
    - Tests on diverse datasets: high-dimensional spherical clusters, non-linear moons, concentric circles, Swiss roll manifolds, and linear structures with noise
    - Demonstrates how data structure (linear vs non-linear, high-dimensional vs low-dimensional) determines technique effectiveness
    - Generates multiple visualizations: main comparison plots, metrics charts, noise sensitivity curves, and parameter sensitivity plots
    - Provides actionable insights for real-world applications: when to use PCA (linear, fast, interpretable), t-SNE (visualization, local structure), or UMAP (non-linear with global structure balance)
    - Focuses on systematic comparisons and decision-making guidance rather than single-technique tutorials