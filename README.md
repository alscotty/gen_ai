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