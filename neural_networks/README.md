# Neural Networks

This directory contains comparative studies of neural network architectures and training strategies using TensorFlow/Keras. Unlike standard tutorials, these examples focus on understanding design trade-offs through systematic comparisons.

## Files

- **basic_nn.py**: Architecture Comparison Study
  - Compares 5 different neural network architectures (shallow, medium, deep, wide, narrow)
  - Uses synthetic classification data for controlled experimentation
  - Analyzes the relationship between model complexity (parameters) and performance
  - Evaluates overfitting patterns across different architectures
  - Provides insights on architecture selection based on efficiency metrics

- **regression_nn.py**: Loss Function Comparison Study
  - Compares 4 different loss functions (MSE, MAE, Huber with two delta values)
  - Uses synthetic regression data with intentional outliers
  - Demonstrates how different loss functions handle outliers and error distributions
  - Analyzes robustness and sensitivity characteristics of each loss function
  - Provides guidance on when to use each loss function

## Key Concepts Demonstrated

- **Architecture Design**: Understanding depth vs width trade-offs, parameter efficiency
- **Loss Function Selection**: MSE vs MAE vs Huber loss, outlier robustness
- **Activation Functions**: ReLU for hidden layers, softmax for multi-class classification
- **Optimizers**: Adam optimizer for gradient descent
- **Data Preprocessing**: Feature scaling (StandardScaler) for neural networks
- **Model Evaluation**: Comprehensive metrics (accuracy, RMSE, MAE) and overfitting analysis
- **Comparative Analysis**: Systematic comparison methodology for neural network design decisions

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

3. Run the examples:
```bash
python basic_nn.py
python regression_nn.py
```

## Output

Both scripts will:
- Display detailed comparison tables and metrics
- Save comprehensive visualization plots (architecture_comparison.png, loss_function_comparison.png)
- Provide key insights and recommendations based on the comparisons
- Show training curves, performance metrics, and prediction quality for all compared approaches

## Unique Features

Unlike standard tutorials that show a single model, these scripts:
- **Systematically compare multiple approaches** to understand trade-offs
- **Use synthetic data** for controlled, reproducible experiments
- **Focus on decision-making** - when to use which architecture or loss function
- **Provide actionable insights** based on quantitative comparisons
- **Analyze efficiency** - performance per parameter, not just raw accuracy

