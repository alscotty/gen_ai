"""
Neural Network Regression: Loss Function Comparison
Explores how different loss functions affect neural network training for regression.
Compares MSE, MAE, and Huber loss to understand their characteristics and when to use each.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("Neural Network Regression: Loss Function Comparison")
print("=" * 70)

# Generate synthetic regression dataset with some outliers
# This allows us to see how different loss functions handle outliers
print("\n1. Generating synthetic regression dataset...")
print("   Creating dataset with intentional outliers to test loss function robustness")
X, y = make_regression(
    n_samples=800,
    n_features=6,
    n_informative=5,
    noise=15,
    random_state=42
)

# Add some outliers to make the comparison more interesting
outlier_indices = np.random.choice(len(y), size=int(0.05 * len(y)), replace=False)
y[outlier_indices] += np.random.choice([-1, 1], size=len(outlier_indices)) * np.abs(y[outlier_indices]) * 0.8

print(f"   Dataset shape: {X.shape}")
print(f"   Target range: {y.min():.2f} to {y.max():.2f}")
print(f"   Target mean: {y.mean():.2f}, std: {y.std():.2f}")
print(f"   Outliers added: {len(outlier_indices)} ({len(outlier_indices)/len(y)*100:.1f}%)")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Standardize target
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"\n   Training set size: {X_train_scaled.shape[0]}")
print(f"   Test set size: {X_test_scaled.shape[0]}")

# Define loss functions to compare
loss_functions = {
    'MSE (Mean Squared Error)': {
        'loss': 'mse',
        'description': 'Sensitive to outliers, penalizes large errors quadratically'
    },
    'MAE (Mean Absolute Error)': {
        'loss': 'mae',
        'description': 'Robust to outliers, treats all errors linearly'
    },
    'Huber Loss (δ=1.0)': {
        'loss': tf.keras.losses.Huber(delta=1.0),
        'description': 'Combines MSE and MAE, robust to outliers'
    },
    'Huber Loss (δ=2.0)': {
        'loss': tf.keras.losses.Huber(delta=2.0),
        'description': 'More MSE-like, less robust than δ=1.0'
    }
}

print("\n2. Building and training models with different loss functions...")
print("   Comparing how each loss function affects training and predictions")

results = {}
histories = {}
models = {}

def build_model():
    """Build a standard regression model architecture"""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

for loss_name, loss_config in loss_functions.items():
    print(f"\n   [{loss_name}]")
    print(f"      {loss_config['description']}")
    
    # Build model
    model = build_model()
    model.compile(
        optimizer='adam',
        loss=loss_config['loss'],
        metrics=['mae', 'mse']
    )
    
    # Train model
    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=60,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate
    train_metrics = model.evaluate(X_train_scaled, y_train_scaled, verbose=0)
    test_metrics = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
    
    # Make predictions
    y_train_pred_scaled = model.predict(X_train_scaled, verbose=0)
    y_test_pred_scaled = model.predict(X_test_scaled, verbose=0)
    
    # Convert back to original scale
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
    
    # Calculate metrics in original scale
    train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    test_mae = np.mean(np.abs(y_test - y_test_pred))
    
    # Calculate outlier handling (how well it predicts outliers)
    outlier_mask = np.isin(np.arange(len(y_test)), 
                           [np.where(y_test == y[i])[0][0] if len(np.where(y_test == y[i])[0]) > 0 
                            else -1 for i in outlier_indices if i < len(y_test)])
    if np.any(outlier_mask):
        outlier_mae = np.mean(np.abs(y_test[outlier_mask] - y_test_pred[outlier_mask]))
    else:
        outlier_mae = np.nan
    
    results[loss_name] = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'outlier_mae': outlier_mae
    }
    histories[loss_name] = history
    models[loss_name] = model
    
    print(f"      Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f}")

# Compare results
print("\n3. Loss Function Comparison Summary:")
print("   " + "-" * 70)
print(f"   {'Loss Function':<25} {'Test RMSE':<12} {'Test MAE':<12} {'Outlier MAE':<12}")
print("   " + "-" * 70)

for loss_name, metrics in sorted(results.items(), key=lambda x: x[1]['test_mae']):
    outlier_str = f"{metrics['outlier_mae']:.4f}" if not np.isnan(metrics['outlier_mae']) else "N/A"
    print(f"   {loss_name:<25} {metrics['test_rmse']:<12.4f} {metrics['test_mae']:<12.4f} {outlier_str:<12}")

# Visualize comparison
print("\n4. Creating comparison visualizations...")

fig = plt.figure(figsize=(16, 10))

# Plot 1: Training loss curves
ax1 = plt.subplot(2, 3, 1)
for loss_name, history in histories.items():
    plt.plot(history.history['loss'], label=loss_name, alpha=0.7, linewidth=2)
plt.title('Training Loss Comparison', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.yscale('log')

# Plot 2: Validation MAE curves
ax2 = plt.subplot(2, 3, 2)
for loss_name, history in histories.items():
    plt.plot(history.history['val_mae'], label=loss_name, alpha=0.7, linewidth=2)
plt.title('Validation MAE Comparison', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(fontsize=8)
plt.grid(alpha=0.3)

# Plot 3: Test metrics comparison
ax3 = plt.subplot(2, 3, 3)
loss_names = list(results.keys())
test_rmses = [results[name]['test_rmse'] for name in loss_names]
test_maes = [results[name]['test_mae'] for name in loss_names]
x_pos = np.arange(len(loss_names))
width = 0.35
plt.bar(x_pos - width/2, test_rmses, width, label='RMSE', alpha=0.7)
plt.bar(x_pos + width/2, test_maes, width, label='MAE', alpha=0.7)
plt.xlabel('Loss Function')
plt.ylabel('Error')
plt.title('Test Set Performance', fontsize=12, fontweight='bold')
plt.xticks(x_pos, [name.split()[0] for name in loss_names], rotation=45, ha='right')
plt.legend()
plt.grid(alpha=0.3, axis='y')

# Plot 4-7: Predictions vs actual for each loss function
for idx, (loss_name, model) in enumerate(models.items(), 4):
    ax = plt.subplot(2, 3, idx)
    y_test_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
    
    plt.scatter(y_test, y_test_pred, alpha=0.5, s=20)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{loss_name.split()[0]}', fontsize=10, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loss_function_comparison.png', dpi=150, bbox_inches='tight')
print("   Comparison saved to 'loss_function_comparison.png'")

# Key insights
print("\n5. Key Insights:")
best_mae = min(results.items(), key=lambda x: x[1]['test_mae'])
best_rmse = min(results.items(), key=lambda x: x[1]['test_rmse'])

print(f"   • Best MAE: {best_mae[0]} ({best_mae[1]['test_mae']:.4f})")
print(f"   • Best RMSE: {best_rmse[0]} ({best_rmse[1]['test_rmse']:.4f})")

# Check outlier handling
if not all(np.isnan([r['outlier_mae'] for r in results.values()])):
    best_outlier = min([(k, v) for k, v in results.items() if not np.isnan(v['outlier_mae'])], 
                      key=lambda x: x[1]['outlier_mae'])
    print(f"   • Best outlier handling: {best_outlier[0]} (Outlier MAE: {best_outlier[1]['outlier_mae']:.4f})")

print("\n   • MSE: Best for normally distributed errors, sensitive to outliers")
print("   • MAE: More robust to outliers, treats all errors equally")
print("   • Huber: Balanced approach, robust but still sensitive to large errors")

print("\n" + "=" * 70)
print("Loss Function Comparison Complete!")
print("=" * 70)

