"""
Neural Network Architecture Comparison
Demonstrates building and comparing multiple neural network architectures to understand
how different layer configurations affect model performance. Uses synthetic data to
create a controlled learning environment.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from collections import defaultdict

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("Neural Network Architecture Comparison Study")
print("=" * 70)

# Generate synthetic multi-class classification dataset
# This gives us control over complexity and avoids overused datasets
print("\n1. Generating synthetic classification dataset...")
print("   Creating a 4-class problem with 5 features and some noise for realism")
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=4,
    n_redundant=1,
    n_classes=4,
    n_clusters_per_class=1,
    random_state=42,
    class_sep=1.2  # Moderate separation between classes
)

print(f"   Dataset shape: {X.shape}")
print(f"   Number of classes: {len(np.unique(y))}")
print(f"   Class distribution: {np.bincount(y)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (critical for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n   Training set size: {X_train_scaled.shape[0]}")
print(f"   Test set size: {X_test_scaled.shape[0]}")

# Convert labels to one-hot encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=4)

# Define multiple architectures to compare
architectures = {
    'Shallow (1 hidden layer)': [
        tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(4, activation='softmax')
    ],
    'Medium (2 hidden layers)': [
        tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ],
    'Deep (3 hidden layers)': [
        tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ],
    'Wide (fewer layers, more neurons)': [
        tf.keras.layers.Dense(128, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ],
    'Narrow (more layers, fewer neurons)': [
        tf.keras.layers.Dense(8, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ]
}

print("\n2. Building and comparing multiple architectures...")
print("   Testing 5 different architectures to understand design trade-offs")

results = {}
histories = {}

for arch_name, layers in architectures.items():
    print(f"\n   [{arch_name}]")
    
    # Build model
    model = tf.keras.Sequential(layers)
    
    # Compile with same settings for fair comparison
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Count parameters
    total_params = model.count_params()
    print(f"      Total parameters: {total_params:,}")
    
    # Train model
    history = model.fit(
        X_train_scaled,
        y_train_onehot,
        epochs=80,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train_scaled, y_train_onehot, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test_onehot, verbose=0)
    
    results[arch_name] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'params': total_params,
        'layers': len(layers) - 1  # Exclude output layer
    }
    histories[arch_name] = history
    
    print(f"      Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# Compare results
print("\n3. Architecture Comparison Summary:")
print("   " + "-" * 65)
print(f"   {'Architecture':<25} {'Params':<12} {'Test Acc':<12} {'Gap':<10}")
print("   " + "-" * 65)

for arch_name, metrics in sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True):
    gap = metrics['train_acc'] - metrics['test_acc']
    print(f"   {arch_name:<25} {metrics['params']:<12,} {metrics['test_acc']:<12.4f} {gap:<10.4f}")

# Visualize comparison
print("\n4. Creating comparison visualizations...")

fig = plt.figure(figsize=(16, 10))

# Plot 1: Training curves for all architectures
ax1 = plt.subplot(2, 3, 1)
for arch_name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=arch_name, alpha=0.7)
plt.title('Validation Accuracy Comparison', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(fontsize=8)
plt.grid(alpha=0.3)

# Plot 2: Test accuracy vs parameters
ax2 = plt.subplot(2, 3, 2)
arch_names = list(results.keys())
test_accs = [results[name]['test_acc'] for name in arch_names]
params = [results[name]['params'] for name in arch_names]
colors = plt.cm.viridis(np.linspace(0, 1, len(arch_names)))
plt.scatter(params, test_accs, s=200, c=colors, alpha=0.6, edgecolors='black')
for i, name in enumerate(arch_names):
    plt.annotate(name.split()[0], (params[i], test_accs[i]), 
                fontsize=8, ha='center', va='bottom')
plt.xlabel('Number of Parameters')
plt.ylabel('Test Accuracy')
plt.title('Accuracy vs Model Complexity', fontsize=12, fontweight='bold')
plt.grid(alpha=0.3)

# Plot 3: Overfitting analysis (train vs test gap)
ax3 = plt.subplot(2, 3, 3)
gaps = [results[name]['train_acc'] - results[name]['test_acc'] for name in arch_names]
plt.barh(arch_names, gaps, color=colors, alpha=0.6, edgecolor='black')
plt.xlabel('Train - Test Accuracy Gap')
plt.title('Overfitting Analysis', fontsize=12, fontweight='bold')
plt.grid(alpha=0.3, axis='x')

# Plot 4-8: Individual training histories
for idx, (arch_name, history) in enumerate(histories.items(), 4):
    ax = plt.subplot(2, 3, idx)
    plt.plot(history.history['accuracy'], label='Train', alpha=0.7)
    plt.plot(history.history['val_accuracy'], label='Val', alpha=0.7)
    plt.title(f'{arch_name}', fontsize=10, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('architecture_comparison.png', dpi=150, bbox_inches='tight')
print("   Comparison saved to 'architecture_comparison.png'")

# Key insights
print("\n5. Key Insights:")
best_arch = max(results.items(), key=lambda x: x[1]['test_acc'])
worst_arch = min(results.items(), key=lambda x: x[1]['test_acc'])
most_params = max(results.items(), key=lambda x: x[1]['params'])
least_params = min(results.items(), key=lambda x: x[1]['params'])

print(f"   • Best performing: {best_arch[0]} (Test Acc: {best_arch[1]['test_acc']:.4f})")
print(f"   • Most parameters: {most_params[0]} ({most_params[1]['params']:,} params)")
print(f"   • Efficiency winner: {best_arch[0]} achieves {best_arch[1]['test_acc']:.4f} with {best_arch[1]['params']:,} params")

# Check for overfitting
overfitting_arch = max(results.items(), key=lambda x: x[1]['train_acc'] - x[1]['test_acc'])
gap = overfitting_arch[1]['train_acc'] - overfitting_arch[1]['test_acc']
if gap > 0.1:
    print(f"   • Overfitting detected: {overfitting_arch[0]} has {gap:.4f} accuracy gap")

print("\n" + "=" * 70)
print("Architecture Comparison Complete!")
print("=" * 70)

