"""
Optimization Dynamics Study:
Comparative analysis of neural network training dynamics under different optimizers
and learning-rate settings. Focuses on *how* models learn over time, not just the
final accuracy, using synthetic data for controlled experiments.

Unique features:
- Measures "time-to-competence": epochs needed to reach a target performance level
- Quantifies generalization gap trajectories (train vs validation over time)
- Computes "learning efficiency" = area under the learning curve (AULC) per update
- Compares optimizers (SGD+momentum, RMSprop, Adam, AdamW-style) across LR grids
- Visualizes stability vs speed trade-offs in optimizer choice
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)
tf.random.set_seed(42)
sns.set_style("whitegrid")


@dataclass
class RunConfig:
    optimizer_name: str
    learning_rate: float
    weight_decay: float | None = None


@dataclass
class RunResults:
    config: RunConfig
    history: Dict[str, List[float]]
    train_time_s: float
    time_to_target_epoch: int | None
    aulc_val: float
    final_metrics: Dict[str, float]


def generate_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic dataset with mildly challenging decision boundary."""
    print("=" * 70)
    print("Optimization Dynamics Study")
    print("=" * 70)

    print("\n1. Generating synthetic classification dataset with structured difficulty...")
    X, y = make_classification(
        n_samples=2000,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=2,
        class_sep=1.0,
        flip_y=0.05,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"   Train shape: {X_train_scaled.shape}, Val shape: {X_val_scaled.shape}")
    print(f"   Class distribution (y_train): {np.bincount(y_train)}")

    return X_train_scaled, X_val_scaled, y_train, y_val


def build_model(input_dim: int, weight_decay: float | None = None) -> tf.keras.Model:
    """Shared architecture so only optimization differs."""
    regularizer = (
        tf.keras.regularizers.l2(weight_decay) if weight_decay is not None else None
    )
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    return model


def make_optimizer(config: RunConfig) -> tf.keras.optimizers.Optimizer:
    """Factory for optimizers with consistent hyperparameter handling."""
    lr = config.learning_rate
    wd = config.weight_decay

    if config.optimizer_name == "sgd_momentum":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    if config.optimizer_name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9)
    if config.optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if config.optimizer_name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd or 1e-4)

    raise ValueError(f"Unknown optimizer: {config.optimizer_name}")


def area_under_curve(values: List[float]) -> float:
    """Simple trapezoidal integration over epochs (manual implementation)."""
    if not values:
        return 0.0
    # Use a small manual implementation instead of numpy.trapz for portability
    total = 0.0
    for i in range(1, len(values)):
        total += 0.5 * (values[i] + values[i - 1])  # Δx = 1 between epochs
    return float(total)


def time_to_target_curve(values: List[float], target: float) -> int | None:
    """First epoch index where value crosses target (0-based)."""
    for i, v in enumerate(values):
        if v >= target:
            return i
    return None


def run_single_experiment(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    config: RunConfig,
    epochs: int = 50,
    batch_size: int = 64,
    target_val_acc: float = 0.85,
) -> RunResults:
    """Train a model under a specific optimizer configuration and collect rich metrics."""
    num_classes = len(np.unique(y_train))
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

    model = build_model(X_train.shape[1], weight_decay=config.weight_decay)
    optimizer = make_optimizer(config)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(
        f"\n   → Training with {config.optimizer_name:<10} | "
        f"lr={config.learning_rate:.4f} | wd={config.weight_decay or 0:.1e}"
    )
    start_time = time.time()
    history = model.fit(
        X_train,
        y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    end_time = time.time()

    train_time = end_time - start_time
    hist = history.history

    val_acc = hist["val_accuracy"]
    val_aulc = area_under_curve(val_acc)
    ttc_epoch = time_to_target_curve(val_acc, target_val_acc)

    final_train_loss = float(hist["loss"][-1])
    final_val_loss = float(hist["val_loss"][-1])
    final_train_acc = float(hist["accuracy"][-1])
    final_val_acc = float(val_acc[-1])

    generalization_gaps = [tr - va for tr, va in zip(hist["accuracy"], hist["val_accuracy"])]
    max_gap = float(max(generalization_gaps))

    results = RunResults(
        config=config,
        history=hist,
        train_time_s=train_time,
        time_to_target_epoch=ttc_epoch,
        aulc_val=val_aulc,
        final_metrics={
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "final_train_acc": final_train_acc,
            "final_val_acc": final_val_acc,
            "max_generalization_gap": max_gap,
        },
    )

    print(
        f"      final val_acc={final_val_acc:.3f}, "
        f"AULC={val_aulc:.1f}, "
        f"time={train_time:.2f}s, "
        f"ttc_epoch={ttc_epoch if ttc_epoch is not None else 'N/A'}"
    )

    return results


def run_grid(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> List[RunResults]:
    """Run a small grid over optimizers and learning rates."""
    print("\n2. Defining optimizer/learning-rate grid...")

    lr_values = [1e-3, 3e-3, 1e-2]
    configs: List[RunConfig] = []

    for optimizer_name in ["sgd_momentum", "rmsprop", "adam", "adamw"]:
        for lr in lr_values:
            if optimizer_name == "adamw":
                cfg = RunConfig(optimizer_name=optimizer_name, learning_rate=lr, weight_decay=1e-3)
            else:
                cfg = RunConfig(optimizer_name=optimizer_name, learning_rate=lr, weight_decay=None)
            configs.append(cfg)

    print(f"   Total runs: {len(configs)} (4 optimizers × {len(lr_values)} learning rates)")

    all_results: List[RunResults] = []
    for cfg in configs:
        res = run_single_experiment(X_train, X_val, y_train, y_val, cfg)
        all_results.append(res)

    return all_results


def summarize_results(results: List[RunResults]) -> None:
    """Print a compact comparison table with dynamics-aware metrics."""
    print("\n3. Optimization Dynamics Summary (per run):")
    print("   " + "-" * 86)
    print(
        "   "
        f"{'Opt':<10} {'LR':<8} {'WD':<8} "
        f"{'Final Val Acc':<14} {'AULC (val)':<14} "
        f"{'TTC Epoch>=0.85':<16} {'Max Gap':<10}"
    )
    print("   " + "-" * 86)

    # Sort by final validation accuracy primarily, then by AULC
    for r in sorted(
        results,
        key=lambda rr: (rr.final_metrics["final_val_acc"], rr.aulc_val),
        reverse=True,
    ):
        cfg = r.config
        fm = r.final_metrics
        print(
            "   "
            f"{cfg.optimizer_name:<10} "
            f"{cfg.learning_rate:<8.3g} "
            f"{(cfg.weight_decay or 0):<8.1e} "
            f"{fm['final_val_acc']:<14.3f} "
            f"{r.aulc_val:<14.1f} "
            f"{(r.time_to_target_epoch if r.time_to_target_epoch is not None else -1):<16} "
            f"{fm['max_generalization_gap']:<10.3f}"
        )


def plot_learning_curves(results: List[RunResults]) -> None:
    """Create multi-panel visualization of dynamics across optimizers."""
    print("\n4. Creating optimization dynamics visualizations...")

    optimizers = sorted({r.config.optimizer_name for r in results})
    lr_values = sorted({r.config.learning_rate for r in results})

    # Figure 1: validation accuracy curves grouped by optimizer
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes1 = axes1.flatten()

    for ax, opt in zip(axes1, optimizers):
        subset = [r for r in results if r.config.optimizer_name == opt]
        for r in subset:
            lr = r.config.learning_rate
            ax.plot(
                r.history["val_accuracy"],
                label=f"lr={lr:.0e}",
                linewidth=2,
                alpha=0.8,
            )
        ax.set_title(f"{opt} – validation accuracy over epochs", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("optimizer_val_accuracy_curves.png", dpi=200, bbox_inches="tight")
    print("   Saved 'optimizer_val_accuracy_curves.png'")

    # Figure 2: generalization gap trajectories (acc_train - acc_val)
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes2 = axes2.flatten()

    for ax, opt in zip(axes2, optimizers):
        subset = [r for r in results if r.config.optimizer_name == opt]
        for r in subset:
            lr = r.config.learning_rate
            train_acc = r.history["accuracy"]
            val_acc = r.history["val_accuracy"]
            gap = [tr - va for tr, va in zip(train_acc, val_acc)]
            ax.plot(
                gap,
                label=f"lr={lr:.0e}",
                linewidth=2,
                alpha=0.8,
            )
        ax.set_title(f"{opt} – generalization gap (train–val)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Acc gap")
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("optimizer_generalization_gap_curves.png", dpi=200, bbox_inches="tight")
    print("   Saved 'optimizer_generalization_gap_curves.png'")

    # Figure 3: speed vs quality scatter (AULC vs final val acc) colored by optimizer
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("tab10", n_colors=len(optimizers))
    opt_to_color = {opt: palette[i] for i, opt in enumerate(optimizers)}

    for r in results:
        cfg = r.config
        fm = r.final_metrics
        ax3.scatter(
            r.aulc_val,
            fm["final_val_acc"],
            s=140,
            color=opt_to_color[cfg.optimizer_name],
            alpha=0.85,
            edgecolors="black",
            linewidth=0.8,
        )
        label = f"{cfg.optimizer_name}, lr={cfg.learning_rate:.0e}"
        ax3.annotate(
            label,
            (r.aulc_val, fm["final_val_acc"]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=8,
        )

    ax3.set_xlabel("AULC (validation accuracy over epochs)")
    ax3.set_ylabel("Final validation accuracy")
    ax3.set_title(
        "Optimization Efficiency: learning-curve area vs final performance",
        fontsize=12,
        fontweight="bold",
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=opt_to_color[opt],
            markeredgecolor="black",
            markersize=10,
            label=opt,
        )
        for opt in optimizers
    ]
    ax3.legend(handles=handles, title="Optimizer", fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("optimizer_efficiency_scatter.png", dpi=200, bbox_inches="tight")
    print("   Saved 'optimizer_efficiency_scatter.png'")

    # Figure 4: heatmap of time-to-competence (epochs to reach target val acc)
    target = 0.85
    heatmap_data = np.full((len(optimizers), len(lr_values)), fill_value=np.nan)

    for i, opt in enumerate(optimizers):
        for j, lr in enumerate(lr_values):
            match = [
                r
                for r in results
                if r.config.optimizer_name == opt and np.isclose(r.config.learning_rate, lr)
            ]
            if not match:
                continue
            ttc = time_to_target_curve(match[0].history["val_accuracy"], target)
            if ttc is not None:
                heatmap_data[i, j] = ttc

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        xticklabels=[f"{lr:.0e}" for lr in lr_values],
        yticklabels=optimizers,
        cbar_kws={"label": f"Epochs to reach val_acc ≥ {target:.2f}"},
        ax=ax4,
    )
    ax4.set_xlabel("Learning rate")
    ax4.set_ylabel("Optimizer")
    ax4.set_title("Time-to-competence heatmap (lower is better)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("optimizer_time_to_competence_heatmap.png", dpi=200, bbox_inches="tight")
    print("   Saved 'optimizer_time_to_competence_heatmap.png'")


def main() -> None:
    X_train, X_val, y_train, y_val = generate_dataset()
    all_results = run_grid(X_train, X_val, y_train, y_val)
    summarize_results(all_results)
    plot_learning_curves(all_results)

    print("\n5. Key Takeaways (study-specific):")
    print(
        """
   • Different optimizers trace very different learning trajectories even when final accuracy is similar.
   • AULC (area under the validation accuracy curve) highlights optimizers that learn *useful* models quickly,
     not just eventually.
   • Time-to-competence surfaces configurations that reach a target quality fast, which matters in
     interactive or resource-constrained settings.
   • Generalization gap curves make it easy to spot combinations that aggressively overfit early.
   • Rather than asking "Which optimizer is best?", this script encourages asking "Which optimizer is best
     for my speed vs stability vs generalization trade-off?".
"""
    )
    print("=" * 70)
    print("Optimization dynamics study complete. See generated PNG files for visuals.")
    print("=" * 70)


if __name__ == "__main__":
    main()

