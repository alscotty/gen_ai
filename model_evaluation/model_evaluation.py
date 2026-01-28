import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
sns.set(style="whitegrid")


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    label: str


def generate_drifted_datasets(
    n_samples: int = 4000,
    n_features: int = 12,
    n_informative: int = 6,
) -> List[DatasetSplit]:
    """
    Generate a base dataset and two drifted variants:
    - Era A: original prevalence, clean features
    - Era B: higher positive prevalence
    - Era C: feature shift + more noise
    """
    rng = np.random.RandomState(RANDOM_STATE)

    def make_split(
        class_weight: float,
        feature_shift: float = 0.0,
        noise_scale: float = 0.0,
        label: str = "era",
    ) -> DatasetSplit:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=2,
            weights=[1 - class_weight, class_weight],
            flip_y=0.02,
            class_sep=1.2,
            random_state=RANDOM_STATE,
        )

        # Apply controlled feature shift and noise
        X = X + feature_shift
        if noise_scale > 0:
            X += rng.normal(scale=noise_scale, size=X.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
        )
        return DatasetSplit(X_train, X_test, y_train, y_test, label)

    era_a = make_split(class_weight=0.25, feature_shift=0.0, noise_scale=0.0, label="Era A (base)")
    era_b = make_split(class_weight=0.40, feature_shift=0.0, noise_scale=0.0, label="Era B (prevalence shift)")
    era_c = make_split(class_weight=0.30, feature_shift=0.6, noise_scale=0.5, label="Era C (feature + noise shift)")

    return [era_a, era_b, era_c]


def build_models() -> Dict[str, object]:
    """Create a small but contrasting pair of models."""
    base_lr = LogisticRegression(max_iter=200, solver="lbfgs")
    # Calibrated RF to contrast discrimination vs calibration
    base_rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=5,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )
    calibrated_rf = CalibratedClassifierCV(base_rf, method="isotonic", cv=3)

    return {
        "LogisticRegression": base_lr,
        "CalibratedRandomForest": calibrated_rf,
    }


def evaluate_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute a rich set of metrics from probabilities."""
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
    }
    return metrics


def fit_and_evaluate_across_eras(
    eras: List[DatasetSplit],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]], StandardScaler]:
    """
    Train on Era A and evaluate on all eras to simulate real-world deployment.
    Returns:
      - metrics table
      - probabilities per model per era for further analysis
      - fitted scaler
    """
    era_a = eras[0]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(era_a.X_train)

    models = build_models()
    probas: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in models}
    rows = []

    for model_name, model in models.items():
        model.fit(X_train_scaled, era_a.y_train)

        for era in eras:
            X_test_scaled = scaler.transform(era.X_test)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            probas[model_name][era.label] = y_proba

            metrics = evaluate_model(era.y_test, y_proba)
            metrics.update({"model": model_name, "era": era.label})
            rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    return metrics_df, probas, scaler


def plot_roc_pr_curves(
    eras: List[DatasetSplit],
    probas: Dict[str, Dict[str, np.ndarray]],
    output_path: str = "model_evaluation_roc_pr.png",
) -> None:
    """Plot ROC and PR curves for the base era only, to keep plots focused."""
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

    era = eras[0]  # Era A
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model_name, era_probs in probas.items():
        y_proba = era_probs[era.label]
        RocCurveDisplay.from_predictions(
            era.y_test, y_proba, name=model_name, ax=axes[0]
        )
        PrecisionRecallDisplay.from_predictions(
            era.y_test, y_proba, name=model_name, ax=axes[1]
        )

    axes[0].set_title("ROC Curves (Era A)")
    axes[1].set_title("Precision-Recall Curves (Era A)")
    fig.suptitle("Discrimination Comparison: ROC & PR Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_calibration_curves(
    eras: List[DatasetSplit],
    probas: Dict[str, Dict[str, np.ndarray]],
    output_path: str = "model_evaluation_calibration.png",
    n_bins: int = 10,
) -> None:
    """Plot simple reliability diagrams for Era A."""
    from sklearn.calibration import calibration_curve

    era = eras[0]
    fig, ax = plt.subplots(figsize=(6, 5))

    for model_name, era_probs in probas.items():
        y_proba = era_probs[era.label]
        prob_true, prob_pred = calibration_curve(
            era.y_test, y_proba, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(prob_pred, prob_true, marker="o", label=model_name)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Diagrams (Era A)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def decision_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> pd.DataFrame:
    """
    Compute expected cost per decision across thresholds.
    This is a light-weight, implementation-focused version of decision curve analysis.
    """
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fp = np.logical_and(y_pred == 1, y_true == 0).sum()
        fn = np.logical_and(y_pred == 0, y_true == 1).sum()
        n = len(y_true)
        expected_cost = (cost_fp * fp + cost_fn * fn) / n
        rows.append({"threshold": t, "expected_cost": expected_cost})
    return pd.DataFrame(rows)


def plot_decision_curves(
    era: DatasetSplit,
    probas: Dict[str, Dict[str, np.ndarray]],
    output_path: str = "model_evaluation_decision_curves.png",
) -> None:
    """
    Visualize how preferred thresholds differ between models
    under asymmetric misclassification costs.
    """
    thresholds = np.linspace(0.05, 0.95, 25)
    cost_fp, cost_fn = 1.0, 5.0  # false negatives are 5x more expensive

    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, era_probs in probas.items():
        y_proba = era_probs[era.label]
        dc = decision_curve(era.y_test, y_proba, thresholds, cost_fp, cost_fn)
        ax.plot(dc["threshold"], dc["expected_cost"], label=model_name)

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost per sample")
    ax.set_title("Decision Curves (cost_fn = 5 × cost_fp)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_drift_sensitivity(
    metrics_df: pd.DataFrame,
    output_path: str = "model_evaluation_drift_sensitivity.png",
) -> None:
    """Show how metrics move across eras for each model."""
    value_vars = ["accuracy", "recall", "roc_auc", "pr_auc", "brier"]
    melted = metrics_df.melt(
        id_vars=["model", "era"], value_vars=value_vars, var_name="metric", value_name="value"
    )

    g = sns.catplot(
        data=melted,
        x="era",
        y="value",
        hue="model",
        col="metric",
        kind="point",
        col_wrap=3,
        height=3,
        sharey=False,
    )
    g.fig.suptitle("Metric Drift Across Deployment Eras", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(g.fig)


def main() -> None:
    print("Generating synthetic datasets with controlled drift...")
    eras = generate_drifted_datasets()

    print("Training models on Era A and evaluating on all eras...")
    metrics_df, probas, _ = fit_and_evaluate_across_eras(eras)
    pd.set_option("display.precision", 3)
    print("\n=== Metrics across eras ===")
    print(metrics_df.pivot_table(index=["model", "era"]))

    print("\nSaving discrimination plots (ROC/PR)...")
    plot_roc_pr_curves(eras, probas)

    print("Saving calibration plots...")
    plot_calibration_curves(eras, probas)

    print("Saving decision curve analysis...")
    plot_decision_curves(eras[0], probas)

    print("Saving drift sensitivity plots...")
    plot_drift_sensitivity(metrics_df)

    print("\nKey observations to inspect manually:")
    print("- Compare how models trade off ROC AUC vs Brier score.")
    print("- Look at how metric rankings change across eras.")
    print("- Inspect decision curves to choose thresholds aligned with your cost structure.")
    print("\nAll plots saved as PNG files in the current directory.")


if __name__ == "__main__":
    main()

