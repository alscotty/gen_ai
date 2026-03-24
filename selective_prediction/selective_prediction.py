#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Selective Prediction Under Shift

This project studies selective prediction, where a model can abstain from
low-confidence predictions. The goal is to reason about the trade-off between:
- Risk on accepted predictions
- Coverage (fraction of samples the model chooses to predict)
- Operational cost of abstaining

Unique emphasis:
1) Compare multiple confidence policies (max-probability, margin, entropy).
2) Evaluate policies not just in one test split, but across drifted eras.
3) Quantify "coverage stability" under drift using a dedicated metric.
4) Optimize threshold by expected deployment cost, not just AUC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_style("whitegrid")
OUTPUT_DIR = Path(__file__).resolve().parent


@dataclass
class EraData:
    name: str
    X: np.ndarray
    y: np.ndarray


@dataclass
class PolicyCurve:
    thresholds: List[float]
    coverage: List[float]
    selective_risk: List[float]
    utility: List[float]


def make_base_data(n_samples: int = 6000) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=6,
        n_redundant=3,
        n_clusters_per_class=2,
        class_sep=1.3,
        flip_y=0.03,
        weights=[0.62, 0.38],
        random_state=RANDOM_STATE,
    )
    return X, y


def make_shifted_eras(X: np.ndarray, y: np.ndarray) -> Dict[str, EraData]:
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test_clean, y_val, y_test_clean = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE + 1
    )

    # Era 1: clean distribution
    era_clean = EraData("clean", X_test_clean.copy(), y_test_clean.copy())

    # Era 2: feature drift + mild extra noise
    X_feature_shift = X_test_clean.copy()
    X_feature_shift[:, :4] = 0.8 * X_feature_shift[:, :4] + 0.7
    y_feature_shift = y_test_clean.copy()
    flip_mask = np.random.rand(len(y_feature_shift)) < 0.06
    y_feature_shift[flip_mask] = 1 - y_feature_shift[flip_mask]
    era_feature_shift = EraData("feature_shift", X_feature_shift, y_feature_shift)

    # Era 3: prevalence + heteroskedastic noise shift
    X_prev_shift = X_test_clean.copy()
    y_prev_shift = y_test_clean.copy()
    positive_idx = np.where(y_prev_shift == 1)[0]
    negative_idx = np.where(y_prev_shift == 0)[0]
    keep_neg = np.random.choice(
        negative_idx, size=int(0.6 * len(negative_idx)), replace=False
    )
    keep_idx = np.concatenate([positive_idx, keep_neg])
    X_prev_shift = X_prev_shift[keep_idx]
    y_prev_shift = y_prev_shift[keep_idx]
    noisy_columns = [2, 7, 9]
    X_prev_shift[:, noisy_columns] += np.random.normal(
        loc=0.0, scale=0.9, size=(len(X_prev_shift), len(noisy_columns))
    )
    era_prev_shift = EraData("prevalence_noise_shift", X_prev_shift, y_prev_shift)

    return {
        "train": EraData("train", X_train, y_train),
        "val": EraData("val", X_val, y_val),
        "clean": era_clean,
        "feature_shift": era_feature_shift,
        "prevalence_noise_shift": era_prev_shift,
    }


def build_model(X_train: np.ndarray, y_train: np.ndarray) -> CalibratedClassifierCV:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Blend linear and non-linear signal, then calibrate to improve probability quality.
    base = RandomForestClassifier(
        n_estimators=220,
        max_depth=7,
        min_samples_leaf=8,
        random_state=RANDOM_STATE,
    )
    calibrator = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
    calibrator.fit(X_train_scaled, y_train)

    calibrator.scaler_ = scaler
    return calibrator


def predict_proba(model: CalibratedClassifierCV, X: np.ndarray) -> np.ndarray:
    X_scaled = model.scaler_.transform(X)
    return model.predict_proba(X_scaled)


def score_max_probability(proba_2d: np.ndarray) -> np.ndarray:
    return np.max(proba_2d, axis=1)


def score_margin(proba_2d: np.ndarray) -> np.ndarray:
    p_sorted = np.sort(proba_2d, axis=1)
    return p_sorted[:, -1] - p_sorted[:, -2]


def score_negative_entropy(proba_2d: np.ndarray) -> np.ndarray:
    eps = 1e-12
    entropy = -np.sum(proba_2d * np.log(proba_2d + eps), axis=1)
    return -entropy


def build_density_scorer(X_reference: np.ndarray, k: int = 20):
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nbrs.fit(X_reference)

    ref_dists, _ = nbrs.kneighbors(X_reference)
    ref_mean_dist = ref_dists.mean(axis=1)
    mu = float(np.mean(ref_mean_dist))
    sigma = float(np.std(ref_mean_dist) + 1e-12)

    def scorer(X_query: np.ndarray) -> np.ndarray:
        query_dists, _ = nbrs.kneighbors(X_query)
        query_mean_dist = query_dists.mean(axis=1)
        density_z = (query_mean_dist - mu) / sigma
        return density_z

    return scorer


def evaluate_selective_policy(
    y_true: np.ndarray,
    proba_2d: np.ndarray,
    confidence_scores: np.ndarray,
    thresholds: np.ndarray,
    abstain_cost: float = 0.08,
) -> PolicyCurve:
    y_pred = np.argmax(proba_2d, axis=1)
    n = len(y_true)

    coverage_values: List[float] = []
    risk_values: List[float] = []
    utility_values: List[float] = []

    for t in thresholds:
        accepted = confidence_scores >= t
        coverage = float(np.mean(accepted))

        if accepted.sum() == 0:
            selective_risk = 1.0
            selective_accuracy = 0.0
        else:
            accepted_true = y_true[accepted]
            accepted_pred = y_pred[accepted]
            selective_accuracy = accuracy_score(accepted_true, accepted_pred)
            selective_risk = 1.0 - selective_accuracy

        abstain_rate = 1.0 - coverage
        utility = selective_accuracy * coverage - abstain_cost * abstain_rate

        coverage_values.append(coverage)
        risk_values.append(selective_risk)
        utility_values.append(float(utility))

    return PolicyCurve(
        thresholds=list(thresholds),
        coverage=coverage_values,
        selective_risk=risk_values,
        utility=utility_values,
    )


def make_quantile_thresholds(scores: np.ndarray, n_points: int = 14) -> np.ndarray:
    q = np.linspace(0.05, 0.95, n_points)
    thresholds = np.quantile(scores, q)
    return np.unique(thresholds)


def area_under_risk_coverage(coverage: List[float], risk: List[float]) -> float:
    cov = np.array(coverage)
    rsk = np.array(risk)
    order = np.argsort(cov)
    cov, rsk = cov[order], rsk[order]
    if len(cov) < 2:
        return 0.0
    return float(np.trapezoid(rsk, cov))


def coverage_stability_index(coverages_by_era: Dict[str, float]) -> float:
    vals = np.array(list(coverages_by_era.values()), dtype=float)
    mean_cov = float(np.mean(vals))
    if mean_cov <= 1e-12:
        return 0.0
    # Higher is better: 1 - coefficient of variation, clipped at 0.
    cv = float(np.std(vals) / mean_cov)
    return float(max(0.0, 1.0 - cv))


def threshold_transfer_regret(
    clean_risk: float, shifted_risks: List[float], shifted_coverages: List[float]
) -> float:
    weights = np.array(shifted_coverages, dtype=float)
    if float(np.sum(weights)) <= 1e-12:
        return 0.0
    weights = weights / np.sum(weights)
    shifted = np.array(shifted_risks, dtype=float)
    return float(np.sum(np.maximum(0.0, shifted - clean_risk) * weights))


def coverage_shock_index(clean_coverage: float, shifted_coverages: List[float]) -> float:
    if clean_coverage <= 1e-12:
        return 0.0
    rel_drops = [
        max(0.0, (clean_coverage - cov) / clean_coverage) for cov in shifted_coverages
    ]
    return float(np.mean(rel_drops))


def choose_threshold_from_validation(
    curve: PolicyCurve, min_coverage: float = 0.55
) -> float:
    best_t = curve.thresholds[0]
    best_u = -1e9
    for t, c, u in zip(curve.thresholds, curve.coverage, curve.utility):
        if c >= min_coverage and u > best_u:
            best_u = u
            best_t = t
    return float(best_t)


def summarize_fixed_threshold(
    era_name: str,
    y_true: np.ndarray,
    proba_2d: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> Tuple[float, float, float]:
    accepted = score >= threshold
    coverage = float(np.mean(accepted))
    y_pred = np.argmax(proba_2d, axis=1)
    if accepted.sum() == 0:
        return coverage, 1.0, 0.0
    acc = accuracy_score(y_true[accepted], y_pred[accepted])
    risk = 1.0 - acc
    f1 = f1_score(y_true[accepted], y_pred[accepted], average="macro")
    print(
        f"{era_name:<24} coverage={coverage:.3f} selective_risk={risk:.3f} selective_f1={f1:.3f}"
    )
    return coverage, risk, f1


def plot_outputs(
    curves_by_policy_era: Dict[str, Dict[str, PolicyCurve]],
    thresholds_selected: Dict[str, float],
) -> None:
    plt.figure(figsize=(9, 6))
    for policy_name, era_curves in curves_by_policy_era.items():
        clean_curve = era_curves["clean"]
        plt.plot(
            clean_curve.coverage,
            clean_curve.selective_risk,
            linewidth=2,
            marker="o",
            label=policy_name,
            alpha=0.9,
        )
    plt.xlabel("Coverage")
    plt.ylabel("Selective risk (1 - accuracy on accepted)")
    plt.title("Risk-Coverage Frontier (Clean Era)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "selective_prediction_risk_coverage.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.figure(figsize=(9, 5))
    policy_names = list(curves_by_policy_era.keys())
    eras = ["clean", "feature_shift", "prevalence_noise_shift"]
    x = np.arange(len(policy_names))
    width = 0.23
    for i, era in enumerate(eras):
        vals = []
        for p in policy_names:
            t = thresholds_selected[p]
            curve = curves_by_policy_era[p][era]
            idx = int(np.argmin(np.abs(np.array(curve.thresholds) - t)))
            vals.append(curve.coverage[idx])
        plt.bar(x + (i - 1) * width, vals, width=width, label=era)

    plt.xticks(x, policy_names, rotation=10)
    plt.ylabel("Coverage at validation-selected threshold")
    plt.title("Coverage Stability Across Drifted Eras")
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "selective_prediction_coverage_stability.png",
        dpi=300,
        bbox_inches="tight",
    )

    print("✓ Saved 'selective_prediction_risk_coverage.png'")
    print("✓ Saved 'selective_prediction_coverage_stability.png'")


def main() -> None:
    print("=" * 78)
    print("Selective Prediction Under Shift")
    print("=" * 78)

    X, y = make_base_data()
    eras = make_shifted_eras(X, y)

    model = build_model(eras["train"].X, eras["train"].y)
    abstain_cost = 0.08
    density_scorer = build_density_scorer(eras["train"].X, k=20)

    base_policies = {
        "max_probability": score_max_probability,
        "margin": score_margin,
        "negative_entropy": score_negative_entropy,
    }

    curves_by_policy_era: Dict[str, Dict[str, PolicyCurve]] = {}
    selected_thresholds: Dict[str, float] = {}

    print("\nModel-level baseline (no abstention):")
    for era_name in ["clean", "feature_shift", "prevalence_noise_shift"]:
        proba = predict_proba(model, eras[era_name].X)
        pred = np.argmax(proba, axis=1)
        acc = accuracy_score(eras[era_name].y, pred)
        f1 = f1_score(eras[era_name].y, pred, average="macro")
        print(f"{era_name:<24} accuracy={acc:.3f} f1={f1:.3f}")

    for policy_name, policy_fn in base_policies.items():
        curves_by_policy_era[policy_name] = {}
        val_proba = predict_proba(model, eras["val"].X)
        val_scores = policy_fn(val_proba)
        thresholds = make_quantile_thresholds(val_scores, n_points=14)
        for era_name in ["val", "clean", "feature_shift", "prevalence_noise_shift"]:
            proba = predict_proba(model, eras[era_name].X)
            conf = policy_fn(proba)
            curve = evaluate_selective_policy(
                y_true=eras[era_name].y,
                proba_2d=proba,
                confidence_scores=conf,
                thresholds=thresholds,
                abstain_cost=abstain_cost,
            )
            curves_by_policy_era[policy_name][era_name] = curve

        selected_t = choose_threshold_from_validation(
            curves_by_policy_era[policy_name]["val"], min_coverage=0.55
        )
        selected_thresholds[policy_name] = selected_t

    # Hybrid policy: confidence adjusted by local density mismatch vs training era.
    # This is intentionally not a monotonic transform of max probability.
    hybrid_name = "density_adjusted_confidence"
    curves_by_policy_era[hybrid_name] = {}
    alpha = 0.20
    val_proba = predict_proba(model, eras["val"].X)
    val_base = score_max_probability(val_proba)
    val_density = density_scorer(eras["val"].X)
    val_hybrid = val_base - alpha * np.maximum(0.0, val_density)
    thresholds = make_quantile_thresholds(val_hybrid, n_points=14)

    for era_name in ["val", "clean", "feature_shift", "prevalence_noise_shift"]:
        proba = predict_proba(model, eras[era_name].X)
        base = score_max_probability(proba)
        density = density_scorer(eras[era_name].X)
        hybrid_score = base - alpha * np.maximum(0.0, density)
        curve = evaluate_selective_policy(
            y_true=eras[era_name].y,
            proba_2d=proba,
            confidence_scores=hybrid_score,
            thresholds=thresholds,
            abstain_cost=abstain_cost,
        )
        curves_by_policy_era[hybrid_name][era_name] = curve

    selected_thresholds[hybrid_name] = choose_threshold_from_validation(
        curves_by_policy_era[hybrid_name]["val"], min_coverage=0.55
    )

    print("\n" + "=" * 78)
    print("Policy Frontier Summary (clean era)")
    print("=" * 78)
    for policy_name in curves_by_policy_era:
        clean_curve = curves_by_policy_era[policy_name]["clean"]
        aurc = area_under_risk_coverage(clean_curve.coverage, clean_curve.selective_risk)
        print(
            f"{policy_name:<20} AURC={aurc:.4f} "
            f"selected_threshold={selected_thresholds[policy_name]:.2f}"
        )

    print("\n" + "=" * 78)
    print("Fixed-threshold Robustness Across Eras")
    print("=" * 78)
    for policy_name in curves_by_policy_era:
        t = selected_thresholds[policy_name]
        print(f"\nPolicy: {policy_name} (threshold selected on validation: {t:.2f})")
        coverage_by_era: Dict[str, float] = {}
        risk_by_era: Dict[str, float] = {}
        for era_name in ["clean", "feature_shift", "prevalence_noise_shift"]:
            proba = predict_proba(model, eras[era_name].X)
            if policy_name == "density_adjusted_confidence":
                base = score_max_probability(proba)
                density = density_scorer(eras[era_name].X)
                conf = base - alpha * np.maximum(0.0, density)
            else:
                conf = base_policies[policy_name](proba)
            cov, risk, _ = summarize_fixed_threshold(
                era_name, eras[era_name].y, proba, conf, t
            )
            coverage_by_era[era_name] = cov
            risk_by_era[era_name] = risk
        csi = coverage_stability_index(coverage_by_era)
        ttr = threshold_transfer_regret(
            clean_risk=risk_by_era["clean"],
            shifted_risks=[
                risk_by_era["feature_shift"],
                risk_by_era["prevalence_noise_shift"],
            ],
            shifted_coverages=[
                coverage_by_era["feature_shift"],
                coverage_by_era["prevalence_noise_shift"],
            ],
        )
        cshock = coverage_shock_index(
            clean_coverage=coverage_by_era["clean"],
            shifted_coverages=[
                coverage_by_era["feature_shift"],
                coverage_by_era["prevalence_noise_shift"],
            ],
        )
        print(f"{'coverage_stability_index':<24} {csi:.3f}")
        print(f"{'threshold_transfer_regret':<24} {ttr:.3f}")
        print(f"{'coverage_shock_index':<24} {cshock:.3f}")

    plot_outputs(curves_by_policy_era, selected_thresholds)

    print("\nInterpretation notes:")
    print("- Lower AURC means lower average risk across coverage levels.")
    print("- Coverage stability index near 1.0 means threshold behavior transfers better under drift.")
    print("- Threshold transfer regret captures risk increase under shift at a fixed threshold.")
    print("- Coverage shock index captures relative coverage collapse under shift.")
    print("- A policy can look strong on clean data but become operationally brittle if coverage collapses under shift.")


if __name__ == "__main__":
    main()
