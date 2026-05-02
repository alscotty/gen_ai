#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shortcut Learning & Spurious Correlations (Tabular)

Studies a core robust-AI failure mode: models latch onto a feature that is
strongly aligned with the label *on the training distribution* but becomes
uninformative or misleading under shift at deployment time.

This script is not a vision CNN demo. It uses controlled synthetic tabular
data so we can measure reliance on a single known shortcut dimension and
compare mitigations without image pipelines.

Distinctive analyses bundled here:
  - Shortcut-to-core tilt vs an LDA reference on the same scaled features:
    ratio |w_shortcut| / ||w_core|| so logistic and LDA are comparable.
  - Spurious rupture curve: smoothly interpolates test-time shortcut fidelity
    between train-like correlation and broken / random shortcut regimes.
  - Importance-weighted debiasing: empirical inverse-frequency weights over
    (y, shortcut) cells on the training set to reduce incentive to hinge on s.
  - Shortcut coefficient trajectory vs inverse L2 strength (C): shows how
    stronger regularization reallocates weight from the cheap feature toward
    slower-but-stable core geometry (when the shortcut is easy).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Use a project-local config dir so matplotlib can cache fonts when ~/.matplotlib
# is not writable (CI, sandboxes, some dev environments).
_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mplconfig"))
(_ROOT / ".mplconfig").mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.style.use("ggplot")
OUTPUT_DIR = Path(__file__).resolve().parent


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_id: np.ndarray
    y_id: np.ndarray
    X_core_id: np.ndarray
    X_ood: np.ndarray
    y_ood: np.ndarray
    X_core_ood: np.ndarray


def sample_shortcut(y: np.ndarray, p_same: float, rng: np.random.Generator) -> np.ndarray:
    """
    Binary shortcut s in {0,1}. With probability p_same, s matches y; otherwise flips.
    p_same=0.5 yields maximum rupture (shortcut carries no marginal signal about y
    when marginally averaged — still structured coupling conditional on y).
    """
    flip = rng.random(len(y)) > p_same
    s = np.where(flip, 1 - y, y).astype(np.int32)
    return s


def build_design_matrix(X_core: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.column_stack([X_core, s.astype(float)])


def make_splits(
    n_samples: int = 9000,
    n_features: int = 10,
    shortcut_train_align: float = 0.92,
    shortcut_id_align: float = 0.91,
    shortcut_ood_align: float = 0.5,
) -> Tuple[SplitData, Dict[str, float]]:
    """
    Core labels come from informative geometry; shortcut is appended.

    - Train / val / ID test: shortcut is deliberately aligned with y (easy cue).
    - OOD test: shortcut decouples from y (broken cue at typical probability 0.5).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    from sklearn.datasets import make_classification

    X_full, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=7,
        n_redundant=1,
        n_clusters_per_class=2,
        class_sep=1.05,
        flip_y=0.02,
        random_state=RANDOM_STATE,
    )

    X_train, X_rest, y_train, y_rest = train_test_split(
        X_full, y, test_size=0.45, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_tmp, y_val, y_tmp = train_test_split(
        X_rest, y_rest, test_size=0.55, stratify=y_rest, random_state=RANDOM_STATE + 1
    )
    X_id, X_ood, y_id, y_ood = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=RANDOM_STATE + 2
    )

    s_train = sample_shortcut(y_train, shortcut_train_align, rng)
    s_val = sample_shortcut(y_val, shortcut_train_align, rng)
    s_id = sample_shortcut(y_id, shortcut_id_align, rng)
    s_ood = sample_shortcut(y_ood, shortcut_ood_align, rng)

    split = SplitData(
        X_train=build_design_matrix(X_train, s_train),
        y_train=y_train,
        X_val=build_design_matrix(X_val, s_val),
        y_val=y_val,
        X_id=build_design_matrix(X_id, s_id),
        y_id=y_id,
        X_core_id=X_id,
        X_ood=build_design_matrix(X_ood, s_ood),
        y_ood=y_ood,
        X_core_ood=X_ood,
    )
    meta = {
        "shortcut_train_align": shortcut_train_align,
        "shortcut_id_align": shortcut_id_align,
        "shortcut_ood_align": shortcut_ood_align,
    }
    return split, meta


def shortcut_weight_share(coef: np.ndarray) -> float:
    """Magnitude share on the last (shortcut) coordinate after scaling."""
    v = np.asarray(coef).ravel()
    if v.size == 0:
        return 0.0
    return float(np.abs(v[-1]) / (np.linalg.norm(v) + 1e-12))


def shortcut_core_tilt(coef: np.ndarray) -> float:
    """|w_s| / ||w_core|| — comparable emphasis ratio across linear heads."""
    v = np.asarray(coef).ravel()
    if v.size < 2:
        return 0.0
    core = v[:-1]
    return float(np.abs(v[-1]) / (np.linalg.norm(core) + 1e-12))


def fit_lda_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    return lda.coef_.ravel()


def compute_inverse_joint_weights(
    y: np.ndarray, s: np.ndarray
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float], Dict[Tuple[int, int], int]]:
    """Inverse empirical frequency weights for (y, s) pairs on the train set."""
    s = s.astype(int)
    counts: Dict[Tuple[int, int], int] = {}
    for yy, ss in zip(y, s):
        counts[(int(yy), int(ss))] = counts.get((int(yy), int(ss)), 0) + 1
    total = len(y)
    inv_p: Dict[Tuple[int, int], float] = {}
    for k, c in counts.items():
        inv_p[k] = total / (c + 1e-12)
    w = np.array([inv_p[(int(yy), int(ss))] for yy, ss in zip(y, s)], dtype=float)
    w *= len(w) / np.sum(w)
    return w, inv_p, counts


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=C,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X, y, clf__sample_weight=sample_weight)
    return pipe


def accuracy(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> float:
    return float(accuracy_score(y, pipe.predict(X)))


def worst_group_accuracy(pipe: Pipeline, X_aug: np.ndarray, y: np.ndarray) -> float:
    """Worst-case accuracy over (y, shortcut) cells — surfaces hidden subgroup failure."""
    s = X_aug[:, -1].astype(int)
    pred = pipe.predict(X_aug)
    accs: List[float] = []
    for yy in (0, 1):
        for ss in (0, 1):
            mask = (y == yy) & (s == ss)
            if np.any(mask):
                accs.append(accuracy_score(y[mask], pred[mask]))
    return float(min(accs)) if accs else 0.0


def rupture_curve_accuracies(
    pipe: Pipeline,
    X_core: np.ndarray,
    y: np.ndarray,
    p_grid: np.ndarray,
) -> np.ndarray:
    """Re-evaluate a fixed trained model under varying test-time shortcut fidelity."""
    rng = np.random.default_rng(RANDOM_STATE + 7)
    out = np.zeros_like(p_grid, dtype=float)
    for i, p in enumerate(p_grid):
        s_new = sample_shortcut(y, float(p), rng)
        X_t = build_design_matrix(X_core, s_new)
        out[i] = accuracy_score(y, pipe.predict(X_t))
    return out


def main() -> None:
    split, meta = make_splits()
    X_tr, y_tr = split.X_train, split.y_train
    s_tr = X_tr[:, -1].astype(int)

    # LDA reference on same augmented design (Gaussian baseline split).
    scaler_lda = StandardScaler()
    X_train_scaled = scaler_lda.fit_transform(X_tr)
    lda_coef = fit_lda_coef(X_train_scaled, y_tr)
    lda_share = shortcut_weight_share(lda_coef)
    lda_tilt = shortcut_core_tilt(lda_coef)

    # Inverse-frequency weights for (y, s).
    iw, _, cell_counts = compute_inverse_joint_weights(y_tr, s_tr)

    erm = fit_logistic(X_tr, y_tr, C=1.0)
    weighted = fit_logistic(X_tr, y_tr, C=1.0, sample_weight=iw)
    oracle_core = fit_logistic(split.X_train[:, :-1], y_tr, C=1.0)

    erm_share = shortcut_weight_share(erm.named_steps["clf"].coef_)
    w_share = shortcut_weight_share(weighted.named_steps["clf"].coef_)
    erm_tilt = shortcut_core_tilt(erm.named_steps["clf"].coef_)
    w_tilt = shortcut_core_tilt(weighted.named_steps["clf"].coef_)

    id_acc_erm = accuracy(erm, split.X_id, split.y_id)
    ood_acc_erm = accuracy(erm, split.X_ood, split.y_ood)
    id_acc_w = accuracy(weighted, split.X_id, split.y_id)
    ood_acc_w = accuracy(weighted, split.X_ood, split.y_ood)
    ood_acc_oracle = accuracy(oracle_core, split.X_core_ood, split.y_ood)

    gap_erm = id_acc_erm - ood_acc_erm
    gap_w = id_acc_w - ood_acc_w
    mitigation_lift = (ood_acc_w - ood_acc_erm) / max(ood_acc_oracle - ood_acc_erm, 1e-6)

    print("Shortcut robustness study (tabular synthetic)")
    print(f"  Train shortcut P(s=y): {meta['shortcut_train_align']:.2f}")
    print(f"  ID test shortcut P(s=y): {meta['shortcut_id_align']:.2f}")
    print(f"  OOD test shortcut P(s=y): {meta['shortcut_ood_align']:.2f} (ruptured)")
    print()
    print(f"LDA reference — L2 share on shortcut: {lda_share:.4f}, tilt |ws|/||w_core||: {lda_tilt:.4f}")
    print(
        f"ERM — L2 share: {erm_share:.4f}, tilt: {erm_tilt:.4f}  "
        f"(tilt gap vs LDA: {erm_tilt - lda_tilt:+.4f})"
    )
    print(
        f"IW — L2 share: {w_share:.4f}, tilt: {w_tilt:.4f}  "
        f"(tilt gap vs LDA: {w_tilt - lda_tilt:+.4f})"
    )
    print()
    print(f"ERM   ID acc={id_acc_erm:.3f}  OOD acc={ood_acc_erm:.3f}  gap={gap_erm:.3f}")
    print(f"IW    ID acc={id_acc_w:.3f}  OOD acc={ood_acc_w:.3f}  gap={gap_w:.3f}")
    print(f"Oracle (no shortcut at train) OOD acc={ood_acc_oracle:.3f}")
    print(f"Mitigation lift (fraction of oracle gap closed by IW): {mitigation_lift:.3f}")
    print()

    wg_erm = worst_group_accuracy(erm, split.X_ood, split.y_ood)
    wg_w = worst_group_accuracy(weighted, split.X_ood, split.y_ood)
    print(f"Worst (y,s) subgroup accuracy on OOD: ERM={wg_erm:.3f}, IW={wg_w:.3f}")

    # Regularization path: C vs shortcut share (ERM).
    c_values = np.logspace(-2, 2, num=18)
    shares: List[float] = []
    ood_accs: List[float] = []
    for C in c_values:
        m = fit_logistic(X_tr, y_tr, C=float(C))
        shares.append(shortcut_weight_share(m.named_steps["clf"].coef_))
        ood_accs.append(accuracy(m, split.X_ood, split.y_ood))

    # Rupture sweep on OOD core (labels fixed).
    p_grid = np.linspace(0.5, meta["shortcut_train_align"], num=14)

    rupture_erm = rupture_curve_accuracies(erm, split.X_core_ood, split.y_ood, p_grid)
    rupture_w = rupture_curve_accuracies(weighted, split.X_core_ood, split.y_ood, p_grid)

    # --- Plots ---
    fig1, ax1 = plt.subplots(figsize=(8.5, 4.5))
    labels = ["ERM", "IW debias", "Oracle (core only)"]
    id_scores = [id_acc_erm, id_acc_w, accuracy(oracle_core, split.X_core_id, split.y_id)]
    ood_scores = [ood_acc_erm, ood_acc_w, ood_acc_oracle]
    x = np.arange(len(labels))
    wbar = 0.35
    ax1.bar(x - wbar / 2, id_scores, width=wbar, label="ID (shortcut intact)", color="#4c72b0")
    ax1.bar(x + wbar / 2, ood_scores, width=wbar, label="OOD (shortcut ruptured)", color="#dd8452")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0.45, 1.0)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Spurious shortcut: ID accuracy can hide OOD failure")
    ax1.legend(loc="lower left")
    fig1.tight_layout()
    p1 = OUTPUT_DIR / "shortcut_robustness_id_vs_ood.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.5))
    ax2.plot(p_grid, rupture_erm, marker="o", label="ERM", color="#4c72b0")
    ax2.plot(p_grid, rupture_w, marker="s", label="IW debias", color="#55a868")
    ax2.axhline(ood_acc_oracle, color="#8172b3", linestyle="--", label="Oracle OOD")
    ax2.set_xlabel(r"Test-time shortcut fidelity $P(s=y)$")
    ax2.set_ylabel("Accuracy (same core labels, resampled shortcut)")
    ax2.set_title("Spurious rupture curve: stress-test shortcut fidelity")
    ax2.legend()
    fig2.tight_layout()
    p2 = OUTPUT_DIR / "shortcut_robustness_rupture_curve.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    ax3a.plot(np.log10(c_values), shares, color="#c44e52", marker="o")
    ax3a.set_ylabel("Shortcut |coef| / ||coef||")
    ax3a.set_title("Regularization path: reallocating weight off the cheap cue")
    ax3b.plot(np.log10(c_values), ood_accs, color="#64b5cd", marker="o")
    ax3b.set_xlabel(r"$\log_{10}$(inverse L2 strength C)")
    ax3b.set_ylabel("OOD accuracy")
    fig3.tight_layout()
    p3 = OUTPUT_DIR / "shortcut_robustness_regularization_path.png"
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)

    print(f"Saved: {p1.name}, {p2.name}, {p3.name}")
    print("(Inverse joint counts for (y,s) on train — diagnosable rare cells)")
    for k in sorted(cell_counts.keys()):
        print(f"  {k}: n={cell_counts[k]}")


if __name__ == "__main__":
    main()
