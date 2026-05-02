# Shortcut Learning & Spurious Correlations (Tabular)

This subproject studies a core robust-AI failure mode: **shortcut learning**—when a model leans on a feature that is tightly aligned with the label on the training distribution but is **not stable** (here, it is deliberately broken at test time). The setup is **tabular and synthetic**, so we know exactly which dimension is the shortcut and can measure reliance without vision pipelines.

The motivating question:

> *Does strong in-distribution accuracy hide catastrophic failure when the cheap cue stops lining up with the label?*

## Files

- `shortcut_robustness.py`
  - Builds a binary classification task from informative core features plus an appended binary shortcut.
  - Keeps shortcut fidelity high on train / in-distribution (ID) test, then **ruptures** shortcut–label alignment out-of-distribution (OOD).
  - Trains **ERM logistic regression** on core + shortcut, **importance-weighted** logistic regression (inverse empirical frequency over `(y, shortcut)` cells), and an **oracle** that trains on core features only.
  - Reports **shortcut L2 share**, **shortcut-to-core tilt** (\(|w_s| / \|w_{\text{core}}\|\)), and tilt gap vs **LDA** on the same scaled design matrix as a linear baseline reference.
  - Computes **mitigation lift**: how much of the ERM→oracle OOD gap inverse-frequency weighting closes, **worst-case (y, shortcut) subgroup accuracy** on OOD, a **spurious rupture curve** (accuracy vs test-time shortcut fidelity), and a **regularization path** (shortcut emphasis and OOD accuracy vs `C`).
  - Saves:
    - `shortcut_robustness_id_vs_ood.png`
    - `shortcut_robustness_rupture_curve.png`
    - `shortcut_robustness_regularization_path.png`

- `requirements.txt` — Python dependencies
- `bash.sh` — quick environment setup helper

## Key Concepts Demonstrated

- **Shortcut / spurious correlation**: a feature that helps prediction under the training regime but should not be trusted when the world changes.
- **ID vs OOD shortcut fidelity**: same core geometry and labels; only the shortcut channel is stressed or broken.
- **Linear attribution**: coefficient share and tilt make reliance on the shortcut dimension inspectable (after scaling).
- **Mitigation without labels at test time**: joint inverse-frequency weighting upweights rare `(y, s)` combinations so the optimizer is less rewarded for betting everything on the majority shortcut pattern.
- **Subgroup robustness**: worst `(y, s)` cell accuracy surfaces failures that averages wash out.
- **Regularization trade-off**: stronger L2 can shift weight off the easy cue toward slower core geometry (when the shortcut dominates).

## Setup

```bash
python3 -m venv path/to/venv
source path/to/venv/bin/activate
pip install -r requirements.txt
```

## Run

From this directory:

```bash
python shortcut_robustness.py
```

This prints metric tables and writes the three PNG figures above.

## Why This Is Distinct

Typical shortcut demos emphasize deep networks on images. This module stays small and explicit: one known shortcut column, controlled rupture, and metrics aimed at **deployment thinking** (OOD gap, subgroup worst case, rupture sweeps, mitigation lift)—not a single accuracy number on one IID split.
