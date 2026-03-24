# Selective Prediction Under Shift

This subproject studies a deployment-relevant AI concept: **selective prediction** (also called abstention/reject option), where a model can refuse low-confidence predictions to reduce risk.

Instead of evaluating one static test split, it asks:

> *How stable is a confidence threshold when data distribution shifts?*

## Files

- `selective_prediction.py`
  - Builds a calibrated classifier on synthetic structured data.
  - Defines three confidence policies:
    - `max_probability`
    - `margin`
    - `negative_entropy`
  - Adds a hybrid policy:
    - `density_adjusted_confidence` (confidence penalized in low-density regions relative to training data)
  - Traces risk-coverage frontiers over confidence thresholds.
  - Chooses thresholds from validation by utility (accuracy under coverage minus abstention cost).
  - Evaluates those fixed thresholds across multiple eras:
    - `clean`
    - `feature_shift`
    - `prevalence_noise_shift`
  - Reports:
    - **AURC** (area under risk-coverage curve)
    - **Coverage Stability Index** (cross-era threshold robustness)
    - **Threshold Transfer Regret** (coverage-weighted risk increase vs clean era)
    - **Coverage Shock Index** (relative coverage drop under shift)
  - Saves:
    - `selective_prediction_risk_coverage.png`
    - `selective_prediction_coverage_stability.png`

- `requirements.txt` - Python dependencies
- `bash.sh` - quick environment setup helper

## Key Concepts Demonstrated

- **Selective prediction / abstention**: trade off error risk versus prediction coverage.
- **Risk-coverage analysis**: compare policies by frontier quality, not single-point accuracy.
- **Distribution-aware confidence**: confidence can be down-weighted for likely out-of-distribution regions.
- **Threshold transfer under shift**: fit threshold on validation, test whether it remains stable.
- **Operational utility framing**: threshold selected by deployment utility including abstention cost.
- **Coverage stability**: if coverage changes drastically across eras, operations can become brittle.
- **Threshold regret under shift**: fixed thresholds can silently increase downstream risk.

## Setup

```bash
python3 -m venv path/to/venv
source path/to/venv/bin/activate
pip install -r requirements.txt
```

## Run

From this directory:

```bash
python selective_prediction.py
```

This will:

- Print baseline (no-abstention) metrics across eras.
- Compute and compare policy frontiers with AURC.
- Select policy thresholds from validation utility.
- Evaluate fixed-threshold robustness under drift.
- Save two visualization files.

## Why This Is Distinct

Many online selective prediction examples stop at one confidence score and one test split. This subproject intentionally combines:

- **Multiple confidence policies** in one framework.
- A **density-adjusted hybrid policy** that blends confidence with local training-data support.
- **Cross-era threshold transfer checks** under explicit drift patterns.
- A dedicated **Coverage Stability Index** to capture operational reliability.
- Two additional shift diagnostics (**Threshold Transfer Regret** and **Coverage Shock Index**) to quantify fixed-threshold failure modes.
- **Utility-based thresholding** with abstention cost, not just risk minimization.

The result is a practical study of whether abstention strategies remain useful after data moves, not just whether they look good on a clean benchmark.
