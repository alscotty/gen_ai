# Model Evaluation Under Drift and Asymmetric Costs

This directory contains a focused, implementation-oriented study of **model evaluation as a decision-making tool**, not just a list of metrics. It explores how model quality changes under **dataset drift** and how **asymmetric misclassification costs** change which model and threshold you should actually deploy.

Unlike standard tutorials that stop at accuracy/F1 on a single test set, this mini-project:

- Evaluates multiple metrics (discrimination, calibration, and probability quality)
- Simulates three deployment eras with different kinds of drift
- Uses **decision curve analysis** to connect metrics to real-world cost trade-offs

The goal is to practice thinking like a production ML engineer: *“Given shifting data and business costs, what should I actually ship?”*

## Files

- **model_evaluation.py**: End-to-end evaluation workflow
  - Generates three synthetic eras:
    - **Era A (base)**: Original prevalence, clean features
    - **Era B (prevalence shift)**: Higher positive rate, same feature distribution
    - **Era C (feature + noise shift)**: Feature shift plus extra noise
  - Trains models on Era A, then evaluates on all eras to simulate deployment drift
  - Compares a **Logistic Regression** and a **Calibrated Random Forest**
  - Computes a rich metric set at a fixed threshold:
    - Discrimination: accuracy, precision, recall, F1, ROC AUC, PR AUC
    - Calibration / probability quality: Brier score
  - Produces several plots:
    - `model_evaluation_roc_pr.png`: ROC and Precision-Recall curves (Era A)
    - `model_evaluation_calibration.png`: Reliability diagrams (Era A)
    - `model_evaluation_decision_curves.png`: Expected cost vs threshold for each model
    - `model_evaluation_drift_sensitivity.png`: How metrics move across eras
  - Highlights how:
    - The “best” model depends on which metric you care about
    - Metric rankings can change under drift
    - Deployment thresholds should be chosen from **cost structure**, not arbitrarily

## Key Concepts Demonstrated

- **Multi-metric evaluation**:
  - Why you should look at ROC AUC, PR AUC, and calibration (Brier score) together
  - How a model with slightly worse ROC AUC can be better calibrated and easier to deploy
- **Dataset drift sensitivity**:
  - How prevalence shift (class imbalance over time) changes accuracy and PR AUC
  - How feature + noise shift can hurt calibration more than ROC AUC
  - How to visualize metric stability across eras
- **Asymmetric cost-aware decisions**:
  - Simple decision curve analysis where false negatives are more expensive than false positives
  - How the *optimal* threshold differs per model once you account for costs
  - Why “0.5 by default” is usually the wrong decision threshold
- **Deployment mindset**:
  - Training on one era and evaluating on several realistic futures
  - Reading plots as if you’re writing a launch note: “Given this drift and these costs, we prefer Model X at threshold Y.”

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

3. Run the evaluation:

```bash
python model_evaluation.py
```

## Output

The script will:

- Print a compact metrics table with:
  - Rows: `(model, era)`
  - Columns: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`, `brier`
- Save four PNG figures:
  - **Discrimination**: ROC and PR curves on the base era
  - **Calibration**: Reliability diagrams showing over/under-confidence
  - **Decision curves**: Expected cost vs threshold for each model, under asymmetric costs
  - **Drift sensitivity**: How all metrics move across Era A/B/C for each model
- Print a few prompts suggesting what to look for when manually inspecting the plots.

## Unique Aspects (Compared to Typical Online Tutorials)

Most online model evaluation examples either:

- Show a single score (accuracy/F1) on a single test split, or
- Show ROC/PR curves without connecting them to deployment decisions.

This mini-project is different because it:

- **Ties evaluation directly to decisions** using an explicit cost model and decision curves
- **Models drift explicitly** (prevalence shift + feature/noise shift) and shows metric stability
- **Contrasts discrimination vs calibration** rather than treating “higher AUC = always better”
- **Keeps everything synthetic and lightweight** so you can rerun and modify it quickly

It is designed as a compact, reproducible “evaluation sandbox” you can adapt when you need to reason about real-world deployment trade-offs, rather than as a one-off metric demo.

