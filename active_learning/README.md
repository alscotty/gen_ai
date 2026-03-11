# Active Learning Label-Efficiency Study

This directory contains a **label-efficiency–focused active learning study** on a synthetic binary classification problem. It is not a generic "pool-based active learning" demo; instead, it is built to answer a concrete question:

> *“When does active learning actually buy you fewer labels than just labeling everything at random?”*

To make this question meaningful, the script simulates multiple query strategies on data with **informative, redundant, and spurious features**, and then quantifies *how quickly* each strategy approaches the performance of a fully supervised model.

## Files

- **active_learning_simulation.py** – Active Learning Label-Efficiency Comparison
  - Generates a synthetic dataset with:
    - Informative features
    - Redundant (correlated) features
    - Spurious features and a “noisy stripe” region where labels are less reliable
  - Trains a logistic regression model under different querying strategies:
    - **Random** – baseline that ignores model uncertainty
    - **Least-confidence** – queries points where the model is closest to 0.5 probability
    - **Margin** – uses the gap between the top two class probabilities as an uncertainty signal
    - **Diversity-aware uncertainty** – filters by uncertainty and then applies a simple k-center greedy step to avoid repeatedly querying near-duplicates
  - Runs a full active learning loop:
    - Starts from a small, stratified seed labeled set
    - Iteratively trains on the labeled pool and acquires new labels in batches
    - Tracks test performance after each batch
  - Computes **label-efficiency metrics** rarely included in short online examples:
    - **Area Under the Learning Curve (AULC)** for accuracy vs. number of labels
    - **Labels needed to reach a target fraction (e.g., 90%) of fully supervised accuracy**
    - Final performance comparison when the labeling budget is exhausted
  - Produces a plot:
    - `active_learning_label_efficiency.png` – accuracy vs labeled examples for all strategies
  - Prints a **label-efficiency summary table** and textual interpretation hints about when each strategy helps or saturates

## Key Concepts Demonstrated

- **Active Learning**: Iterative process of selecting which unlabeled points to annotate in order to maximize performance under a label budget.
- **Uncertainty Sampling**:
  - **Least-confidence**: Query where model is least certain (probability closest to 0.5).
  - **Margin-based**: Query where the gap between the top two class probabilities is smallest.
- **Diversity-Aware Selection**:
  - Uses a simple **k-center greedy** step on top of uncertainty filtering so that batches cover new regions rather than refocusing on near-duplicates.
- **Label-Efficiency Metrics**:
  - Instead of just plotting curves, the script explicitly quantifies:
    - Area under the accuracy curve vs labeled examples
    - Label count needed to reach a configurable fraction of the fully supervised accuracy
- **Noisy / Spurious Structure**:
  - Synthetic data includes a “stripe” region where labels are noisier and several spurious/redundant features, making uncertainty more informative in some regions than others.

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv path/to/venv
source path/to/venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

From this directory:

```bash
python active_learning_simulation.py
```

This will:

- Train a fully supervised baseline (all labels) and report accuracy, F1, and Brier score.
- Run active learning loops for:
  - Random
  - Least-confidence
  - Margin
  - Diversity-aware uncertainty
- Save the learning curve plot `active_learning_label_efficiency.png`.
- Print a **label-efficiency summary table** with:
  - AULC (area under learning curve)
  - Labels needed to reach a target fraction of the fully supervised accuracy
- Print interpretation hints on how to read the curves and when active learning is actually helping.

## Unique Aspects (Compared to Typical Online Tutorials)

Most quick active learning examples online:

- Implement only one uncertainty strategy in isolation.
- Show a qualitative curve but do not quantify how many labels are saved.
- Use clean toy data that does not stress-test uncertainty vs noise or redundancy.

This script intentionally goes further by:

- **Combining multiple query strategies** (random, least-confidence, margin, diversity-aware uncertainty) in a single reproducible experiment.
- **Quantifying label-efficiency explicitly** via:
  - Area under the learning curve (accuracy vs labeled examples).
  - Labels required to reach a target fraction of the fully supervised baseline accuracy.
- **Embedding spurious and redundant features**, plus a noisy region, so that:
  - Some uncertain regions are actually misleading.
  - Diversity-aware querying behaves differently from plain uncertainty.
- Providing **textual guidance** at the end that focuses on:
  - When active learning meaningfully reduces labels.
  - When random sampling is already competitive and active learning offers marginal gains.

This makes the project less about “how to code active learning” and more about **how to reason about label budgets and strategy choice** in practical settings.

