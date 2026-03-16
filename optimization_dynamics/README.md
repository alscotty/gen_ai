# Optimization Dynamics Study

This directory contains a **comparative study of optimization dynamics** for neural network training using TensorFlow/Keras. Instead of asking “which optimizer is best?”, this project focuses on **how different optimizers and learning rates behave over time** and what that means for real-world training decisions.

## Files

- **optimizer_dynamics.py**: Optimization Dynamics Comparison Study  
  - Compares four optimizers on the *same* network and dataset:
    - SGD with Nesterov momentum
    - RMSprop
    - Adam
    - AdamW-style optimizer with explicit weight decay
  - Sweeps a small grid of learning rates for each optimizer to expose:
    - Fast-but-unstable vs slow-but-stable regimes
    - When weight decay meaningfully changes training behavior
  - Uses a synthetic 3-class dataset with:
    - 12 features with 8 informative and 2 redundant dimensions
    - Modest label noise to make the boundary realistically messy
  - Tracks **training dynamics, not just final scalar metrics**:
    - Validation accuracy and loss curves
    - Train–validation generalization gap over time
    - Epochs required to reach a target validation accuracy (time-to-competence)
    - Area under the validation accuracy curve (AULC) as a measure of “learning efficiency”

## Key Concepts Demonstrated

- **Optimizer Choice as a Trade-off**  
  Shows that optimizers form a *Pareto frontier* between:
  - Convergence speed (time-to-competence)
  - Final validation performance
  - Stability / overfitting behavior (generalization gap)

- **Learning Efficiency (AULC)**  
  Introduces a simple but powerful metric:
  - AULC = area under the validation accuracy curve
  - Rewards configurations that become useful *early* rather than only at the final epoch

- **Time-to-Competence**  
  - Measures how many epochs each configuration needs to reach a target validation accuracy (e.g. 0.85)
  - Highlights configurations that are ideal for:
    - Rapid prototyping
    - Limited-budget training (few epochs allowed)
    - Interactive use cases

- **Generalization Gap Trajectories**  
  - Plots `(train_accuracy - val_accuracy)` across epochs
  - Makes it easy to see:
    - Optimizers that overshoot and overfit early
    - Configurations that stay well-regularized throughout training

## Visualizations Generated

Running `optimizer_dynamics.py` produces several plots:

- `optimizer_val_accuracy_curves.png`  
  - Validation accuracy curves for each optimizer
  - One subplot per optimizer, multiple lines per learning rate

- `optimizer_generalization_gap_curves.png`  
  - Generalization gap (train–val accuracy) over time
  - Reveals overfitting patterns specific to each optimizer/LR pair

- `optimizer_efficiency_scatter.png`  
  - Scatter plot of **AULC vs final validation accuracy**
  - Points labeled by optimizer and learning rate
  - Makes the speed–quality trade-off visually obvious

- `optimizer_time_to_competence_heatmap.png`  
  - Heatmap of epochs needed to reach a target validation accuracy
  - Axes: optimizer (rows) × learning rate (columns)
  - Cells encode “time-to-competence” in epochs (lower is better)

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

3. Run the study:

```bash
python optimizer_dynamics.py
```

## Unique Features (Not Standard in Online Tutorials)

Unlike typical optimizer tutorials that show a single training curve for one optimizer and one learning rate, this script:

- **Systematically compares multiple optimizers and learning rates** on a shared architecture and dataset.
- **Introduces learning-efficiency metrics**:
  - Area under the validation accuracy curve (AULC)
  - Time-to-competence for hitting a target accuracy threshold.
- **Analyzes overfitting as a trajectory**, not just as a final gap:
  - Generalization gap curves show *when* each configuration begins to overfit.
- **Separates “who wins at the end” from “who is useful quickly”**:
  - The optimizer that reaches the highest final accuracy is not always the one that delivers the most value per epoch.
- **Emphasizes decision-making for practitioners**:
  - Choose an optimizer based on:
    - Available training budget (epochs/steps)
    - Tolerance for early overfitting
    - Need for fast initial progress vs best final performance

This makes the study a **training-dynamics playground** rather than a one-off result, giving you intuition for how optimization choices play out over the entire training process.

