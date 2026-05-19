# ml-from-scratch

A machine learning learning project where core algorithms are implemented from scratch using only Python and NumPy.

## Overview

This repository is focused on understanding machine learning fundamentals by building the underlying math and logic step by step instead of relying on high-level ML libraries such as Scikit-learn. The long-term goal is to use these implementations as the foundation for a later credit risk modeling project. 

## Why this project

The purpose of this project is to build a strong understanding of how machine learning algorithms actually work internally. Writing the algorithms from scratch makes concepts like matrix operations, probability, optimisation, and model training much easier to understand and explain.

## What this repository includes

This repository contains both small foundation scripts and larger algorithm-building blocks.

Current focus areas include:

- Matrix operations with NumPy
- Sigmoid function and its derivative
- Probability utilities such as joint probability and Bayes' theorem
- Gradient descent and loss functions
- Logistic regression from scratch
- Decision tree fundamentals

## Repository structure

```text
ml-from-scratch/
├── foundations/
│   ├── matrix_ops.py
│   ├── sigmoid.py
│   ├── gradient_descent.py
│   ├── loss_functions.py
│   ├── logistic_regression.py
│   └── decision_tree_basic.py
├── utils/
│   ├── metrics.py
│   └── visualise.py
└── notebooks/
```

- `foundations/` - core mathematical and algorithm implementations:
    - `matrix_ops.py`
    - `sigmoid.py`
    - `gradient_descent.py`
    - `loss_functions.py`
    - `logistic_regression.py`
    - `decision_tree_basic.py`
- `utils/` - helper utilities such as metrics and visualisation functions:
    - `metrics.py`
    - `visualise.py`
- `notebooks/` - optional experiments, notes, and concept exploration

## Project approach

Each concept is implemented as a small, focused script so the learning process stays visible and easy to follow. The project starts with mathematical and programming foundations, then gradually builds toward complete machine learning algorithms.

## Current status

This is an active learning project and the repository is being updated incrementally. Some files are intentionally simple at first and will be refined over time as the implementations become cleaner, more reusable, and better documented.

## Future direction

The main goal is to finish building logistic regression and decision tree implementations from scratch and reuse them in a separate credit risk model project. This repository acts as the foundation layer for that larger applied ML project.

## Tech stack

- Python
- NumPy
- Matplotlib

## Note

This repository is primarily for learning and depth of understanding. The emphasis is on clarity, implementation from first principles, and steady improvement over time.

## experiment
Learning Rate Experiment

Goal:
To study how different learning rates affect gradient descent on the same linear regression problem.

Functions used:

    compute_descent_step()

    train_linear_model()

    plot_loss_history()

Method:
I trained the same model with three learning rates: 0.001, 0.01, and 0.1.
The dataset, model, and number of epochs stayed the same, so only the learning rate changed.

Results:

    0.001: loss decreased slowly, so convergence was slow.

    0.01: loss dropped faster and reached a much better value.

    0.1: loss exploded, so the model diverged.

Important terms:

    Convergence: loss decreases and approaches a stable minimum.

    Divergence: loss grows instead of shrinking.

    Overshooting minima: the update step is too large and jumps past the best point.

    Learning rate: the step size used to update weights.

    Epoch: one full pass of the training loop.

Interpretation:
A small learning rate is stable but slow.
A medium learning rate is often the best tradeoff because it learns fast without becoming unstable.
A large learning rate can make the model unstable because the weights change too much each step.

Conclusion:
This experiment shows the core tradeoff in gradient descent: smaller learning rates are safer, larger learning rates are faster but risky, and a middle value often gives the best training behavior
