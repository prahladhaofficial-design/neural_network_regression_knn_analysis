# Neural Network Regression and KNN Analysis

A regression analysis project comparing neural-network training behavior with K-nearest-neighbor error sensitivity.

## Overview

Regression models can be compared from multiple perspectives: learned parametric functions and distance-based local prediction. This repository combines both through neural training curves and KNN error analysis.
The project includes data, script code, notebook analysis, generated figures, and derivation notes.

## Project Design

- Load a two-feature regression dataset.
- Compute Euclidean distances for KNN-style prediction.
- Measure mean squared error across different neighbor counts.
- Train a neural regression model and track loss over epochs.
- Compare the resulting plots and derivations.

## Key Capabilities

- Regression data loading from CSV.
- Sigmoid activation and derivative helpers.
- Euclidean-distance calculation.
- KNN prediction and MSE analysis.
- Neural training-loss visualization.
- Markdown derivations and notebook workflow.

## Repository Structure

```text
src/neural_regression.py           Script implementation.
data/regression_data.csv           Regression dataset.
notebooks/neural_regression.ipynb  Notebook workflow.
docs/                              Project brief and derivations.
figures/                           Generated plots.
```

## Running Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/neural_regression.py
jupyter notebook notebooks/neural_regression.ipynb
```

## Results and Artifacts

- `figures/mse vs neighbors(k).png` shows KNN error sensitivity.
- `figures/Training loss vs epochs.png` shows neural training behavior.
- `docs/technical-derivations.md` preserves mathematical notes.

## Reproducibility Notes

- Keep synthetic data generation parameters fixed when comparing neural regression and KNN behavior.
- Run notebooks from top to bottom so trained models and evaluation plots share the same split.
- Regenerate figures after changing hidden units, neighborhood size, or distance metrics.
- Compare smoothness, bias, and variance rather than only final numeric error.

## Detailed Workflow

- Run the script to regenerate the main figures.
- Open the notebook to inspect the step-by-step analysis.
- Review the KNN neighbor-count sweep before comparing training curves.
- Read the derivations to connect formulas with implementation.
- Use the figures folder as the quickest summary of current results.

## Implementation Details

- The script keeps core helper functions visible: sigmoid, derivative, distance calculation, and KNN prediction.
- The data file contains two input features and one continuous target.
- KNN analysis measures sensitivity to the number of neighbors.
- Neural training behavior is summarized through the training-loss curve.
- The notebook and script provide two ways to inspect the same modeling problem.

## Data and Inputs

- `data/regression_data.csv` is the primary dataset.
- The columns represent two input variables and a target value.
- KNN predictions depend directly on distance geometry in the feature space.
- Feature scaling may be important if the input variables have very different ranges.

## How To Interpret the Results

- The MSE-versus-neighbors plot shows how local averaging changes prediction error.
- Small neighbor counts can be noisy, while large counts can oversmooth predictions.
- The training-loss plot shows whether the neural model is converging.
- Comparing both views helps separate distance-based behavior from learned parametric behavior.

## Limitations and Assumptions

- The current script focuses on analysis rather than a reusable model package.
- Additional train-test evaluation would strengthen the comparison.
- KNN performance depends on feature scaling and distance choice.
- The neural model can be extended with validation metrics and hyperparameter sweeps.


## Review Guide

Use this checklist when reviewing the repository on GitHub:

- Start with the overview to understand the project objective before opening notebooks or source files.
- Read the repository structure section to identify the main entry point.
- Inspect generated outputs before rerunning experiments; they provide a quick view of the current project state.
- Review scripts and notebooks together when both are present, since notebooks often explain decisions while scripts preserve reusable execution paths.
- Check the limitations section before interpreting results too strongly.
- Use the possible extensions section to understand how the project can grow without changing its core purpose.

## Troubleshooting

Common issues and fixes:

- If imports fail, confirm that the virtual environment is activated and dependencies were installed from `requirements.txt`.
- If a notebook cell fails because a variable is missing, restart the kernel and run the notebook from the first cell.
- If a script cannot find a data file, run it from the repository root unless the README gives a different working directory.
- If plots are not regenerated, check that the output folder exists and that the script has write permission.
- If a package version causes errors, create a fresh environment and reinstall dependencies before debugging the source code.
- If GPU-related packages are unavailable, run CPU-compatible sections first and treat heavier deep-learning experiments separately.

## Development Notes

The repository is organized to be understandable before it is optimized. Most code favors clear structure, visible intermediate steps, and saved artifacts over highly compressed implementations. When extending the project, prefer small changes that keep the original experiment reproducible.

Recommended development practices:

- Keep new data files under `data/`, `datasets/`, or the existing local data folder for that project.
- Keep generated figures under `figures/`, `plots/`, `outputs/`, or `results/` depending on the existing convention.
- Add new experiments as separate scripts or clearly labeled notebook sections.
- Avoid overwriting existing outputs unless the experiment is intentionally being refreshed.
- Document any new dependency in `requirements.txt`.
- Prefer relative paths so the repository works after cloning on another machine.

## Project Status

This repository is prepared as a standalone GitHub project. The current version includes the core implementation, documentation, and available artifacts needed to understand the work. Future improvements can focus on stronger automation, more tests, richer visual summaries, and cleaner packaging where appropriate.

## Possible Extensions

- Add train-test splitting to compare generalization across methods.
- Add multiple distance metrics for KNN.
- Expose model parameters through command-line options.
- Add automated metric summaries.

## Tech Stack

Python, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Jupyter Notebook.

## Author

Prahladha, IIT Hyderabad
