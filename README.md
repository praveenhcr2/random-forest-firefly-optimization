# Random Forest Hyperparameter Optimization using Firefly Algorithm

## Overview

This project focuses on optimizing Random Forest hyperparameters using the Firefly Algorithm and comparing its performance with a default Random Forest model and Grid Search.

The approach is evaluated on three classification datasets: Wine, Breast Cancer, and Digits.

The project compares the three approaches based on classification accuracy and execution time.

## Objectives

- Optimize Random Forest hyperparameters using the Firefly Algorithm.
- Compare Firefly-based optimization with Default Random Forest and Grid Search.
- Evaluate the approaches across multiple classification datasets.
- Analyze model accuracy and optimization execution time.

## Methods

### 1. Default Random Forest

A Random Forest classifier with default hyperparameters is trained and evaluated as the baseline model.

### 2. Grid Search

Grid Search evaluates predefined combinations of Random Forest hyperparameters using 3-fold cross-validation to identify a suitable configuration.

### 3. Firefly Algorithm

The Firefly Algorithm is used as a population-based optimization method to search for suitable Random Forest hyperparameters.

Each firefly represents a possible combination of Random Forest parameters. The brightness of a firefly is determined by the validation accuracy of the corresponding Random Forest model.

The Firefly Algorithm searches for better parameter combinations by moving fireflies toward brighter solutions.

## Hyperparameters Optimized

The Firefly Algorithm searches for values of:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

## Datasets

The project uses three datasets available through Scikit-learn:

- **Wine** — multi-class classification
- **Breast Cancer** — binary classification
- **Digits** — handwritten digit classification

Each dataset is divided into:

- 70% training data
- 15% validation data
- 15% test data

The experiments are repeated for 3 runs using different random seeds.

## Experimental Results

### Average Accuracy

| Dataset | Default Random Forest | Grid Search | Firefly |
|---|---:|---:|---:|
| Wine | 100.00% | 100.00% | 98.77% |
| Breast Cancer | 95.74% | 95.35% | **96.51%** |
| Digits | **97.04%** | 96.67% | 96.67% |

### Average Execution Time

| Dataset | Default Random Forest | Grid Search | Firefly |
|---|---:|---:|---:|
| Wine | 0.23 s | 21.17 s | **2.82 s** |
| Breast Cancer | 0.34 s | 24.38 s | **4.68 s** |
| Digits | 0.42 s | 34.08 s | **7.14 s** |

The Firefly approach completed the optimization considerably faster than Grid Search across all three datasets.

On the Breast Cancer dataset, Firefly achieved an average accuracy of **96.51%**, compared with **95.35%** for Grid Search, while taking **4.68 seconds** compared with **24.38 seconds** for Grid Search.

## Firefly Configuration

The experiments use:

- Fireflies: **10**
- Iterations: **5**
- Experimental runs per dataset: **3**

## Technologies Used

- Python
- Scikit-learn
- NumPy
- Matplotlib
