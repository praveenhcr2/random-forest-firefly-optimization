# ============================================================
# FIREFLY ALGORITHM FOR RANDOM FOREST OPTIMIZATION
# Comparison:
# 1. Default Random Forest
# 2. Grid Search
# 3. Firefly Algorithm
#
# Datasets:
# 1. Wine
# 2. Breast Cancer
# 3. Digits
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. GLOBAL CACHE
# ============================================================

fitness_cache = {}


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset(name):

    if name == "wine":
        data = load_wine()

    elif name == "breast_cancer":
        data = load_breast_cancer()

    elif name == "digits":
        data = load_digits()

    else:
        raise ValueError("Unknown dataset")

    X = data.data
    y = data.target

    return X, y


# ============================================================
# 4. FIREFLY FITNESS FUNCTION
# ============================================================
#
# Instead of performing cross-validation inside every
# Firefly evaluation, we directly calculate validation accuracy.
#
# This makes the algorithm much faster.
# ============================================================

def fitness(params, X_train, y_train, X_val, y_val):

    # Convert parameters to integers
    n_estimators = int(round(params[0]))
    max_depth = int(round(params[1]))
    min_samples_split = int(round(params[2]))
    min_samples_leaf = int(round(params[3]))

    # Create a unique key for caching
    key = (
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf
    )

    # Return cached result if already evaluated
    if key in fitness_cache:
        return fitness_cache[key]

    # Create Random Forest
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Predict validation data
    predictions = model.predict(X_val)

    # Calculate validation accuracy
    score = accuracy_score(y_val, predictions)

    # Store result
    fitness_cache[key] = score

    return score


# ============================================================
# 5. FIREFLY ALGORITHM
# ============================================================

def firefly_algorithm(
    X_train,
    y_train,
    X_val,
    y_val,
    n_fireflies=10,
    iterations=5,
    alpha=0.3,
    beta0=1,
    gamma=1
):

    # --------------------------------------------------------
    # Parameter ranges
    # --------------------------------------------------------

    lower_bounds = np.array([
        10,   # n_estimators
        1,    # max_depth
        2,    # min_samples_split
        1     # min_samples_leaf
    ])

    upper_bounds = np.array([
        200,  # n_estimators
        20,   # max_depth
        10,   # min_samples_split
        10    # min_samples_leaf
    ])


    # --------------------------------------------------------
    # Initialize random fireflies
    # --------------------------------------------------------

    fireflies = np.random.uniform(
        lower_bounds,
        upper_bounds,
        size=(n_fireflies, 4)
    )


    convergence = []


    # ========================================================
    # MAIN FIREFLY LOOP
    # ========================================================

    for iteration in range(iterations):

        # ----------------------------------------------------
        # Calculate brightness / fitness for all fireflies
        # ----------------------------------------------------

        scores = np.array([
            fitness(
                firefly,
                X_train,
                y_train,
                X_val,
                y_val
            )
            for firefly in fireflies
        ])


        # ----------------------------------------------------
        # Move fireflies toward brighter fireflies
        # ----------------------------------------------------

        for i in range(n_fireflies):

            for j in range(n_fireflies):

                # If firefly j is brighter than firefly i
                if scores[j] > scores[i]:

                    # Distance between fireflies
                    distance = np.linalg.norm(
                        fireflies[i] - fireflies[j]
                    )

                    # Attractiveness
                    beta = beta0 * np.exp(
                        -gamma * distance ** 2
                    )

                    # Random movement
                    random_step = alpha * (
                        np.random.rand(4) - 0.5
                    )

                    # Move firefly i toward j
                    new_position = (
                        fireflies[i]
                        + beta * (
                            fireflies[j] - fireflies[i]
                        )
                        + random_step
                    )

                    # Keep parameters within valid ranges
                    new_position = np.clip(
                        new_position,
                        lower_bounds,
                        upper_bounds
                    )

                    # Evaluate new position
                    new_score = fitness(
                        new_position,
                        X_train,
                        y_train,
                        X_val,
                        y_val
                    )

                    # Accept movement only if it improves fitness
                    if new_score > scores[i]:

                        fireflies[i] = new_position
                        scores[i] = new_score


        # ----------------------------------------------------
        # Store best score for convergence graph
        # ----------------------------------------------------

        best_score = np.max(scores)

        convergence.append(best_score)

        print(
            f"      Firefly Iteration "
            f"{iteration + 1}/{iterations} "
            f"- Best Validation Accuracy: "
            f"{best_score:.4f}"
        )


    # ========================================================
    # FIND BEST FIREFLY
    # ========================================================

    best_index = np.argmax(scores)

    best_params = fireflies[best_index]

    best_score = scores[best_index]


    return best_params, best_score, convergence


# ============================================================
# 6. RUN ONE EXPERIMENT
# ============================================================

def run_experiment(dataset_name, seed=0):

    global fitness_cache

    # --------------------------------------------------------
    # Set random seed
    # --------------------------------------------------------

    np.random.seed(seed)


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    X, y = load_dataset(dataset_name)


    # --------------------------------------------------------
    # Split data
    #
    # 70% Training
    # 15% Validation
    # 15% Testing
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp
    )


    # Clear Firefly cache
    fitness_cache = {}


    # ========================================================
    # BASELINE RANDOM FOREST
    # ========================================================

    start_time = time.time()


    baseline_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )


    baseline_model.fit(
        X_train,
        y_train
    )


    baseline_predictions = baseline_model.predict(
        X_test
    )


    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions
    )


    baseline_time = time.time() - start_time


    # ========================================================
    # GRID SEARCH
    # ========================================================

    start_time = time.time()


    param_grid = {

        "n_estimators": [50, 100, 150],

        "max_depth": [5, 10, 20],

        "min_samples_split": [2, 5],

        "min_samples_leaf": [1, 2]
    }


    grid_model = GridSearchCV(

        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),

        param_grid=param_grid,

        cv=3,

        scoring="accuracy",

        n_jobs=-1
    )


    grid_model.fit(
        X_train,
        y_train
    )


    grid_predictions = grid_model.predict(
        X_test
    )


    grid_accuracy = accuracy_score(
        y_test,
        grid_predictions
    )


    grid_time = time.time() - start_time


    # ========================================================
    # FIREFLY ALGORITHM
    # ========================================================

    start_time = time.time()


    best_firefly_params, firefly_validation_score, convergence = (
        firefly_algorithm(
            X_train,
            y_train,
            X_val,
            y_val,

            # Reduced settings for faster execution
            n_fireflies=10,
            iterations=5,

            alpha=0.3,
            beta0=1,
            gamma=1
        )
    )


    # Convert parameters to integers

    best_firefly_params = [

        int(round(best_firefly_params[0])),

        int(round(best_firefly_params[1])),

        int(round(best_firefly_params[2])),

        int(round(best_firefly_params[3]))
    ]


    # --------------------------------------------------------
    # Train final Firefly Random Forest
    # --------------------------------------------------------

    firefly_model = RandomForestClassifier(

        n_estimators=best_firefly_params[0],

        max_depth=best_firefly_params[1],

        min_samples_split=best_firefly_params[2],

        min_samples_leaf=best_firefly_params[3],

        random_state=42,

        n_jobs=-1
    )


    firefly_model.fit(
        X_train,
        y_train
    )


    firefly_predictions = firefly_model.predict(
        X_test
    )


    firefly_accuracy = accuracy_score(
        y_test,
        firefly_predictions
    )


    firefly_time = time.time() - start_time


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    firefly_cm = confusion_matrix(
        y_test,
        firefly_predictions
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "dataset": dataset_name,

        "seed": seed,

        "baseline": baseline_accuracy,

        "grid": grid_accuracy,

        "firefly": firefly_accuracy,

        "baseline_time": baseline_time,

        "grid_time": grid_time,

        "firefly_time": firefly_time,

        "firefly_params": best_firefly_params,

        "grid_params": grid_model.best_params_,

        "convergence": convergence,

        "confusion_matrix": firefly_cm
    }


# ============================================================
# 7. RUN MULTIPLE EXPERIMENTS
# ============================================================

def multi_run(dataset_name, n_runs=3):

    results = []


    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 70)


    for run in range(n_runs):

        print(
            f"\nRunning experiment "
            f"{run + 1}/{n_runs}"
        )

        result = run_experiment(
            dataset_name,
            seed=run
        )

        results.append(result)


        print(
            f"Baseline Accuracy : "
            f"{result['baseline'] * 100:.2f}%"
        )

        print(
            f"Grid Search Accuracy : "
            f"{result['grid'] * 100:.2f}%"
        )

        print(
            f"Firefly Accuracy : "
            f"{result['firefly'] * 100:.2f}%"
        )

        print(
            f"Firefly Parameters : "
            f"{result['firefly_params']}"
        )


    return results


# ============================================================
# 8. CALCULATE AVERAGE RESULTS
# ============================================================

def summarize_results(results):

    baseline_scores = [
        r["baseline"]
        for r in results
    ]

    grid_scores = [
        r["grid"]
        for r in results
    ]

    firefly_scores = [
        r["firefly"]
        for r in results
    ]


    baseline_times = [
        r["baseline_time"]
        for r in results
    ]

    grid_times = [
        r["grid_time"]
        for r in results
    ]

    firefly_times = [
        r["firefly_time"]
        for r in results
    ]


    print("\n" + "-" * 60)
    print("AVERAGE RESULTS")
    print("-" * 60)


    print(
        f"Baseline Accuracy : "
        f"{np.mean(baseline_scores) * 100:.2f}% "
        f"+/- {np.std(baseline_scores) * 100:.2f}%"
    )


    print(
        f"Grid Search Accuracy : "
        f"{np.mean(grid_scores) * 100:.2f}% "
        f"+/- {np.std(grid_scores) * 100:.2f}%"
    )


    print(
        f"Firefly Accuracy : "
        f"{np.mean(firefly_scores) * 100:.2f}% "
        f"+/- {np.std(firefly_scores) * 100:.2f}%"
    )


    print("\nExecution Time")


    print(
        f"Baseline : "
        f"{np.mean(baseline_times):.2f} seconds"
    )


    print(
        f"Grid Search : "
        f"{np.mean(grid_times):.2f} seconds"
    )


    print(
        f"Firefly : "
        f"{np.mean(firefly_times):.2f} seconds"
    )


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

datasets = [
    "wine",
    "breast_cancer",
    "digits"
]


all_results = {}


# ------------------------------------------------------------
# Run all datasets
# ------------------------------------------------------------

for dataset in datasets:

    results = multi_run(
        dataset,
        n_runs=3
    )

    all_results[dataset] = results

    summarize_results(results)


# ============================================================
# 10. COMBINED RESULTS TABLE
# ============================================================

print("\n\n")
print("=" * 80)
print("FINAL DATASET COMPARISON")
print("=" * 80)


print(
    f"{'Dataset':<20}"
    f"{'Baseline':<15}"
    f"{'Grid Search':<15}"
    f"{'Firefly':<15}"
)


print("-" * 65)


for dataset in datasets:

    results = all_results[dataset]


    baseline_mean = np.mean([
        r["baseline"]
        for r in results
    ]) * 100


    grid_mean = np.mean([
        r["grid"]
        for r in results
    ]) * 100


    firefly_mean = np.mean([
        r["firefly"]
        for r in results
    ]) * 100


    print(
        f"{dataset:<20}"
        f"{baseline_mean:<15.2f}"
        f"{grid_mean:<15.2f}"
        f"{firefly_mean:<15.2f}"
    )


# ============================================================
# 11. ACCURACY COMPARISON GRAPH
# ============================================================

dataset_labels = []

baseline_means = []
grid_means = []
firefly_means = []


for dataset in datasets:

    dataset_labels.append(
        dataset.replace("_", " ").title()
    )


    results = all_results[dataset]


    baseline_means.append(
        np.mean([
            r["baseline"]
            for r in results
        ]) * 100
    )


    grid_means.append(
        np.mean([
            r["grid"]
            for r in results
        ]) * 100
    )


    firefly_means.append(
        np.mean([
            r["firefly"]
            for r in results
        ]) * 100
    )


x = np.arange(len(datasets))

width = 0.25


plt.figure(figsize=(10, 6))


plt.bar(
    x - width,
    baseline_means,
    width,
    label="Default Random Forest"
)


plt.bar(
    x,
    grid_means,
    width,
    label="Grid Search"
)


plt.bar(
    x + width,
    firefly_means,
    width,
    label="Firefly Algorithm"
)


plt.xticks(
    x,
    dataset_labels
)


plt.ylabel("Accuracy (%)")

plt.xlabel("Dataset")

plt.title(
    "Accuracy Comparison Across Datasets"
)


plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)


plt.show()


# ============================================================
# 12. EXECUTION TIME COMPARISON
# ============================================================

baseline_times_mean = []
grid_times_mean = []
firefly_times_mean = []


for dataset in datasets:

    results = all_results[dataset]


    baseline_times_mean.append(
        np.mean([
            r["baseline_time"]
            for r in results
        ])
    )


    grid_times_mean.append(
        np.mean([
            r["grid_time"]
            for r in results
        ])
    )


    firefly_times_mean.append(
        np.mean([
            r["firefly_time"]
            for r in results
        ])
    )


x = np.arange(len(datasets))


plt.figure(figsize=(10, 6))


plt.bar(
    x - width,
    baseline_times_mean,
    width,
    label="Default Random Forest"
)


plt.bar(
    x,
    grid_times_mean,
    width,
    label="Grid Search"
)


plt.bar(
    x + width,
    firefly_times_mean,
    width,
    label="Firefly Algorithm"
)


plt.xticks(
    x,
    dataset_labels
)


plt.ylabel("Execution Time (seconds)")

plt.xlabel("Dataset")

plt.title(
    "Execution Time Comparison"
)


plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)


plt.show()


# ============================================================
# 13. FIREFLY CONVERGENCE GRAPHS
# ============================================================

for dataset in datasets:

    results = all_results[dataset]


    plt.figure(figsize=(8, 5))


    for i, result in enumerate(results):

        plt.plot(
            range(
                1,
                len(result["convergence"]) + 1
            ),

            result["convergence"],

            marker="o",

            label=f"Run {i + 1}"
        )


    plt.xlabel("Iteration")

    plt.ylabel("Validation Accuracy")

    plt.title(
        f"Firefly Algorithm Convergence - "
        f"{dataset.replace('_', ' ').title()}"
    )


    plt.legend()

    plt.grid(alpha=0.3)

    plt.show()


# ============================================================
# 14. BEST FIREFLY RUN + CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 80)
print("BEST FIREFLY RESULTS")
print("=" * 80)


for dataset in datasets:

    results = all_results[dataset]


    # Select actual best Firefly run
    best_result = max(
        results,
        key=lambda r: r["firefly"]
    )


    print(
        f"\nDataset: "
        f"{dataset.replace('_', ' ').title()}"
    )


    print(
        f"Best Firefly Test Accuracy: "
        f"{best_result['firefly'] * 100:.2f}%"
    )


    print(
        f"Best Firefly Parameters: "
        f"{best_result['firefly_params']}"
    )


    print(
        f"Grid Search Parameters: "
        f"{best_result['grid_params']}"
    )


    print("\nConfusion Matrix:")

    print(
        best_result["confusion_matrix"]
    )


    # --------------------------------------------------------
    # Display confusion matrix
    # --------------------------------------------------------

    disp = ConfusionMatrixDisplay(
        confusion_matrix=best_result["confusion_matrix"]
    )


    disp.plot()

    plt.title(
        f"Firefly Random Forest - "
        f"{dataset.replace('_', ' ').title()}"
    )


    plt.show()


# ============================================================
# 15. PROJECT COMPLETED
# ============================================================

print("\n")
print("=" * 80)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nDatasets tested:")
print("1. Wine")
print("2. Breast Cancer")
print("3. Digits")

print("\nMethods compared:")
print("1. Default Random Forest")
print("2. Grid Search")
print("3. Firefly Algorithm")

print("\nFirefly settings:")
print("Fireflies = 10")
print("Iterations = 5")
print("Runs per dataset = 3")