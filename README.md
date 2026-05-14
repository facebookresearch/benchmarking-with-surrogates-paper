# About

This repository is a set of self-contained utils to to enable reproducible benchmarking in the following paper:
"How Surrogate Fidelity and Dataset Size Shape Benchmarking Conclusions: A Case Study in Ads Ranking" AutoML Conference, 2026.

It loads benchmark data from JSON files, trains surrogate models, and runs standardized comparisons using [Ax](https://ax.dev/)'s benchmarking framework.

## Installation

```bash
pip install ax-platform botorch scikit-learn numpy pandas torch
```

## Quick Start

```python
import numpy as np
from benchmark_data_loader import (
    load_benchmark_json,
    create_benchmark_test_function,
    cross_validate_surrogate,
    create_benchmark_problem,
    get_sobol_method,
    get_bo_method,
    SURROGATE_MODELS,
)
from ax.benchmark.benchmark import benchmark_multiple_problems_methods

# 1. Load data
data = load_benchmark_json("path/to/dataset.json")
print(f"Loaded {len(data['trials'])} trials with {len(data['search_space'])} parameters")

# 2. Train surrogates, cross-validate, and create benchmark problems
benchmark_problems = []
for name, factory in SURROGATE_MODELS.items():
    test_fn = create_benchmark_test_function(data, surrogate=factory())
    cv = cross_validate_surrogate(data, surrogate=factory())
    print(f"{name}: R²={cv['prediction/r-squared']:.4f}, MAPE={cv['prediction/mape']:.4f}")

    problem = create_benchmark_problem(data, test_function=test_fn, num_trials=20)
    benchmark_problems.append(problem)

# 3. Run benchmarking
results = benchmark_multiple_problems_methods(
    problems=benchmark_problems,
    methods=[get_sobol_method(), get_bo_method(num_sobol_trials=5)],
    seeds=range(3),
)

# 4. Plot results
from ax.plot.trace import optimization_trace_all_methods

optimization_trace_all_methods(
    y_dict={
        r.name: np.stack([res.optimization_trace for res in r.results])
        for r in results
    },
    title="Benchmark Results",
    ylabel="Optimization trace",
)
```

## API Reference

### Data Loading

**`load_benchmark_json(source)`** — Load and validate a benchmark JSON file or string. Returns a dict with `search_space`, `trials`, and optionally `minimize`.

### Surrogate Models

**`create_benchmark_test_function(data, metric_name="metric", surrogate=None)`** — Train a surrogate model on the trial data. Returns a `BenchmarkTestFunction` that can evaluate new parameter configurations. Defaults to `LinearRegression()` if no surrogate is provided.

**`cross_validate_surrogate(data, metric_name="metric", surrogate=None, n_splits=5)`** — Cross-validate a surrogate model using GroupKFold. Returns a `pd.Series` with R-squared, MAPE, and sample count.

**`SURROGATE_MODELS`** — Built-in surrogate model factories:

| Key | Model |
|-----|-------|
| `"gradient_boost"` | `GradientBoostingRegressor(n_estimators=100)` |
| `"linear_regression"` | `LinearRegression()` |

### Benchmark Problem Creation

**`create_benchmark_problem(data, test_function, num_trials=20, metric_name="metric", noise_std=0.0)`** — Create an Ax `BenchmarkProblem` from JSON data and a trained surrogate.

**`build_benchmark_problem(search_space, test_function, metric_name, minimize, optimal_value, baseline_value, ...)`** — Lower-level builder for creating a `BenchmarkProblem` from pre-built components. Used when integrating with existing pipelines.

### Benchmark Methods

**`get_sobol_method()`** — Quasi-random Sobol search (baseline).

**`get_bo_method(num_sobol_trials=5)`** — Bayesian optimization with SingleTaskGP and qLogNoisyExpectedImprovement.

**`get_default_bo_method(num_sobol_trials=5)`** — Bayesian optimization using Ax's auto-dispatch with fully Bayesian SAAS and input warping. Recommended for high-dimensional problems.

## How It Works

1. **Data Loading**: JSON files containing historical hyperparameter trials and their objective values are loaded and validated.

2. **Surrogate Training**: An sklearn regressor (e.g., Gradient Boosting) is trained on the historical data. Categorical parameters are one-hot encoded, and missing features are handled automatically.

3. **Cross-Validation**: GroupKFold cross-validation evaluates surrogate quality. Trials with identical parameter configurations are grouped together to prevent data leakage.

4. **Benchmark Problem**: The trained surrogate is wrapped as an Ax `BenchmarkTestFunction`, combined with the search space and optimization config into a `BenchmarkProblem`.

5. **Benchmarking**: Ax's `benchmark_multiple_problems_methods` runs optimization methods (Sobol, BO) against the surrogate across multiple random seeds, producing `AggregatedBenchmarkResult` objects with optimization traces.

## License

This project is licensed under CC-BY-NC 4.0, as found in the LICENSE file. See [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for details.
