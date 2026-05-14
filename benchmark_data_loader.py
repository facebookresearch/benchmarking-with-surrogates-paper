# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Self-contained loader for benchmark JSON data files.

Loads benchmark JSON files and provides a complete pipeline for creating
Ax BenchmarkProblems — with no dependencies on GAIN internals.

Example usage::

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

    data = load_benchmark_json("path/to/dataset.json")
    for name, factory in SURROGATE_MODELS.items():
        test_fn = create_benchmark_test_function(data, surrogate=factory())
        cv_results = cross_validate_surrogate(data, surrogate=factory())
        problem = create_benchmark_problem(data, test_function=test_fn)

    results = benchmark_multiple_problems_methods(
        problems=[problem],
        methods=[get_sobol_method(), get_bo_method()],
        seeds=range(3),
    )
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ax.api.utils.generation_strategy_dispatch import choose_generation_strategy
from ax.api.utils.structs import GenerationStrategyDispatchStruct
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_metric import BenchmarkMetric
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.benchmark.methods.modular_botorch import get_sobol_botorch_modular_acquisition
from ax.benchmark.methods.sobol import get_sobol_benchmark_method
from ax.core.objective import Objective as AxObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace
from ax.core.types import TParamValue
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models import SingleTaskGP
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupKFold
from torch import Tensor

logger: logging.Logger = logging.getLogger(__name__)

METRIC_NAME: str = "metric"

PARAMETER_TYPE_MAP: dict[str, ParameterType] = {
    "float": ParameterType.FLOAT,
    "int": ParameterType.INT,
    "str": ParameterType.STRING,
    "bool": ParameterType.BOOL,
}

SURROGATE_MODELS: dict[str, Any] = {
    "gradient_boost": lambda: GradientBoostingRegressor(
        n_estimators=100, random_state=42
    ),
    "linear_regression": lambda: LinearRegression(),
}


# ---------------------------------------------------------------------------
# JSON data loading
# ---------------------------------------------------------------------------


def load_benchmark_json(source: str | Path) -> dict[str, Any]:
    """Load and validate a benchmark JSON file.

    Args:
        source: Path to a JSON file, or a JSON string.

    Returns:
        Parsed dict with "search_space", "trials", and optionally "minimize".
    """
    source_str = str(source)
    if source_str.strip().startswith("{"):
        data = json.loads(source_str)
    else:
        with open(source_str) as f:
            data = json.load(f)

    if "search_space" not in data or "trials" not in data:
        raise ValueError("JSON must contain 'search_space' and 'trials' keys.")
    return data


# ---------------------------------------------------------------------------
# BenchmarkTestFunction — surrogate-based evaluation
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class SurrogateFromDataTestFunction(BenchmarkTestFunction):
    """A BenchmarkTestFunction that trains an sklearn surrogate on historical
    trial data and uses it for evaluation.

    Args:
        outcome_names: Single-element list with the metric name.
        train_parameters: List of trial parameter dicts.
        train_metrics: 1-D array of metric values corresponding to each trial.
        surrogate: An sklearn-compatible regressor (fit/predict interface).
    """

    train_parameters: list[dict[str, TParamValue]]
    train_metrics: np.ndarray  # pyre-ignore[13]
    surrogate: Any

    _feature_names: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if len(self.outcome_names) != 1:
            raise ValueError("Only single-objective problems are supported.")
        self._train_surrogate()

    def evaluate_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        X = self._params_to_features([dict(params)])
        y = self.surrogate.predict(X)
        return torch.tensor(y, dtype=torch.double).reshape(1, 1)

    def _params_to_features(
        self, param_dicts: list[dict[str, TParamValue]]
    ) -> pd.DataFrame:
        df = pd.DataFrame(param_dicts)

        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            dummies = pd.get_dummies(df[non_numeric]).astype(float)
            df = df.drop(columns=non_numeric)
            df = pd.concat([df, dummies], axis=1)

        if not self._feature_names:
            self._feature_names = df.columns.tolist()
        else:
            for col in set(self._feature_names) - set(df.columns):
                df[col] = 0.0
            df = df[[c for c in self._feature_names if c in df.columns]]

        return df.fillna(0.0)

    def _train_surrogate(self) -> None:
        X = self._params_to_features(self.train_parameters)
        self.surrogate.fit(X, self.train_metrics)
        logger.info(f"Trained surrogate on {len(self.train_metrics)} samples.")


# ---------------------------------------------------------------------------
# Surrogate creation and cross-validation
# ---------------------------------------------------------------------------


def _extract_trial_data(
    data: dict[str, Any],
    metric_name: str = METRIC_NAME,
) -> tuple[list[dict[str, TParamValue]], np.ndarray]:
    """Extract parameter dicts and metric values from parsed JSON data."""
    param_dicts = [t["parameters"] for t in data["trials"]]
    metric_values = np.array(
        [float(t[metric_name]) for t in data["trials"]], dtype=np.float64
    )
    return param_dicts, metric_values


def create_benchmark_test_function(
    data: dict[str, Any],
    metric_name: str = METRIC_NAME,
    surrogate: Any | None = None,
) -> SurrogateFromDataTestFunction:
    """Create a trained SurrogateFromDataTestFunction from JSON data.

    Args:
        data: Parsed JSON dict (from load_benchmark_json).
        metric_name: Key used for metric values in the trial dicts.
        surrogate: An sklearn-compatible regressor. Defaults to Ridge(alpha=0.1).
    """
    param_dicts, metric_values = _extract_trial_data(data, metric_name)
    if surrogate is None:
        surrogate = LinearRegression()
    return SurrogateFromDataTestFunction(
        outcome_names=[metric_name],
        train_parameters=param_dicts,
        train_metrics=metric_values,
        surrogate=surrogate,
    )


def cross_validate_surrogate(
    data: dict[str, Any],
    metric_name: str = METRIC_NAME,
    surrogate: Any | None = None,
    n_splits: int = 5,
) -> pd.Series:
    """Cross-validate a surrogate model on the benchmark data.

    Args:
        data: Parsed JSON dict (from load_benchmark_json).
        metric_name: Key used for metric values in the trial dicts.
        surrogate: An sklearn-compatible regressor. Defaults to Ridge(alpha=0.1).
        n_splits: Number of cross-validation folds.

    Returns:
        pd.Series with R², MAPE, sample count, and fold count.
    """
    param_dicts, metric_values = _extract_trial_data(data, metric_name)
    if surrogate is None:
        surrogate = LinearRegression()

    df = pd.DataFrame(param_dicts)
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        dummies = pd.get_dummies(df[non_numeric]).astype(float)
        df = df.drop(columns=non_numeric)
        df = pd.concat([df, dummies], axis=1)
    X = df.fillna(0.0)
    y = metric_values

    groups = X.groupby(X.columns.tolist()).ngroup()
    n_unique_groups = groups.nunique()
    n_splits = min(n_splits, n_unique_groups)

    kf = GroupKFold(n_splits=n_splits)

    y_pred_all = []
    y_true_all = []
    for train_idx, test_idx in kf.split(X, y, groups):
        model = clone(surrogate)
        model.fit(X.iloc[train_idx], y[train_idx])
        y_pred_all.extend(model.predict(X.iloc[test_idx]))
        y_true_all.extend(y[test_idx])

    return pd.Series(
        {
            "prediction/r-squared": r2_score(y_true_all, y_pred_all),
            "prediction/mape": mean_absolute_percentage_error(y_true_all, y_pred_all),
            "n_splits": n_splits,
            "n_samples": len(y_true_all),
        }
    )


# ---------------------------------------------------------------------------
# Search space construction
# ---------------------------------------------------------------------------

_LOG_SCALE_VALUES: set[str] = {"log"}


def _build_parameter(
    config: dict[str, Any],
) -> RangeParameter | ChoiceParameter | FixedParameter:
    """Build an Ax Parameter from a config dict."""
    ptype = PARAMETER_TYPE_MAP.get(
        config.get("parameter_type", "float"), ParameterType.FLOAT
    )

    if config["type"] == "range":
        bounds = config["bounds"]
        kwargs: dict[str, Any] = {
            "name": config["name"],
            "parameter_type": ptype,
            "lower": bounds[0],
            "upper": bounds[1],
        }
        scaling = config.get("scaling")
        if scaling and scaling in _LOG_SCALE_VALUES:
            kwargs["log_scale"] = True
        return RangeParameter(**kwargs)

    elif config["type"] == "choice":
        values: list[TParamValue] = config["values"]
        if len(values) == 1:
            return FixedParameter(
                name=config["name"],
                parameter_type=ptype,
                value=values[0],
            )
        return ChoiceParameter(
            name=config["name"],
            parameter_type=ptype,
            values=values,
            is_ordered=config.get("is_ordered", False),
            dependents=config.get("dependent_parameters"),
        )
    else:
        raise ValueError(f"Unknown parameter type: {config['type']}")


def build_search_space(
    config_dicts: list[dict[str, Any]],
) -> SearchSpace:
    """Build an Ax SearchSpace from parameter config dicts."""
    parameters = [_build_parameter(d) for d in config_dicts]
    if any(p.is_hierarchical for p in parameters):
        return HierarchicalSearchSpace(parameters=parameters)
    return SearchSpace(parameters=parameters)


# ---------------------------------------------------------------------------
# BenchmarkProblem creation
# ---------------------------------------------------------------------------


def build_benchmark_problem(
    search_space: SearchSpace,
    test_function: BenchmarkTestFunction,
    metric_name: str,
    minimize: bool,
    optimal_value: float,
    baseline_value: float,
    num_trials: int = 20,
    noise_std: float = 0.0,
    name: str | None = None,
) -> BenchmarkProblem:
    """Create an Ax BenchmarkProblem with proper metric handling.

    This is the shared builder used by both the JSON and DB data paths.
    It sets opt_config_metrics to ensure BenchmarkMetric objects are
    preserved in the experiment (required by the Orchestrator).
    """
    benchmark_metric = BenchmarkMetric(
        name=metric_name,
        lower_is_better=minimize,
        observe_noise_sd=False,
    )
    opt_config = OptimizationConfig(
        objective=AxObjective(metric=benchmark_metric, minimize=minimize),
    )

    if name is None:
        name = f"benchmark_{test_function.outcome_names[0]}"

    return BenchmarkProblem(
        name=name,
        search_space=search_space,
        optimization_config=opt_config,
        num_trials=num_trials,
        optimal_value=optimal_value,
        test_function=test_function,
        noise_std=noise_std,
        baseline_value=baseline_value,
        opt_config_metrics=[benchmark_metric],
    )


def create_benchmark_problem(
    data: dict[str, Any],
    test_function: SurrogateFromDataTestFunction,
    num_trials: int = 20,
    metric_name: str = METRIC_NAME,
    noise_std: float = 0.0,
) -> BenchmarkProblem:
    """Create an Ax BenchmarkProblem from JSON data and a trained test function.

    Args:
        data: Parsed JSON dict (from load_benchmark_json).
        test_function: A trained SurrogateFromDataTestFunction.
        num_trials: Number of optimization trials for the benchmark.
        metric_name: Key used for metric values in the trial dicts.
        noise_std: Standard deviation of synthetic noise added to evaluations.
    """
    minimize = bool(data.get("minimize", True))
    search_space = build_search_space(data["search_space"])

    _, metric_values = _extract_trial_data(data, metric_name)
    optimal_value = float(metric_values.min() if minimize else metric_values.max())
    baseline_value = float(np.median(metric_values))

    return build_benchmark_problem(
        search_space=search_space,
        test_function=test_function,
        metric_name=metric_name,
        minimize=minimize,
        optimal_value=optimal_value,
        baseline_value=baseline_value,
        num_trials=num_trials,
        noise_std=noise_std,
    )


# ---------------------------------------------------------------------------
# Benchmark method helpers
# ---------------------------------------------------------------------------


def get_sobol_method() -> BenchmarkMethod:
    """Create a Sobol (quasi-random) benchmark method."""
    return get_sobol_benchmark_method()


def get_bo_method(
    num_sobol_trials: int = 5,
) -> BenchmarkMethod:
    """Create a Bayesian Optimization benchmark method (Sobol + BoTorch).

    Uses SingleTaskGP with qLogNoisyExpectedImprovement. Equivalent to the
    FASTER_AX generation method (standard GP, no SAASBO).
    """

    return get_sobol_botorch_modular_acquisition(
        model_cls=SingleTaskGP,
        acquisition_cls=qLogNoisyExpectedImprovement,
        num_sobol_trials=num_sobol_trials,
    )


def get_default_bo_method(
    num_sobol_trials: int = 5,
) -> BenchmarkMethod:
    """Create a BO benchmark method using Ax's auto-dispatch.

    Uses method="quality" (fully Bayesian SAAS + input warping) which is
    equivalent to the DEFAULT generation method for high-dimensional spaces.
    """
    gs = choose_generation_strategy(
        GenerationStrategyDispatchStruct(
            method="quality",
            initialization_budget=num_sobol_trials,
        )
    )
    return BenchmarkMethod(
        generation_strategy=gs,
        name="DEFAULT_BO",
    )
