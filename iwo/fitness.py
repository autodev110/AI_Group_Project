"""Fitness evaluation utilities for the IWO search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from portfolio.stats import PortfolioMetrics, compute_portfolio_metrics, equal_weight_metrics


@dataclass
class ConstraintConfig:
    """Soft-constraint configuration for portfolio weights."""

    max_weight: float | None = 0.3
    penalty_strength: float = 10.0
    sum_target: float = 1.0
    negative_tolerance: float = 0.0


@dataclass
class FitnessResult:
    """Value object with cost and rich metrics for a candidate portfolio."""

    cost: float
    penalty: float
    metrics: PortfolioMetrics
    objective_value: float


def _constraint_penalty(weights: np.ndarray, config: ConstraintConfig) -> float:
    """Compute a quadratic penalty for violations of the budgeting rules."""
    deviation = abs(weights.sum() - config.sum_target)
    below_zero = np.abs(weights[weights < config.negative_tolerance]).sum()
    diversification = 0.0
    if config.max_weight is not None:
        diversification = np.abs(np.clip(weights - config.max_weight, 0, None)).sum()
    return deviation**2 + below_zero**2 + diversification**2


def evaluate_portfolio(
    weights: Iterable[float],
    log_returns: pd.DataFrame,
    risk_free: float,
    constraints: ConstraintConfig,
    objective: str = "sharpe",
    return_vol_weight: float = 0.5,
) -> FitnessResult:
    """Return the scalar cost and supporting metrics for a candidate weight vector."""
    weights_arr = np.asarray(list(weights), dtype=float)
    metrics = compute_portfolio_metrics(weights_arr, log_returns, risk_free=risk_free)
    penalty = _constraint_penalty(weights_arr, constraints)
    mode = objective.lower()
    if mode == "return_vol":
        weight = float(np.clip(return_vol_weight, 0.0, 1.0))
        score = weight * metrics.annual_return - (1 - weight) * metrics.volatility
    else:
        score = metrics.sharpe
    score = float(score)
    cost = -score + constraints.penalty_strength * penalty
    if not np.isfinite(cost):
        cost = float("inf")
    return FitnessResult(cost=float(cost), penalty=float(penalty), metrics=metrics, objective_value=score)


def baseline_metrics(log_returns: pd.DataFrame, risk_free: float) -> PortfolioMetrics:
    """Helper exposing the equal-weight baseline for quick comparisons."""
    return equal_weight_metrics(log_returns, risk_free=risk_free)
