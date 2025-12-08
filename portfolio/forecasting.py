"""Monte Carlo forecasting utilities for projecting future portfolio behavior."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class ForecastConfig:
    """Parameters controlling the Monte Carlo projection."""

    horizon_days: int = 252
    n_paths: int = 1000
    method: str = "bootstrap"  # or "gbm"
    sample_paths: int = 40
    random_seed: int | None = None


@dataclass
class ForecastResult:
    """Summary statistics and sample path data from simulations."""

    expected_return: float
    pct10: float
    pct50: float
    pct90: float
    var_95: float
    prob_loss: float
    expected_max_drawdown: float
    ending_distribution: np.ndarray
    sample_paths: pd.DataFrame
    median_path: pd.Series

    def metric_dict(self) -> Dict[str, float]:
        return {
            "expected_return": self.expected_return,
            "pct10": self.pct10,
            "pct50": self.pct50,
            "pct90": self.pct90,
            "var_95": self.var_95,
            "prob_loss": self.prob_loss,
            "expected_max_drawdown": self.expected_max_drawdown,
        }


def _simulate_log_returns(log_returns: np.ndarray, config: ForecastConfig) -> np.ndarray:
    """Draw synthetic daily log returns via bootstrap or GBM."""
    rng = np.random.default_rng(config.random_seed)
    if config.method == "gbm":
        mu = float(np.mean(log_returns))
        sigma = float(np.std(log_returns, ddof=1))
        draws = rng.normal(mu, sigma, size=(config.n_paths, config.horizon_days))
    else:
        draws = rng.choice(log_returns, size=(config.n_paths, config.horizon_days), replace=True)
    return draws


def _wealth_paths_from_draws(log_draws: np.ndarray) -> np.ndarray:
    """Convert simulated log returns into wealth paths starting at $1."""
    cumulative = np.cumsum(log_draws, axis=1)
    wealth = np.exp(cumulative)
    wealth = np.concatenate([np.ones((wealth.shape[0], 1)), wealth], axis=1)
    return wealth


def _max_drawdown_per_path(paths: np.ndarray) -> np.ndarray:
    """Compute the max drawdown for every simulated path."""
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = paths / running_max - 1
    return drawdowns.min(axis=1)


def simulate_paths(
    daily_log_returns: np.ndarray,
    config: ForecastConfig,
    start_date: pd.Timestamp | None = None,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Simulate wealth paths and return the array plus timeline index."""
    log_draws = _simulate_log_returns(daily_log_returns, config)
    wealth = _wealth_paths_from_draws(log_draws)
    if start_date is None:
        timeline = pd.RangeIndex(0, wealth.shape[1])
    else:
        timeline = pd.bdate_range(start=start_date, periods=wealth.shape[1], freq="B")
    return wealth, timeline


def project_portfolio(
    daily_log_returns: np.ndarray,
    config: ForecastConfig,
    start_date: pd.Timestamp | None = None,
) -> ForecastResult:
    """Generate Monte Carlo projections and summarize key risk/return KPIs."""
    wealth, timeline = simulate_paths(daily_log_returns, config, start_date=start_date)
    ending = wealth[:, -1]
    expected_return = float(ending.mean() - 1)
    pct10, pct50, pct90 = np.percentile(ending, [10, 50, 90])
    var_95 = float(np.percentile(ending, 5) - 1)
    prob_loss = float(np.mean(ending < 1.0))
    max_drawdowns = _max_drawdown_per_path(wealth)
    expected_max_drawdown = float(max_drawdowns.mean())
    # Sample paths for fan charts
    sample_count = min(config.sample_paths, wealth.shape[0])
    sample_idx = np.linspace(0, wealth.shape[0] - 1, sample_count, dtype=int)
    sample_paths = pd.DataFrame(wealth[sample_idx].T, index=timeline)
    median_path = pd.Series(np.median(wealth, axis=0), index=timeline, name="Median Path")
    return ForecastResult(
        expected_return=expected_return,
        pct10=float(pct10),
        pct50=float(pct50),
        pct90=float(pct90),
        var_95=var_95,
        prob_loss=prob_loss,
        expected_max_drawdown=expected_max_drawdown,
        ending_distribution=ending,
        sample_paths=sample_paths,
        median_path=median_path,
    )
