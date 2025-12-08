"""Portfolio statistic helpers used by the IWO optimizer and dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PortfolioMetrics:
    """Typed view of the computed summary statistics."""

    annual_return: float
    volatility: float
    sharpe: float
    max_drawdown: float
    sortino: float
    hit_rate: float
    wealth_curve: pd.Series
    daily_log_returns: np.ndarray
    daily_simple_returns: np.ndarray

    def as_dict(self) -> Dict[str, float]:
        return {
            "annual_return": self.annual_return,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "sortino": self.sortino,
            "hit_rate": self.hit_rate,
        }


def _validate_weights(weights: Iterable[float], n_assets: int) -> np.ndarray:
    arr = np.asarray(list(weights), dtype=float)
    if arr.ndim != 1:
        raise ValueError("Weights must be a 1-D array-like")
    if arr.size != n_assets:
        raise ValueError(f"Expected {n_assets} weights, received {arr.size}")
    return arr


def _max_drawdown(series: pd.Series) -> float:
    rolling_max = series.cummax()
    drawdowns = series / rolling_max - 1
    return float(drawdowns.min()) if not drawdowns.empty else 0.0


def compute_portfolio_metrics(weights: Iterable[float], log_returns: pd.DataFrame, risk_free: float = 0.0) -> PortfolioMetrics:
    """Compute annualized metrics for the supplied weight vector."""
    weights_arr = _validate_weights(weights, log_returns.shape[1])
    portfolio_log_returns = log_returns.to_numpy() @ weights_arr
    simple_returns = np.exp(portfolio_log_returns) - 1.0
    mean_log_return = np.mean(portfolio_log_returns)
    vol = np.std(portfolio_log_returns, ddof=1)
    annual_return = float(np.exp(mean_log_return * TRADING_DAYS) - 1)
    annual_vol = float(vol * np.sqrt(TRADING_DAYS))
    sharpe = float(((annual_return - risk_free) / annual_vol) if annual_vol > 0 else np.nan)
    wealth_curve = pd.Series(np.exp(np.cumsum(portfolio_log_returns)), index=log_returns.index, name="Wealth")
    max_drawdown = _max_drawdown(wealth_curve)
    downside = np.minimum(simple_returns, 0)
    downside_std = np.sqrt(np.mean(np.square(downside))) * np.sqrt(TRADING_DAYS)
    sortino = float(((annual_return - risk_free) / downside_std) if downside_std > 0 else np.nan)
    hit_rate = float(np.mean(simple_returns > 0)) if simple_returns.size else 0.0
    return PortfolioMetrics(
        annual_return=annual_return,
        volatility=annual_vol,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        sortino=sortino,
        hit_rate=hit_rate,
        wealth_curve=wealth_curve,
        daily_log_returns=portfolio_log_returns,
        daily_simple_returns=simple_returns,
    )


def equal_weight_metrics(log_returns: pd.DataFrame, risk_free: float = 0.0) -> PortfolioMetrics:
    """Convenience helper for the baseline equal-weight portfolio."""
    n_assets = log_returns.shape[1]
    weights = np.full(n_assets, 1.0 / n_assets)
    return compute_portfolio_metrics(weights, log_returns, risk_free=risk_free)
