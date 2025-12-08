"""KPI aggregation utilities for the dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from portfolio.forecasting import ForecastResult
from portfolio.stats import PortfolioMetrics


@dataclass
class PortfolioKPIBundle:
    """Container joining historical metrics and forecast outputs."""

    historical: Dict[str, float]
    forecast: Dict[str, float]
    metrics: PortfolioMetrics
    forecast_result: ForecastResult


def historical_kpis(metrics: PortfolioMetrics) -> Dict[str, float]:
    return {
        "annual_return": metrics.annual_return,
        "volatility": metrics.volatility,
        "sharpe": metrics.sharpe,
        "max_drawdown": metrics.max_drawdown,
        "sortino": metrics.sortino,
        "hit_rate": metrics.hit_rate,
    }


def forecast_kpis(result: ForecastResult) -> Dict[str, float]:
    return result.metric_dict()


def build_portfolio_kpis(metrics: PortfolioMetrics, forecast: ForecastResult) -> PortfolioKPIBundle:
    return PortfolioKPIBundle(
        historical=historical_kpis(metrics),
        forecast=forecast_kpis(forecast),
        metrics=metrics,
        forecast_result=forecast,
    )


def iwo_performance_kpis(history: List[Dict], baseline_metrics: Dict[str, float]) -> Dict[str, float]:
    final_state = history[-1]
    final_metrics = final_state["best"]["metrics"]
    sharpe_improvement = final_metrics["sharpe"] - baseline_metrics["sharpe"]
    drawdown_reduction = baseline_metrics["max_drawdown"] - final_metrics["max_drawdown"]
    final_cost = final_state["best"]["cost"]
    target = 0.9 * final_metrics["sharpe"]
    convergence_iter = final_state["iter"]
    for state in history:
        if state["best"]["metrics"]["sharpe"] >= target:
            convergence_iter = state["iter"]
            break
    return {
        "sharpe_improvement": sharpe_improvement,
        "drawdown_reduction": drawdown_reduction,
        "convergence_iter": convergence_iter,
        "final_cost": final_cost,
    }
