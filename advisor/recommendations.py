"""Rule-based advisory layer translating KPI values into actionable guidance.

Each helper in this module focuses on turning a slice of the KPI bundle into
human-readable insight so the Streamlit app can explain results without any
LLM calls.  All rules are intentionally simple and transparent: thresholds are
hard-coded and justified inline so the project report can reference the exact
logic used to produce a recommendation.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from analytics.kpis import PortfolioKPIBundle


def _risk_profile(volatility: float, var_95: float) -> str:
    """Map volatility + Value-at-Risk into a coarse risk label."""
    if volatility < 0.12 and var_95 > -0.1:
        return "Conservative"
    if volatility < 0.2 and var_95 > -0.2:
        return "Moderate"
    return "Aggressive"


def _performance_summary(baseline: PortfolioKPIBundle, optimized: PortfolioKPIBundle) -> str:
    """Summarize Sharpe improvements relative to the benchmark."""
    sharpe_delta = optimized.historical["sharpe"] - baseline.historical["sharpe"]
    if sharpe_delta > 0.2:
        return (
            f"IWO materially improves risk-adjusted returns (Sharpe +{sharpe_delta:.2f}) while keeping volatility"
            " in a comparable range."
        )
    if sharpe_delta > 0.05:
        return (
            f"IWO delivers a modest Sharpe uplift (+{sharpe_delta:.2f}); expect slightly better reward for roughly"
            " the same risk."
        )
    if sharpe_delta < -0.05:
        return (
            f"The optimized portfolio trails the baseline on Sharpe (-{abs(sharpe_delta):.2f}); consider rerunning with"
            " different constraints or risk settings."
        )
    return "IWO performs similarly to the S&P 500 baseline in terms of risk-adjusted return."


def _diversification_comment(weights: Sequence[float], assets: Sequence[str]) -> str:
    """Highlight the most concentrated sleeve and warn the user if needed."""
    max_weight = max(weights)
    idx = weights.index(max_weight)
    asset = assets[idx]
    if max_weight > 0.3:
        return f"Portfolio is highly concentrated in {asset} ({max_weight:.1%}). Large moves there will dominate results."
    if max_weight > 0.2:
        return f"{asset} is the single largest sleeve ({max_weight:.1%}); monitor it for outsized impact."
    return "Weights stay well diversified; no single name exceeds 20% of the capital."


def _actionable_points(
    baseline: PortfolioKPIBundle,
    optimized: PortfolioKPIBundle,
    weights: Sequence[float],
    assets: Sequence[str],
) -> List[str]:
    """Produce a bullet list of annotated coaching points derived from KPIs."""
    actions: List[str] = []
    sharpe_delta = optimized.historical["sharpe"] - baseline.historical["sharpe"]
    if sharpe_delta > 0:
        actions.append(f"Favor the IWO mix: Sharpe improves by {sharpe_delta:.2f} vs the S&P 500.")
    else:
        actions.append("Sharpe does not improve; tune algorithm parameters or stick with the index baseline.")
    dd_delta = optimized.historical["max_drawdown"] - baseline.historical["max_drawdown"]
    if dd_delta < -0.02:
        actions.append("Drawdowns improve meaningfully under IWO; risk of deep losses is reduced.")
    elif dd_delta > 0.02:
        actions.append("Expect deeper drawdowns with the optimized weights—size positions accordingly.")
    loss_prob_delta = optimized.forecast["prob_loss"] - baseline.forecast["prob_loss"]
    if loss_prob_delta < 0:
        actions.append("Probability of losing money in a year drops versus the baseline.")
    elif loss_prob_delta > 0.05:
        actions.append("Higher chance of a losing year—consider shortening horizon or hedging.")
    var95 = optimized.forecast["var_95"]
    if var95 < -0.2:
        actions.append("In a bad year you could lose more than 20%—only suitable for aggressive capital.")
    for weight, asset in zip(weights, assets):
        if weight > 0.25:
            actions.append(f"Consider trimming {asset} ({weight:.1%}) to diversify further.")
            break
    return actions


def generate_advice(
    baseline: PortfolioKPIBundle,
    optimized: PortfolioKPIBundle,
    weights: Sequence[float],
    assets: Sequence[str],
) -> Dict[str, List[str] | str]:
    """Assemble the full advisory payload consumed by the dashboard."""
    risk = _risk_profile(optimized.historical["volatility"], optimized.forecast["var_95"])
    summary = _performance_summary(baseline, optimized)
    diversification = _diversification_comment(list(weights), assets)
    actions = _actionable_points(baseline, optimized, list(weights), assets)
    return {
        "risk_profile": risk,
        "performance_summary": summary,
        "diversification_comment": diversification,
        "actionable_advice": actions,
    }
