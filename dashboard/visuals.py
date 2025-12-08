"""Plotly figure builders and helper utilities for the Streamlit app."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def convergence_figure(history: Sequence[Dict]) -> go.Figure:
    iterations = [state["iter"] for state in history]
    costs = [state["best"]["cost"] for state in history]
    sharpes = [state["best"]["metrics"]["sharpe"] for state in history]
    returns = [state["best"]["metrics"]["annual_return"] for state in history]
    volatilities = [state["best"]["metrics"]["volatility"] for state in history]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=costs,
            mode="lines+markers",
            name="Cost (lower is better)",
            text=[
                f"Sharpe: {s:.2f}<br>Return: {r:.1%}<br>Volatility: {v:.1%}"
                for s, r, v in zip(sharpes, returns, volatilities)
            ],
            hovertemplate="Iter %{x}<br>Cost %{y:.3f}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Fitness Convergence",
        xaxis_title="Iteration",
        yaxis_title="Cost = -Sharpe + penalty",
        template="plotly_white",
    )
    return fig


def wealth_curve_figure(baseline_curve: pd.Series, optimized_curve: pd.Series, baseline_label: str = "Baseline") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=baseline_curve.index, y=baseline_curve.values, name=baseline_label))
    fig.add_trace(
        go.Scatter(x=optimized_curve.index, y=optimized_curve.values, name="IWO Portfolio", line=dict(width=3))
    )
    fig.update_layout(
        title="Growth of $1",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
    )
    return fig


def search_landscape(population: Sequence[Dict]) -> go.Figure:
    vol = [cand["metrics"]["volatility"] for cand in population]
    ret = [cand["metrics"]["annual_return"] for cand in population]
    sharpe = [cand["metrics"]["sharpe"] for cand in population]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vol,
            y=ret,
            mode="markers",
            marker=dict(size=10, color=sharpe, colorscale="Viridis", showscale=True, colorbar_title="Sharpe"),
            text=[f"Sharpe: {s:.2f}" for s in sharpe],
            hovertemplate="Vol %{x:.1%}<br>Return %{y:.1%}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Risk vs. Return Landscape",
        xaxis_title="Volatility",
        yaxis_title="Annual Return",
        template="plotly_white",
    )
    return fig


def highlight_best_point(fig: go.Figure, best_metrics: Dict[str, float]) -> go.Figure:
    fig.add_trace(
        go.Scatter(
            x=[best_metrics["volatility"]],
            y=[best_metrics["annual_return"]],
            mode="markers",
            marker=dict(size=16, color="crimson", symbol="star"),
            name="Best Portfolio",
        )
    )
    return fig


def weights_bubble_chart(asset_names: Sequence[str], weights: Sequence[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=asset_names,
            y=weights,
            mode="markers",
            marker=dict(size=[max(w, 0.01) * 200 for w in weights], color=weights, colorscale="Blues", showscale=False),
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Dominant Species: Portfolio Weights",
        xaxis_title="Asset",
        yaxis_title="Weight",
        template="plotly_white",
    )
    return fig


def forecast_fan_chart(sample_paths: pd.DataFrame, median_path: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    for column in sample_paths.columns:
        fig.add_trace(
            go.Scatter(
                x=sample_paths.index,
                y=sample_paths[column],
                mode="lines",
                line=dict(color="rgba(100,149,237,0.2)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=median_path.index,
            y=median_path.values,
            mode="lines",
            line=dict(color="navy", width=3),
            name="Median Path",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Future Days",
        yaxis_title="Wealth ($1 start)",
        template="plotly_white",
    )
    return fig


def ending_distribution_histogram(ending_wealth: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=ending_wealth,
            nbinsx=40,
            marker=dict(color="teal"),
            opacity=0.75,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Ending Wealth",
        yaxis_title="Frequency",
        template="plotly_white",
    )
    return fig


def phase_from_iteration(iteration: int, max_iter: int) -> str:
    fraction = iteration / max(1, max_iter)
    if fraction <= 0.2:
        return "Initialization / Global Exploration"
    if fraction <= 0.7:
        return "Colonization / Mixed Exploration"
    return "Local Refinement / Convergence"
