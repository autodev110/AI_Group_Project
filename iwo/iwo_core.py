"""Invasive Weed Optimization implementation specialized for portfolio search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, List

import numpy as np
import pandas as pd

from .fitness import ConstraintConfig, FitnessResult, baseline_metrics, evaluate_portfolio


@dataclass
class IWOConfig:
    """Configuration bundle for the IWO metaheuristic."""

    it_max: int = 500
    p_max: int = 100
    s_min: int = 2
    s_max: int = 5
    sigma_initial: float = 0.2
    sigma_final: float = 0.01
    modulation_index: float = 3.0
    p_init: int = 20
    rng_seed: int | None = 7
    risk_free: float = 0.0
    max_weight: float | None = 0.3
    penalty_strength: float = 10.0
    objective_mode: str = "sharpe"
    return_vol_weight: float = 0.5


def _project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex (long-only, sum=1)."""
    v = vector.astype(float)
    if v.sum() <= 0:
        v = np.ones_like(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u - (cssv - 1) / (np.arange(1, len(v) + 1)) > 0)[0]
    if rho.size == 0:
        return np.full_like(v, 1.0 / len(v))
    rho_val = rho[-1]
    theta = (cssv[rho_val] - 1) / (rho_val + 1)
    w = np.maximum(v - theta, 0)
    if w.sum() == 0:
        return np.full_like(v, 1.0 / len(v))
    return w / w.sum()


def _random_weights(n_assets: int, rng: np.random.Generator) -> np.ndarray:
    weights = rng.random(n_assets)
    return weights / weights.sum()


def _sigma_for_iteration(config: IWOConfig, iteration: int) -> float:
    progress = iteration / max(config.it_max, 1)
    annealing = (1 - progress) ** config.modulation_index
    return config.sigma_final + annealing * (config.sigma_initial - config.sigma_final)


def _serialize_candidate(candidate: Dict, include_curve: bool = False) -> Dict:
    metrics = candidate["metrics"]
    metrics_dict = metrics.as_dict()
    if include_curve:
        metrics_dict["wealth_curve"] = metrics.wealth_curve
    return {
        "weights": candidate["weights"].tolist(),
        "cost": candidate["cost"],
        "penalty": candidate["penalty"],
        "objective_value": candidate.get("objective_value"),
        "metrics": metrics_dict,
    }


def _new_candidate(weights: np.ndarray, log_returns: pd.DataFrame, fitness_cfg: ConstraintConfig, config: IWOConfig) -> Dict:
    result: FitnessResult = evaluate_portfolio(
        weights,
        log_returns,
        risk_free=config.risk_free,
        constraints=fitness_cfg,
        objective=config.objective_mode,
        return_vol_weight=config.return_vol_weight,
    )
    return {
        "weights": weights,
        "cost": result.cost,
        "penalty": result.penalty,
        "metrics": result.metrics,
        "objective_value": result.objective_value,
    }


def _assign_seed_count(cost: float, best_cost: float, worst_cost: float, config: IWOConfig) -> int:
    if not np.isfinite(best_cost) or not np.isfinite(worst_cost) or best_cost == worst_cost:
        return config.s_min
    ratio = (worst_cost - cost) / (worst_cost - best_cost)
    seeds = int(np.round(config.s_min + ratio * (config.s_max - config.s_min)))
    return int(np.clip(seeds, config.s_min, config.s_max))


def run_iwo(log_returns: pd.DataFrame, config: IWOConfig) -> Generator[Dict, None, None]:
    """Yield per-iteration population statistics for the IWO search."""
    rng = np.random.default_rng(config.rng_seed)
    n_assets = log_returns.shape[1]
    fitness_cfg = ConstraintConfig(max_weight=config.max_weight, penalty_strength=config.penalty_strength)
    population: List[Dict] = []
    for _ in range(config.p_init):
        weights = _random_weights(n_assets, rng)
        population.append(_new_candidate(weights, log_returns, fitness_cfg, config))
    population.sort(key=lambda c: c["cost"])
    baseline = baseline_metrics(log_returns, risk_free=config.risk_free)
    baseline_summary = baseline.as_dict()
    baseline_summary["wealth_curve"] = baseline.wealth_curve

    def _state(iteration: int) -> Dict:
        best_candidate = population[0]
        return {
            "iter": iteration,
            "population": [_serialize_candidate(cand) for cand in population],
            "best": _serialize_candidate(best_candidate, include_curve=True),
            "baseline": baseline_summary,
        }

    yield _state(0)

    for iteration in range(1, config.it_max + 1):
        sigma = _sigma_for_iteration(config, iteration)
        parents = list(population)
        offspring: List[Dict] = []
        best_cost = parents[0]["cost"]
        worst_cost = parents[-1]["cost"]
        for cand in parents:
            seed_count = _assign_seed_count(cand["cost"], best_cost, worst_cost, config)
            for _ in range(seed_count):
                noise = rng.normal(0, sigma, size=n_assets)
                child_weights = _project_to_simplex(cand["weights"] + noise)
                offspring.append(_new_candidate(child_weights, log_returns, fitness_cfg, config))
        population.extend(offspring)
        population.sort(key=lambda c: c["cost"])
        population = population[: config.p_max]
        yield _state(iteration)
