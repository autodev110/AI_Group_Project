"""Streamlit dashboard for visualizing Invasive Weed Optimization on portfolios."""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  # used for sticky timeline JS hook
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from advisor.recommendations import generate_advice
from analytics.kpis import build_portfolio_kpis, iwo_performance_kpis
from dashboard import visuals
from iwo.iwo_core import IWOConfig, run_iwo
from portfolio.data_loader import DataRequest, load_prices_and_returns, load_sp500_benchmark
from portfolio.forecasting import ForecastConfig, project_portfolio
from portfolio.stats import compute_portfolio_metrics
from portfolio.universe import get_top_sp500_universe

st.set_page_config(page_title="IWO Portfolio Evolution", layout="wide")  # widescreen so charts have breathing room
st.markdown(
    """
    <style>
    .timeline-sticky-block {
        position: sticky;
        bottom: 0;
        z-index: 999;
        padding: 0.6rem 0.75rem 0.35rem;
        background-color: var(--background-color, #ffffff);
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 -6px 12px rgba(0, 0, 0, 0.1);
    }
    .timeline-sticky-block .stSlider {
        padding-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_YEARS = 4  # default lookback window for fetching prices
MAX_UNIVERSE = 250  # upper bound on selectable SPY constituents
DEFAULT_UNIVERSE_SIZE = 50  # default number of tickers pulled from SPY holdings

HISTORICAL_SPECS = [
    ("Annual Return", "annual_return", "percent", "Average yearly growth based on historical prices."),
    ("Sharpe Ratio", "sharpe", "number", "Return earned per unit of total volatility (higher is better)."),
    ("Volatility", "volatility", "percent", "Annualized standard deviation of daily returns."),
    ("Max Drawdown", "max_drawdown", "percent", "Largest peak-to-trough loss in the sample."),
    ("Sortino Ratio", "sortino", "number", "Return per unit of downside volatility (penalizes losses more)."),
    ("Hit Rate", "hit_rate", "percent", "Percent of days the portfolio finished positive."),
]

FORECAST_SPECS = [
    ("Expected 1Y Return", "expected_return", "percent", "Average return across simulated 1-year paths."),
    ("10th Percentile Wealth", "pct10", "multiple", "Bad-but-plausible outcome (only 10% of paths ended below)."),
    ("Median Wealth", "pct50", "multiple", "Most-likely outcome across the simulations."),
    ("90th Percentile Wealth", "pct90", "multiple", "Optimistic but still realistic end wealth."),
    ("95% VaR", "var_95", "percent", "There is only a 5% chance of doing worse than this return."),
    ("Probability of Loss", "prob_loss", "percent", "Share of simulations that finished below the starting $1."),
]


def _format_value(value: float, fmt: str) -> str:
    """Pretty-print KPI values according to a simple format keyword."""
    if value is None or pd.isna(value):
        return "—"  # gracefully show an em dash when metrics are missing
    if fmt == "percent":
        return f"{value:.2%}"  # convert to percent with two decimals
    if fmt == "multiple":
        return f"{value:.2f}x"  # express Monte Carlo wealth multipliers
    return f"{value:.2f}"  # default numeric display


def _format_delta(delta: float, fmt: str) -> str | None:
    """Format the delta between two KPIs, respecting percent/multiple units."""
    if delta is None or pd.isna(delta):
        return None  # Streamlit hides the delta arrow when None is returned
    if fmt == "percent":
        return f"{delta:+.2%}"
    if fmt == "multiple":
        return f"{delta:+.2f}x"
    return f"{delta:+.2f}"


def _render_metric_cards(title: str, specs, primary: Dict[str, float], comparison: Dict[str, float] | None) -> None:
    """Render a row of st.metric widgets plus a caption of plain-English hints."""
    st.markdown(f"**{title}**")  # section header
    cols = st.columns(len(specs))  # lay out one metric per column
    for (label, key, fmt, desc) , col in zip(specs, cols):
        # Grab the metric, optionally compute the delta vs. comparison, then display it.
        value = primary.get(key)
        delta = None
        if comparison is not None:
            comp_value = comparison.get(key)
            delta = value - comp_value if value is not None and comp_value is not None else None
        col.metric(label, _format_value(value, fmt), _format_delta(delta, fmt))
        # Provide inline help for each metric right under the card.
        col.caption(desc)


def _forecast_config_from_sidebar(state: Dict) -> ForecastConfig:
    """Build the ForecastConfig object from raw sidebar values."""
    return ForecastConfig(
        horizon_days=state["forecast_horizon"],  # convert slider -> dataclass
        n_paths=state["forecast_paths"],
        method=state["forecast_method"],
        random_seed=state["forecast_seed"],
    )


def _clone_forecast_config(cfg: ForecastConfig, seed: int | None) -> ForecastConfig:
    """Clone forecast settings while letting us change the RNG seed."""
    return ForecastConfig(
        horizon_days=cfg.horizon_days,
        n_paths=cfg.n_paths,
        method=cfg.method,
        sample_paths=cfg.sample_paths,
        random_seed=seed,
    )


def _get_company_metadata(tickers: List[str]) -> Dict[str, Dict[str, str | float | None]]:
    """Fetch and cache company name, sector, and latest price for tickers."""
    cache: Dict[str, Dict[str, str | float | None]] = st.session_state.setdefault("company_metadata", {})
    missing = [ticker for ticker in tickers if ticker not in cache]
    for ticker in missing:
        info: Dict[str, str | float | None] = {"name": ticker, "sector": "N/A", "price": None}
        try:
            ticker_obj = yf.Ticker(ticker)
            raw_info = getattr(ticker_obj, "info", {}) or {}
            fast = getattr(ticker_obj, "fast_info", None)
            info["name"] = raw_info.get("longName") or raw_info.get("shortName") or ticker
            info["sector"] = raw_info.get("sector") or raw_info.get("industry") or "N/A"
            if fast and getattr(fast, "last_price", None) is not None:
                info["price"] = float(fast.last_price)
            elif raw_info.get("currentPrice") is not None:
                info["price"] = float(raw_info["currentPrice"])
        except Exception:
            pass  # fallback info dict already initialized
        cache[ticker] = info
    return {ticker: cache[ticker] for ticker in tickers}


def _build_kpi_bundles(weights: List[float], forecast_cfg: ForecastConfig):
    """Compute historical + forecast KPIs for both IWO and the S&P 500."""
    log_returns = st.session_state.returns  # in-sample asset return matrix
    benchmark_returns = st.session_state.sp500_returns  # ^GSPC series
    if benchmark_returns is None:
        raise RuntimeError("Benchmark data unavailable. Please re-run the simulation.")
    risk_free = st.session_state.iwo_config.risk_free  # capture user input
    iwo_metrics = compute_portfolio_metrics(weights, log_returns, risk_free=risk_free)
    benchmark_metrics = compute_portfolio_metrics([1.0], benchmark_returns, risk_free=risk_free)
    start_date = log_returns.index[-1] if len(log_returns.index) else None  # align future timeline
    base_seed = forecast_cfg.random_seed
    iwo_seed = None if base_seed is None else base_seed + 1  # ensure distinct random draws
    benchmark_forecast = project_portfolio(
        benchmark_metrics.daily_log_returns,
        _clone_forecast_config(forecast_cfg, base_seed),
        start_date=start_date,
    )
    iwo_forecast = project_portfolio(
        iwo_metrics.daily_log_returns,
        _clone_forecast_config(forecast_cfg, iwo_seed),
        start_date=start_date,
    )
    benchmark_bundle = build_portfolio_kpis(benchmark_metrics, benchmark_forecast)
    iwo_bundle = build_portfolio_kpis(iwo_metrics, iwo_forecast)
    return benchmark_bundle, iwo_bundle


def _init_state() -> None:
    """Set up Streamlit session_state keys so reruns stay predictable."""
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("prices", None)
    st.session_state.setdefault("returns", None)
    st.session_state.setdefault("sp500_returns", None)
    st.session_state.setdefault("iwo_config", None)
    st.session_state.setdefault("data_request", None)
    st.session_state.setdefault("iteration_slider", 0)
    st.session_state.setdefault("iteration_slider_pending", None)
    st.session_state.setdefault("dropped_tickers", [])
    st.session_state.setdefault("auto_play_toggle", False)
    st.session_state.setdefault("play_speed", 800)
    st.session_state.setdefault("company_metadata", {})  # cache ticker metadata for the holdings list


def _run_simulation(config: IWOConfig, request: DataRequest) -> None:
    """Fetch data, run IWO, and stash results in session_state."""
    with st.spinner("Downloading data and growing the weed colony..."):
        prices, log_returns, dropped = load_prices_and_returns(request)  # asset panel
        _, sp500_returns = load_sp500_benchmark(request.start, request.end, request.force_refresh)  # index series
        history = list(run_iwo(log_returns, config))  # run generator to completion for later playback
    st.session_state.history = history
    st.session_state.prices = prices
    st.session_state.returns = log_returns
    st.session_state.sp500_returns = sp500_returns
    st.session_state.iwo_config = config
    st.session_state.data_request = request
    st.session_state["iteration_slider_pending"] = 0
    st.session_state.dropped_tickers = dropped
    st.session_state["auto_play_toggle"] = True


def _sidebar_controls() -> Dict:
    """Render all sidebar inputs and return their current values."""
    st.sidebar.header("Data Choices")  # label the first group of controls
    start_default = dt.date.today() - dt.timedelta(days=365 * DEFAULT_YEARS)  # default start = N years back
    end_default = dt.date.today()  # default end = today
    company_count = int(
        st.sidebar.number_input(
            "Number of top S&P 500 companies",
            min_value=10,  # enforce minimum breadth to avoid degenerate panels
            max_value=MAX_UNIVERSE,  # obey the maximum we support for performance reasons
            value=DEFAULT_UNIVERSE_SIZE,  # default to a mid-sized basket (50 names)
            step=5,  # step in increments of 5 so the widget is easy to use
        )
    )
    start_date = st.sidebar.date_input("Start date", value=start_default)  # let users change the lookback window
    end_date = st.sidebar.date_input("End date", value=end_default)
    risk_free = st.sidebar.number_input(
        "Risk-free rate (annual)", min_value=0.0, max_value=0.1, value=0.01, step=0.001
    )  # Sharpe baseline
    max_weight = st.sidebar.slider("Diversification cap (max weight)", 0.05, 1.0, 0.3, 0.05)  # constraint knob
    force_universe = st.sidebar.checkbox("Refresh ticker universe", value=False)  # skip cache when needed
    force_refresh = st.sidebar.checkbox("Force price data refresh", value=False)  # likewise for prices
    tickers = get_top_sp500_universe(
        company_count, force_refresh=force_universe
    )  # autopopulated universe honoring the slider

    st.sidebar.header("IWO Configuration")
    iterations = st.sidebar.slider("Iterations", min_value=10, max_value=1000, value=300, step=10)
    population_max = st.sidebar.slider("Max population (plants)", min_value=20, max_value=300, value=120, step=10)
    population_init = st.sidebar.slider("Initial seeds", min_value=10, max_value=population_max, value=40, step=5)
    seeds_min = st.sidebar.slider("Seeds min", min_value=1, max_value=10, value=2)
    seeds_max = st.sidebar.slider("Seeds max", min_value=seeds_min, max_value=15, value=5)
    sigma_initial = st.sidebar.slider("Initial sigma", min_value=0.01, max_value=1.0, value=0.3)
    sigma_final = st.sidebar.slider("Final sigma", min_value=0.001, max_value=0.2, value=0.01)
    modulation_index = st.sidebar.slider("Modulation index", min_value=0.5, max_value=6.0, value=3.0)

    st.sidebar.header("Objective")
    objective_mode = st.sidebar.radio("Goal", options=["Sharpe ratio", "Return-Vol tradeoff"], index=0)
    return_vol_weight = st.sidebar.slider(
        "Return emphasis (for tradeoff mode)", min_value=0.0, max_value=1.0, value=0.6, step=0.05
    )

    st.sidebar.header("Forecast Settings")
    horizon_days = st.sidebar.slider("Projection horizon (days)", min_value=60, max_value=756, value=252, step=21)
    n_paths = st.sidebar.slider("Simulation paths", min_value=200, max_value=2000, value=1000, step=100)
    forecast_method = st.sidebar.radio("Simulation method", options=["Bootstrap", "Geometric Brownian Motion"], index=0)
    forecast_seed = st.sidebar.number_input("Simulation seed", min_value=0, max_value=9999, value=42, step=1)

    init_btn = st.sidebar.button("Initialize Simulation", type="primary")
    replay_btn = st.sidebar.button("Replay Last Run", disabled=not st.session_state.history)

    sidebar_state = {
        "tickers": tickers,
        "company_count": company_count,
        "start_date": start_date,
        "end_date": end_date,
        "risk_free": risk_free,
        "max_weight": max_weight,
        "iterations": iterations,
        "population_max": population_max,
        "population_init": population_init,
        "seeds_min": seeds_min,
        "seeds_max": seeds_max,
        "sigma_initial": sigma_initial,
        "sigma_final": sigma_final,
        "modulation_index": modulation_index,
        "objective": "return_vol" if objective_mode.startswith("Return") else "sharpe",
        "return_vol_weight": return_vol_weight,
        "forecast_horizon": horizon_days,
        "forecast_paths": n_paths,
        "forecast_method": "gbm" if forecast_method.startswith("Geometric") else "bootstrap",
        "forecast_seed": int(forecast_seed),
        "force_refresh": force_refresh,
        "init_clicked": init_btn,
        "replay_clicked": replay_btn,
    }
    return sidebar_state


def _ensure_simulation(state: Dict) -> None:
    """Validate sidebar inputs and kick off an IWO run if inputs look good."""
    if not state["tickers"]:
        st.warning("Please choose at least one asset to begin.")
        return
    start = pd.Timestamp(state["start_date"])
    end = pd.Timestamp(state["end_date"])
    if start >= end:
        st.warning("Start date must be before end date")
        return
    request = DataRequest(tickers=state["tickers"], start=start, end=end, force_refresh=state["force_refresh"])
    config = IWOConfig(
        it_max=state["iterations"],
        p_max=state["population_max"],
        p_init=state["population_init"],
        s_min=state["seeds_min"],
        s_max=state["seeds_max"],
        sigma_initial=state["sigma_initial"],
        sigma_final=state["sigma_final"],
        modulation_index=state["modulation_index"],
        risk_free=state["risk_free"],
        max_weight=state["max_weight"],
        objective_mode=state["objective"],
        return_vol_weight=state["return_vol_weight"],
    )
    _run_simulation(config, request)


def _metrics_explanation() -> None:
    """Provide a one-sentence refresher on the statistical terms shown."""
    st.caption(
        "Annual return and volatility are annualized from daily log returns. Sharpe ratio measures return per unit of risk,"
        " and max drawdown captures the worst peak-to-trough loss over the sample."
    )


_init_state()
sidebar_state = _sidebar_controls()
if sidebar_state["init_clicked"]:
    try:
        _ensure_simulation(sidebar_state)
    except Exception as exc:  # pragma: no cover - Streamlit surface
        st.error(f"Simulation failed: {exc}")
if sidebar_state["replay_clicked"] and st.session_state.history:
    st.session_state["iteration_slider_pending"] = 0
    st.session_state["auto_play_toggle"] = True

st.title("IWO Portfolio Evolution")
st.subheader("Bio-inspired optimization of financial portfolios")
st.markdown(
    "This dashboard treats portfolio search like a living weed colony. Seeds (candidate portfolios) reproduce, spread,"
    " and compete. The colony learns where the attractive risk-return regions live and eventually stabilizes on a highly"
    " diversified mix of assets."
)

st.info(
    "**Story so far:** You are an investor choosing how to spread your capital across the selected assets."
    " The Invasive Weed Optimization algorithm keeps growing many candidate portfolios and nudging them toward"
    " higher Sharpe ratios while respecting diversification constraints."
)
st.markdown(
    "**What is a portfolio?** It is simply a recipe that says what fraction of your money goes into each asset."
    " The weights always add up to 100% and must stay non-negative in this demo (long-only investing)."
)
st.markdown(
    "**What is IWO?** Think of each weight vector as a plant. Every cycle the plants reproduce, their seeds land"
    " nearby with some randomness, and the environment keeps only the strongest plants. That evolutionary loop"
    " gradually concentrates wealth in smart, diversified allocations."
)
if st.session_state.dropped_tickers:
    removed = ", ".join(st.session_state.dropped_tickers)
    st.warning(f"No usable data for: {removed}. Those tickers were removed from the simulation window.")
selected_universe = sidebar_state["tickers"]
st.caption(
    f"Universe: pulling the top {len(selected_universe)} SPY holdings from Yahoo Finance for optimization."
)  # remind the user how tickers were selected
with st.expander("View included tickers"):
    st.write(
        ", ".join(selected_universe)
    )  # give power users the option to inspect the exact ticker list used in the run

if not st.session_state.history:
    st.info("Tweak the sidebar parameters and click **Initialize Simulation** to grow the colony.")
    st.stop()

history: List[Dict] = st.session_state.history
max_iter = history[-1]["iter"]
pending_value = st.session_state.get("iteration_slider_pending")
if pending_value is not None:
    st.session_state["iteration_slider"] = min(max(pending_value, 0), max_iter)  # clamp to slider bounds
    st.session_state["iteration_slider_pending"] = None  # mark as consumed so Streamlit won't complain
timeline_container = st.container()
with timeline_container:
    iteration = st.slider(
        "Iteration",
        min_value=0,
        max_value=max_iter,
        value=st.session_state.get("iteration_slider", max_iter),
        step=1,
        key="iteration_slider",
    )  # slider drives the "current" world state being visualized
current_state = history[iteration]
state_best_metrics = current_state["best"]["metrics"]  # metrics snapshot for whichever iteration we're showing
phase_label = visuals.phase_from_iteration(iteration, max_iter)
st.markdown(f"**Current IWO Phase:** {phase_label}")
best_weights = current_state["best"]["weights"]
forecast_cfg = _forecast_config_from_sidebar(sidebar_state)
benchmark_bundle, iwo_bundle = _build_kpi_bundles(best_weights, forecast_cfg)
history_subset = history[: iteration + 1]
iwo_performance = iwo_performance_kpis(history_subset, benchmark_bundle.historical)
asset_names = st.session_state.returns.columns.tolist()
advisor_output = generate_advice(benchmark_bundle, iwo_bundle, best_weights, asset_names)

with timeline_container:
    control_cols = st.columns([2, 1, 1])
    with control_cols[0]:
        st.caption(f"Iteration {iteration} / {max_iter}")  # simple progress indicator
    with control_cols[1]:
        auto_play_state = st.session_state.get("auto_play_toggle", False)
        play_label = "Pause ⏸" if auto_play_state else "Play ▶"  # dynamic button text
        if st.button(play_label, key="play_pause_button"):
            next_state = not auto_play_state
            st.session_state["auto_play_toggle"] = next_state
            if next_state and iteration >= max_iter:
                st.session_state["iteration_slider_pending"] = 0  # restart loop when re-enabling playback
            st.rerun()
    with control_cols[2]:
        step_ms = st.slider(
            "Play speed (ms)",
            min_value=50,
            max_value=2000,
            value=st.session_state.get("play_speed", 800),
            step=50,
            key="play_speed",
        )
components.html(
    """
    <script>
    // Locate the iteration slider in the parent DOM and append our sticky class.
    const sliderRoot = window.parent.document.querySelector('div[data-testid="stSlider"][aria-label="Iteration"]');
    if (sliderRoot) {
        const block = sliderRoot.closest('section[data-testid="stVerticalBlock"]');
        if (block && !block.classList.contains('timeline-sticky-block')) {
            block.classList.add('timeline-sticky-block');
        }
    }
    </script>
    """,
    height=0,
)
auto_play = st.session_state.get("auto_play_toggle", False)

st.markdown("## KPI Overview")
kpi_tabs = st.tabs(["IWO Portfolio", "S&P 500 Baseline"])  # allow quick toggling between portfolios
with kpi_tabs[0]:
    _render_metric_cards("Historical KPIs", HISTORICAL_SPECS, iwo_bundle.historical, benchmark_bundle.historical)
with kpi_tabs[1]:
    _render_metric_cards("Historical KPIs", HISTORICAL_SPECS, benchmark_bundle.historical, None)

forecast_tabs = st.tabs(["IWO Forecast KPIs", "S&P 500 Forecast KPIs"])
with forecast_tabs[0]:
    _render_metric_cards("Forecast KPIs", FORECAST_SPECS, iwo_bundle.forecast, benchmark_bundle.forecast)
with forecast_tabs[1]:
    _render_metric_cards("Forecast KPIs", FORECAST_SPECS, benchmark_bundle.forecast, None)

st.markdown("### IWO Performance Lens")
iwo_cols = st.columns(4)
iwo_cols[0].metric("Sharpe Improvement", _format_value(iwo_performance["sharpe_improvement"], "number"))
iwo_cols[1].metric("Drawdown Reduction", _format_value(iwo_performance["drawdown_reduction"], "percent"))
iwo_cols[2].metric("90% Sharpe Reached By Iter", str(int(iwo_performance["convergence_iter"])))
iwo_cols[3].metric("Final Cost", f"{iwo_performance['final_cost']:.3f}")
st.caption(
    "IWO improves risk-adjusted return relative to the S&P 500 benchmark and converges once 90% of the final Sharpe ratio is achieved."
)

st.markdown("### Core Dynamics")
left, right = st.columns(2)
with left:
    st.markdown("**Fitness Convergence** — The colony steadily lowers cost (raises Sharpe) as weaker portfolios are replaced.")
    st.plotly_chart(visuals.convergence_figure(history[: iteration + 1]), use_container_width=True)
with right:
    st.markdown("**Wealth Generation** — Compare the live best weed vs. the S&P 500 benchmark in dollars.")
    baseline_curve = benchmark_bundle.metrics.wealth_curve
    optimized_curve = iwo_bundle.metrics.wealth_curve
    st.plotly_chart(
        visuals.wealth_curve_figure(baseline_curve, optimized_curve, baseline_label="S&P 500"),
        use_container_width=True,
    )

st.markdown("### Search Landscape & Dominant Species")
landscape_col, genome_col = st.columns(2)
with landscape_col:
    st.markdown(
        "Each dot is a candidate plant (portfolio) at the selected iteration. Colors show Sharpe ratio — the colony pushes"
        " toward the top-left: lower risk, higher return."
    )
    landscape_fig = visuals.search_landscape(current_state["population"])
    visuals.highlight_best_point(landscape_fig, state_best_metrics)
    st.plotly_chart(landscape_fig, use_container_width=True)
with genome_col:
    st.markdown("Weights describe the DNA of the lead plant. Bigger bubbles signal heavier allocations.")
    tickers = st.session_state.returns.columns.tolist()
    st.plotly_chart(visuals.weights_bubble_chart(tickers, current_state["best"]["weights"]), use_container_width=True)
    weights_table = pd.DataFrame({"Asset": tickers, "Weight": current_state["best"]["weights"]})
    st.dataframe(weights_table.style.format({"Weight": "{:.2%}"}), hide_index=True)

st.markdown("### Projections & Scenario Analysis")
projection_tabs = st.tabs(["IWO Monte Carlo", "S&P 500 Monte Carlo"])
with projection_tabs[0]:
    st.markdown(
        "We simulate thousands of futures using historical daily returns as the DNA for new random paths. The blue fan"
        " shows how $1 could evolve; the median line is the most typical journey."
    )
    iwo_forecast = iwo_bundle.forecast_result
    st.plotly_chart(
        visuals.forecast_fan_chart(iwo_forecast.sample_paths, iwo_forecast.median_path, "IWO Simulated Wealth"),
        use_container_width=True,
    )
    st.markdown("**Optimized Holdings Snapshot** — ranks current weights with live fundamentals.")
    metadata = _get_company_metadata(asset_names)
    sorted_positions = sorted(
        zip(asset_names, current_state["best"]["weights"]),
        key=lambda pair: pair[1],
        reverse=True,
    )
    for ticker, weight in sorted_positions:
        if weight <= 0:
            continue
        info = metadata.get(ticker, {})
        price = info.get("price")
        price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "N/A"
        name = info.get("name", ticker)
        sector = info.get("sector", "N/A")
        st.markdown(
            f"**{ticker}** — {name} | {sector} | Price {price_str} | Weight {weight:.2%}",
            help="Sorted by allocation so you can instantly see the dominant positions.",
        )
with projection_tabs[1]:
    st.markdown("Baseline projection uses the same simulator but feeds in the S&P 500 index return stream.")
    base_forecast = benchmark_bundle.forecast_result
    st.plotly_chart(
        visuals.forecast_fan_chart(base_forecast.sample_paths, base_forecast.median_path, "Baseline Simulated Wealth"),
        use_container_width=True,
    )
st.caption(
    "Fan charts approximate a range of plausible outcomes. The holdings snapshot above shows the live composition of the IWO portfolio."
)

st.markdown("### Interpretation & Advice")
st.info(f"**Risk Profile:** {advisor_output['risk_profile']}")
st.markdown(f"**Performance Summary:** {advisor_output['performance_summary']}")
st.markdown(f"**Diversification Check:** {advisor_output['diversification_comment']}")
st.markdown("**Actionable Ideas:**")
for bullet in advisor_output["actionable_advice"]:
    st.markdown(f"- {bullet}")

auto_play = st.session_state.get("auto_play_toggle", False)
if auto_play and iteration < max_iter:
    time.sleep(step_ms / 1000)  # throttle so the animation is visible
    st.session_state["iteration_slider_pending"] = iteration + 1  # request the next frame
    st.rerun()  # re-run the app immediately to simulate animation
