# Artificial Intelligence Group Project

**Semester:** Fall 2025  
**Due Date:** December 9, 2025  
**Project Type:** Group Project

---

## Overview
Complete an end-to-end optimization workflow using one of the assigned research articles as your methodological guide. Teams must execute the selected optimization approach, document the process, and share well-commented code alongside the project report.

---

## Project Requirements
1. **Optimization Execution:** Choose one method from the three articles assigned across the group. Collect suitable open-source or benchmark data (subsample if needed) and apply the chosen optimization approach.
2. **Project Report:** Produce a concise report (maximum 8 pages, including visuals) detailing methodology, experiments, results, analyses, and discussions. Cite all referenced literature.
3. **Code Submission:** Submit Python code that reproduces the experiments, ensuring clear, explanatory comments throughout.

**Note:** Step 2 must be completed using the Python programming language.

---

## Team Members
- Daniel Nikiforov
- Dev Thaker
- Nana Offei

---

## Implementation Overview
This repository now contains a full end-to-end workflow for the **Invasive Weed Optimization (IWO)** approach to long-only portfolio construction:

- `portfolio/` – utilities to fetch Yahoo Finance prices (`data_loader.py`) and compute portfolio statistics such as annual return, volatility, Sharpe ratio, max drawdown, and wealth curves (`stats.py`).
- `portfolio/universe.py` – pulls the top SPY holdings from Yahoo Finance (with caching/fallback) so you can specify “use the top N S&P 500 names” instead of manually selecting tickers.
- `portfolio/forecasting.py` – Monte Carlo engines (bootstrap or GBM) that simulate future wealth paths, compute Value-at-Risk, percentile outcomes, and probability-of-loss KPIs.
- `analytics/` – KPI aggregation helpers (`kpis.py`) that align historical stats with forecast metrics and report IWO-specific KPIs (Sharpe improvement, drawdown reduction, convergence iteration, final cost).
- `advisor/` – the rule-based recommendation layer that turns KPIs into human-readable risk profiles, summaries, and action items.
- `iwo/` – the pedagogical IWO implementation (`iwo_core.py`) plus cost/penalty helpers (`fitness.py`). It exposes a generator that yields the entire population every iteration so the UI can replay the optimization timeline.
- `dashboard/` – a Streamlit application (`app.py`) and Plotly figure factory (`visuals.py`). The app highlights the weed-colony narrative, explains the math with inline notes/expanders, compares the optimized portfolio with an S&P 500 baseline, and now surfaces KPI cards, simulations, and advisory text.
- `data/` – cached parquet files for quick demos. Fetch once, reuse offline.
- `report/figures/` – export plots from the dashboard directly into the 8-page report.

## Getting Started
1. Create a virtual environment (recommended) and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Launch the explorable dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
3. Use the sidebar to specify how many of the top SPY holdings you want (defaults to 50), set the sample window, and adjust IWO controls (iterations, population size, dispersal sigmas, objective mode, diversification cap, etc.). Click **Initialize Simulation** to grow a fresh colony or **Replay Last Run** to scrub through the cached history.

The application downloads adjusted close data from Yahoo Finance, caches it to `data/cached_prices.parquet`, computes log returns, and streams the weed population for every iteration. The S&P 500 index (^GSPC) is fetched separately and used as the baseline for all KPIs, including wealth comparisons and forecasts. All charts (fitness convergence, wealth curves, risk-return scatter, bubble weights, Monte Carlo fan charts, forecast histograms) can be exported for the 8-page report under `report/figures/`. KPI cards at the top summarize both historical and forecast performance for the S&P 500 baseline and the IWO-optimized portfolio, while the advisory layer explains what the numbers mean in practice.

## Writing the Report
The dashboard embeds all the elements requested in the project plan:

1. **Introduction / Story:** The header explains the “you have capital + weeds evolve portfolios” narrative in plain language.
2. **Math / Objective:** Inline captions and expanders define the portfolio constraints, Sharpe ratio, and cost function. The sidebar shows the objective toggle (pure Sharpe vs. return-volatility trade-off).
3. **Algorithm Section:** The UI footer/timeline and “Narrated Run” expander describe initialization, reproduction, dispersal, and competitive exclusion phases while displaying the live colony.
4. **Results Figures:** Export convergence curves, wealth curves, risk-return landscapes, bubble-weight visualizations, Monte Carlo fan charts, and ending-wealth distributions from `dashboard/visuals.py` for direct inclusion in the PDF.

Use the metrics tables plus the baseline vs. optimized comparison to write the discussion and connect observations back to the antenna-paper motivation cited in the course notes. When referencing numbers in the report, cite the Streamlit-generated figures so the grader can replicate them.


