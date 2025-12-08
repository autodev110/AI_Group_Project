"""Utilities for downloading and caching price data for the IWO demo.

This module centralizes all filesystem I/O so other packages interact purely
with pandas DataFrames.  It handles three main responsibilities:

1. Pulling multi-asset panels from Yahoo Finance, forward-filling / cleaning
   them, and caching them to `data/cached_prices.parquet`.
2. Tracking metadata about the cached window so we only refresh when the user
   widens the requested dates or forces a refresh.
3. Downloading and caching the S&P 500 (^GSPC) benchmark separately so the
   dashboard can always compare against a consistent baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import json

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path("data")
CACHE_FILE = DATA_DIR / "cached_prices.parquet"
META_FILE = DATA_DIR / "cached_prices_meta.json"
BENCH_FILE = DATA_DIR / "sp500_benchmark.parquet"
BENCH_META_FILE = DATA_DIR / "sp500_benchmark_meta.json"


@dataclass
class DataRequest:
    """Container describing a price-data request."""

    tickers: List[str]
    start: pd.Timestamp
    end: pd.Timestamp
    force_refresh: bool = False

    def normalized(self) -> "DataRequest":
        """Return a normalized copy with sorted tickers and naive timestamps."""
        start = pd.Timestamp(self.start).tz_localize(None)
        end = pd.Timestamp(self.end).tz_localize(None)
        tickers = sorted({t.upper().strip() for t in self.tickers})
        if not tickers:
            raise ValueError("At least one ticker must be provided")
        if start >= end:
            raise ValueError("Start date must be before end date")
        return DataRequest(tickers=tickers, start=start, end=end, force_refresh=self.force_refresh)


@dataclass
class CacheMetadata:
    """Metadata describing the cached dataset on disk."""

    tickers: List[str]
    start: str
    end: str

    @classmethod
    def from_disk(cls, path: Path = META_FILE) -> "CacheMetadata | None":
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
            return cls(**payload)
        except Exception:
            return None

    def to_disk(self, path: Path = META_FILE) -> None:
        path.write_text(json.dumps({"tickers": self.tickers, "start": self.start, "end": self.end}))


def _download_prices(request: DataRequest) -> pd.DataFrame:
    """Download adjusted close prices for the requested tickers."""
    data = yf.download(
        request.tickers,
        start=request.start,
        end=request.end + pd.Timedelta(days=1),
        progress=False,
        auto_adjust=False,
        actions=False,
    )
    adj_close = data.get("Adj Close") if isinstance(data, pd.DataFrame) else data  # defensive: Series when single ticker
    if adj_close is None:
        raise RuntimeError("Yahoo Finance response did not contain adjusted close prices")
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name=request.tickers[0])
    adj_close = adj_close.sort_index().dropna(how="all")
    if getattr(adj_close.index, "tz", None) is not None:
        adj_close.index = adj_close.index.tz_localize(None)
    return adj_close


def _cache_is_valid(request: DataRequest, meta: CacheMetadata | None) -> bool:
    if request.force_refresh:
        return False
    if meta is None:
        return False  # no metadata means no cache
    requested = set(request.tickers)
    cached = set(meta.tickers)
    if not requested.issubset(cached):
        return False  # cache missing one of the desired tickers
    return pd.Timestamp(meta.start) <= request.start and pd.Timestamp(meta.end) >= request.end


def _load_cached_prices(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _clean_price_panel(raw: pd.DataFrame, tickers: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Filter to overlapping history, forward-fill, and flag removed tickers."""
    subset = raw.loc[:, tickers]
    subset = subset.sort_index().astype(float)
    subset = subset.dropna(how="all")
    dropped_all_nan = [col for col in subset.columns if subset[col].notna().sum() == 0]
    if dropped_all_nan:
        subset = subset.drop(columns=dropped_all_nan)
    subset = subset.ffill().dropna(how="any")
    if subset.empty:
        raise ValueError("No overlapping price history for the requested assets and date range.")
    return subset, dropped_all_nan


def load_prices_and_returns(request: DataRequest) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Return price/log-return DataFrames plus any dropped tickers."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    norm_request = request.normalized()
    meta = CacheMetadata.from_disk()
    cached_prices = _load_cached_prices(CACHE_FILE)

    if cached_prices is None or not _cache_is_valid(norm_request, meta):
        prices = _download_prices(norm_request)
        prices.to_parquet(CACHE_FILE)
        CacheMetadata(
            tickers=norm_request.tickers,
            start=str(norm_request.start.date()),
            end=str(norm_request.end.date()),
        ).to_disk()
    else:
        prices = cached_prices

    window = prices.loc[norm_request.start : norm_request.end, norm_request.tickers]
    cleaned, dropped = _clean_price_panel(window, [col for col in window.columns])  # keep contiguous overlapping prices
    log_returns = np.log(cleaned / cleaned.shift(1)).dropna(how="any")
    if log_returns.empty:
        raise ValueError("Not enough price observations remain after cleaning the data.")
    return cleaned, log_returns, dropped


def _download_sp500_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data = yf.download(
        "^GSPC",
        start=start,
        end=end + pd.Timedelta(days=1),
        progress=False,
        auto_adjust=False,
        actions=False,
    )
    adj_close = data.get("Adj Close") if isinstance(data, pd.DataFrame) else data
    if adj_close is None:
        raise RuntimeError("Yahoo Finance response did not contain benchmark prices.")
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name="^GSPC")
    adj_close = adj_close.sort_index().dropna(how="all")
    if getattr(adj_close.index, "tz", None) is not None:
        adj_close.index = adj_close.index.tz_localize(None)
    return adj_close


def load_sp500_benchmark(
    start: pd.Timestamp,
    end: pd.Timestamp,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return price/log-return DataFrames for the S&P 500 benchmark."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(start).tz_localize(None)
    end = pd.Timestamp(end).tz_localize(None)
    meta = CacheMetadata.from_disk(BENCH_META_FILE)
    cached_prices = _load_cached_prices(BENCH_FILE)
    needs_refresh = (
        force_refresh
        or cached_prices is None
        or meta is None
        or pd.Timestamp(meta.start) > start
        or pd.Timestamp(meta.end) < end
    )
    if needs_refresh:
        prices = _download_sp500_prices(start, end)
        prices.to_parquet(BENCH_FILE)
        CacheMetadata(
            tickers=["^GSPC"],
            start=str(prices.index.min().date()),
            end=str(prices.index.max().date()),
        ).to_disk(BENCH_META_FILE)
    else:
        prices = cached_prices
    window = prices.loc[start:end, :]
    window = window.ffill().dropna(how="any")
    log_returns = np.log(window / window.shift(1)).dropna(how="any")
    if log_returns.empty:
        raise ValueError("Benchmark series does not contain sufficient data for the selected range.")
    return window, log_returns
