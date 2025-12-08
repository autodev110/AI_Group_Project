"""Utilities for selecting a large-cap U.S. equity universe."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List
import json

import yfinance as yf

DATA_DIR = Path("data")
UNIVERSE_CACHE = DATA_DIR / "spy_constituents.json"

# Fallback list approximating a broad slice of the S&P 500 in case live downloads fail.
FALLBACK_TICKERS: List[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK.B", "LLY", "TSLA", "JPM",
    "V", "UNH", "XOM", "MA", "AVGO", "JNJ", "WMT", "PG", "HD", "CVX",
    "MRK", "ABBV", "COST", "PEP", "KO", "BAC", "ADBE", "CRM", "PFE", "CSCO",
    "ORCL", "NFLX", "ABT", "TMO", "AMD", "MCD", "ACN", "DHR", "WFC", "LIN",
    "TXN", "PM", "INTC", "HON", "AMGN", "NEE", "QCOM", "BMY", "AMAT", "UPS",
    "SCHW", "RTX", "IBM", "CAT", "COP", "SPGI", "LOW", "MS", "INTU", "AMT",
    "SYK", "BA", "NOW", "ELV", "DE", "BKNG", "LMT", "GE", "ADI", "BLK",
    "AXP", "MDT", "ISRG", "T", "TTE", "MMC", "PYPL", "CI", "TJX", "ADP",
    "MDLZ", "PLD", "PGR", "GILD", "CB", "DUK", "REGN", "SO", "C", "USB",
    "MU", "SHW", "CSX", "VRTX", "FI", "PNC", "FDX", "BDX", "CL", "GM",
    "APD", "EW", "ICE", "NKE", "ADI", "FISV", "ITW", "AON", "PH", "MNST",
    "ZTS", "ETN", "MRNA", "EQIX", "FDX", "MO", "D", "ADSK", "LRCX", "HCA",
    "MMC", "PSA", "EOG", "WM", "CTAS", "NSC", "ORLY", "MPC", "ROP", "AZO",
    "MCO", "EL", "AEP", "SBUX", "AFL", "GD", "TRV", "IDXX", "KMB", "MAR",
    "TDG", "PRU", "EMR", "VZ", "CMCSA", "CRWD", "CME", "PCAR", "YUM", "RSG",
    "KMI", "GIS", "HPQ", "CDNS", "LULU", "CARR", "VLO", "DVN", "MSCI", "HAL",
    "ROST", "COR", "OTIS", "WELL", "IQV", "DAL", "PSX", "NXPI", "PAYX", "VRSK",
    "HLT", "MRVL", "KHC", "DLR", "BIIB", "ANET", "MTD", "ALL", "NOC", "CTSH",
    "TEL", "AIG", "XLNX", "WMB", "DD", "GLW", "EXC", "DOW", "SBAC", "BK",
    "CMG", "AMP", "DHI", "F", "SRE", "ECL", "STZ", "PPG", "A", "MSI",
    "HLN", "STT", "EA", "AME", "FTNT", "WBD", "HUM", "PXD", "KEYS", "OTIS",
    "KDP", "PCG", "ANSS", "TSN", "ODFL", "DLTR", "SWK", "LEN", "KR", "EFX",
    "ED", "FLT", "AWK", "AVB", "K", "AEE", "OKE", "TER", "LHX", "FAST",
    "ALGN", "MTB", "CHTR", "BKR", "CTRA", "AEM", "VICI", "CF", "HBAN", "WST",
    "BAX", "NUE", "WAT", "PAYC", "BR", "CEG", "ZBH", "MLM", "FIS", "EXPE",
    "MKC", "JNPR", "BBY", "WBA", "GWW", "NEM", "ACGL", "VMC", "LKQ", "AAP",
    "TRGP", "BALL", "LVS", "PTC", "STLD", "DPZ", "ETSY", "HPE", "NTRS", "AES",
    "APA", "CMS", "TFX", "LUV", "CPRT", "CNP", "BRO", "KEY", "PKI", "CLX",
    "CINF", "CZR", "IEX", "HSIC", "ALLE", "MPWR", "GL", "RJF", "NDAQ", "IRM",
    "MKTX", "PKG", "RCL", "PPL", "WRB", "BEN", "WHR", "HIG", "TROW", "DG",
    "NRG", "XYL", "KIM", "EVRG", "INCY", "MTCH", "UDR", "FMC", "MCHP", "PHM",
    "SJM", "HSY", "DTE", "AES", "RF", "CAG", "TER", "ZBRA", "AVY", "CPB",
    "MOS", "CDW", "HWM", "PWR", "JBHT", "CTLT", "TXT", "LKQ", "WRK", "UHS",
    "PARA", "BBWI", "TPR", "PFG", "GPC", "AAL", "HRL", "WHR", "TAP", "HII",
    "MRO", "JNPR", "NCLH", "HAS", "FRT", "NI", "VTR", "LNT", "CCL", "SEE",
    "NWSA", "NWS", "AOS", "DXC", "PENN", "OGN", "PNR", "HST", "IP", "GNRC",
    "AIZ", "L" , "RHI", "FOX", "FOXA", "BIO", "CPB", "DISH", "CRL", "IPG",
    "NWL", "VFC", "ALK", "PVH", "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META",
]


def _download_spy_holdings() -> List[str]:
    """Use yfinance to pull SPY holdings sorted by weight."""
    ticker = yf.Ticker("SPY")
    holdings = getattr(ticker, "fund_holdings", None)
    if holdings is None or "symbol" not in holdings:
        return []
    df = holdings.dropna(subset=["symbol"])
    if "holdingPercent" in df.columns:
        df = df.sort_values("holdingPercent", ascending=False)
    return df["symbol"].tolist()


def _load_cached_universe() -> List[str] | None:
    """Return cached holdings from disk if they exist."""
    if not UNIVERSE_CACHE.exists():
        return None
    try:
        return json.loads(UNIVERSE_CACHE.read_text())
    except Exception:
        return None


def _store_universe(symbols: List[str]) -> None:
    """Persist the downloaded holdings list for reuse."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        UNIVERSE_CACHE.write_text(json.dumps(symbols))
    except Exception:
        pass


@lru_cache(maxsize=1)
def get_top_sp500_universe(count: int, force_refresh: bool = False) -> List[str]:
    """Return up to ``count`` tickers ordered by SPY holding weight."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    symbols: List[str] | None = None
    if not force_refresh:
        symbols = _load_cached_universe()
    if symbols is None:
        symbols = _download_spy_holdings()
        if symbols:
            _store_universe(symbols)
    if not symbols:
        symbols = FALLBACK_TICKERS
    symbols = list(dict.fromkeys(symbols))
    available = min(max(count, 1), len(symbols))
    return symbols[:available]
