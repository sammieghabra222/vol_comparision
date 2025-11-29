"""
Data utilities for fetching market data and computing volatility metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

ANNUALIZATION_FACTOR = np.sqrt(252)


@dataclass
class ExpectedMove:
    expiration: pd.Timestamp
    strike: float
    move: float
    move_pct: float


def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Pull historical price data."""
    ticker_obj = yf.Ticker(ticker)
    history = ticker_obj.history(period=period, auto_adjust=False)
    if history.empty:
        raise ValueError(f"No price history found for {ticker}")
    history.index = history.index.tz_localize(None)
    return history


def realized_vol_series(history: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling realized volatility series (annualized)."""
    log_returns = np.log(history["Close"]).diff().dropna()
    vol = log_returns.rolling(window).std() * ANNUALIZATION_FACTOR
    return vol.dropna()


def realized_vol_snapshot(history: pd.DataFrame, windows: Iterable[int]) -> dict:
    """Single realized volatility number per window."""
    log_returns = np.log(history["Close"]).diff().dropna()
    snapshot = {}
    for window in windows:
        if len(log_returns) < window:
            snapshot[window] = np.nan
            continue
        snapshot[window] = log_returns.tail(window).std() * ANNUALIZATION_FACTOR
    return snapshot


def fetch_options_surface(
    ticker: str, max_expirations: int = 6
) -> pd.DataFrame:
    """
    Fetch options chains across expirations and return a unified surface.
    max_expirations limits requests to keep things responsive.
    """
    ticker_obj = yf.Ticker(ticker)
    expirations = ticker_obj.options
    if not expirations:
        raise ValueError(f"No listed options for {ticker}")

    surface_frames: List[pd.DataFrame] = []
    for exp in expirations[:max_expirations]:
        try:
            chain = ticker_obj.option_chain(exp)
        except Exception:
            continue
        for option_type, df in (("call", chain.calls), ("put", chain.puts)):
            if df.empty or "impliedVolatility" not in df:
                continue
            frame = df.assign(
                expiration=pd.to_datetime(exp),
                option_type=option_type,
            )
            surface_frames.append(frame)

    if not surface_frames:
        raise ValueError(f"Unable to build options surface for {ticker}")

    surface = pd.concat(surface_frames, ignore_index=True)
    return surface


def _mid_from_row(row: pd.Series) -> float:
    """Prefer bid/ask mid; fall back to last price."""
    bid = row.get("bid")
    ask = row.get("ask")
    last = row.get("lastPrice")
    if pd.notnull(bid) and pd.notnull(ask) and ask > 0:
        return float((bid + ask) / 2)
    if pd.notnull(last):
        return float(last)
    return np.nan


def estimate_expected_move(surface: pd.DataFrame, spot: float) -> Optional[ExpectedMove]:
    """
    Use the nearest expiration ATM straddle to estimate expected move.
    Returns None if we cannot compute.
    """
    if surface.empty:
        return None
    nearest_exp = surface["expiration"].min()
    exp_slice = surface[surface["expiration"] == nearest_exp]
    if exp_slice.empty:
        return None

    exp_slice = exp_slice.sort_values("strike")
    exp_slice["distance"] = (exp_slice["strike"] - spot).abs()
    atm_strike = exp_slice.loc[exp_slice["distance"].idxmin(), "strike"]

    call_row = (
        exp_slice[(exp_slice["option_type"] == "call") & (exp_slice["strike"] == atm_strike)]
        .head(1)
    )
    put_row = (
        exp_slice[(exp_slice["option_type"] == "put") & (exp_slice["strike"] == atm_strike)]
        .head(1)
    )
    if call_row.empty or put_row.empty:
        return None

    call_mid = _mid_from_row(call_row.iloc[0])
    put_mid = _mid_from_row(put_row.iloc[0])
    if not np.isfinite(call_mid) or not np.isfinite(put_mid):
        return None

    move = call_mid + put_mid
    move_pct = move / spot if spot else np.nan
    return ExpectedMove(expiration=nearest_exp, strike=float(atm_strike), move=move, move_pct=move_pct)


def peer_realized_vols(
    tickers: Iterable[str],
    window: int = 20,
    period: str = "1y",
) -> pd.DataFrame:
    """Compute realized volatility for a list of peer tickers."""
    rows = []
    for peer in tickers:
        peer = peer.strip().upper()
        if not peer:
            continue
        try:
            history = fetch_price_history(peer, period=period)
            vol_value = realized_vol_snapshot(history, [window])[window]
            rows.append({"ticker": peer, "realized_vol": vol_value})
        except Exception:
            rows.append({"ticker": peer, "realized_vol": np.nan})
    if not rows:
        return pd.DataFrame(columns=["ticker", "realized_vol"])
    return pd.DataFrame(rows)
