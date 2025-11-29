"""
Data utilities for fetching market data and computing volatility metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class SkewSnapshot:
    expiration: pd.Timestamp
    atm_iv: float
    downside_iv: float
    upside_iv: float
    risk_reversal: float
    butterfly: float


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


def _atm_row(surface: pd.DataFrame, spot: float, expiration: pd.Timestamp) -> Optional[pd.Series]:
    exp_slice = surface[surface["expiration"] == expiration]
    if exp_slice.empty:
        return None
    exp_slice = exp_slice.sort_values("strike")
    exp_slice["distance"] = (exp_slice["strike"] - spot).abs()
    idx = exp_slice["distance"].idxmin()
    return exp_slice.loc[idx]


def estimate_expected_move(
    surface: pd.DataFrame, spot: float, expiration: Optional[pd.Timestamp] = None
) -> Optional[ExpectedMove]:
    """
    Use the specified or nearest expiration ATM straddle to estimate expected move.
    Returns None if we cannot compute.
    """
    if surface.empty:
        return None
    if expiration is None:
        expiration = surface["expiration"].min()

    exp_slice = surface[surface["expiration"] == expiration]
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
    return ExpectedMove(expiration=expiration, strike=float(atm_strike), move=move, move_pct=move_pct)


def atm_iv_by_expiration(surface: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Return ATM IV per expiration."""
    rows = []
    for exp, group in surface.groupby(surface["expiration"]):
        atm_row = _atm_row(surface, spot, exp)
        if atm_row is None:
            continue
        rows.append({"expiration": exp, "atm_iv": atm_row["impliedVolatility"]})
    return pd.DataFrame(rows).sort_values("expiration")


def skew_metrics(surface: pd.DataFrame, spot: float) -> List[SkewSnapshot]:
    """
    Approximate skew using +/-10% moneyness IVs for the earliest expirations.
    Risk reversal = upside IV - downside IV
    Butterfly = (upside + downside - 2*ATM)
    """
    snapshots: List[SkewSnapshot] = []
    if surface.empty:
        return snapshots
    for exp in sorted(surface["expiration"].unique())[:4]:
        slice_exp = surface[surface["expiration"] == exp].copy()
        if slice_exp.empty:
            continue
        atm = _atm_row(surface, spot, exp)
        if atm is None:
            continue
        downside = (
            slice_exp[slice_exp["strike"] <= spot * 0.9]
            .sort_values("strike", ascending=False)
            .head(1)
        )
        upside = (
            slice_exp[slice_exp["strike"] >= spot * 1.1]
            .sort_values("strike", ascending=True)
            .head(1)
        )
        downside_iv = downside["impliedVolatility"].iloc[0] if not downside.empty else np.nan
        upside_iv = upside["impliedVolatility"].iloc[0] if not upside.empty else np.nan
        atm_iv = atm["impliedVolatility"]
        risk_reversal = upside_iv - downside_iv if np.isfinite(upside_iv) and np.isfinite(downside_iv) else np.nan
        butterfly = (upside_iv + downside_iv - 2 * atm_iv) if np.isfinite(upside_iv) and np.isfinite(downside_iv) else np.nan
        snapshots.append(
            SkewSnapshot(
                expiration=exp,
                atm_iv=float(atm_iv),
                downside_iv=float(downside_iv) if np.isfinite(downside_iv) else np.nan,
                upside_iv=float(upside_iv) if np.isfinite(upside_iv) else np.nan,
                risk_reversal=float(risk_reversal) if np.isfinite(risk_reversal) else np.nan,
                butterfly=float(butterfly) if np.isfinite(butterfly) else np.nan,
            )
        )
    return snapshots


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


def peer_atm_iv(
    tickers: Iterable[str],
    spot_map: Optional[dict] = None,
) -> pd.DataFrame:
    """Approximate ATM implied volatility for each peer using nearest expiration."""
    rows = []
    for peer in tickers:
        peer = peer.strip().upper()
        if not peer:
            continue
        try:
            spot = None
            if spot_map and peer in spot_map:
                spot = spot_map[peer]
            if spot is None:
                hist = fetch_price_history(peer, period="1mo")
                spot = float(hist["Close"].iloc[-1])
            surface = fetch_options_surface(peer, max_expirations=3)
            iv_df = atm_iv_by_expiration(surface, spot)
            atm_iv = iv_df["atm_iv"].iloc[0] if not iv_df.empty else np.nan
            rows.append({"ticker": peer, "atm_iv": atm_iv})
        except Exception:
            rows.append({"ticker": peer, "atm_iv": np.nan})
    if not rows:
        return pd.DataFrame(columns=["ticker", "atm_iv"])
    return pd.DataFrame(rows)
