import math
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import (
    ExpectedMove,
    estimate_expected_move,
    fetch_options_surface,
    fetch_price_history,
    peer_realized_vols,
    realized_vol_series,
    realized_vol_snapshot,
)


st.set_page_config(page_title="Options Volatility Lab", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load_history(ticker: str, period: str) -> pd.DataFrame:
    return fetch_price_history(ticker, period=period)


@st.cache_data(show_spinner=False)
def load_surface(ticker: str, max_expirations: int) -> pd.DataFrame:
    return fetch_options_surface(ticker, max_expirations=max_expirations)


def format_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    return f"{value:.2%}"


def expected_move_chart(spot: float, expected: ExpectedMove | None) -> go.Figure:
    if not expected or expected.move <= 0:
        return go.Figure()
    sigma = expected.move
    x_min, x_max = spot - 3 * sigma, spot + 3 * sigma
    x_vals = np.linspace(x_min, x_max, 200)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - spot) / sigma) ** 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=pdf, mode="lines", fill="tozeroy", name="Expected distribution"))
    fig.add_vrect(x0=spot - sigma, x1=spot + sigma, fillcolor="LightSkyBlue", opacity=0.3, line_width=0, name="±1σ")
    fig.add_vline(x=spot, line_dash="dash", line_color="black", name="Spot")
    fig.update_layout(
        title=f"Expected Move into {expected.expiration.date()}",
        xaxis_title="Price",
        yaxis_title="Density (scaled)",
        showlegend=False,
    )
    return fig


def iv_surface_chart(surface: pd.DataFrame) -> go.Figure:
    if surface.empty:
        return go.Figure()
    df = surface.copy()
    df["expiration"] = df["expiration"].dt.date
    fig = px.density_heatmap(
        df,
        x="strike",
        y="expiration",
        z="impliedVolatility",
        histfunc="avg",
        color_continuous_scale="Viridis",
        title="Implied Volatility Surface (avg by strike/expiry)",
    )
    fig.update_layout(yaxis_title="Expiration", xaxis_title="Strike", coloraxis_colorbar_title="IV")
    return fig


def term_structure_chart(surface: pd.DataFrame, spot: float) -> go.Figure:
    if surface.empty:
        return go.Figure()
    rows = []
    for exp, group in surface.groupby(surface["expiration"]):
        group = group.sort_values("strike")
        group["distance"] = (group["strike"] - spot).abs()
        atm_row = group.loc[group["distance"].idxmin()]
        rows.append({"expiration": exp.date(), "iv": atm_row["impliedVolatility"]})
    term_df = pd.DataFrame(rows).sort_values("expiration")
    fig = px.line(term_df, x="expiration", y="iv", markers=True, title="ATM Implied Volatility Term Structure")
    fig.update_layout(xaxis_title="Expiration", yaxis_title="IV (ATM)")
    return fig


def historical_vol_chart(history: pd.DataFrame, window: int) -> go.Figure:
    series = realized_vol_series(history, window=window)
    fig = px.line(series, labels={"value": "Realized Vol", "index": "Date"})
    fig.update_layout(title=f"Rolling Realized Volatility (window={window}d)", yaxis_tickformat=".0%")
    return fig


def peer_chart(peers: pd.DataFrame) -> go.Figure:
    if peers.empty:
        return go.Figure()
    fig = px.bar(
        peers.sort_values("realized_vol", ascending=False),
        x="ticker",
        y="realized_vol",
        title="Peer Realized Volatility",
    )
    fig.update_layout(yaxis_tickformat=".1%")
    return fig


def main():
    st.title("Options Volatility Lab")
    st.caption("Visualize implied and historical volatility from an options perspective.")

    with st.sidebar:
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        history_period = st.selectbox("Historical window", options=["6mo", "1y", "2y", "5y"], index=1)
        realized_window = st.select_slider("Realized vol lookback (days)", options=[10, 20, 30, 60, 90], value=30)
        max_expirations = st.slider("Max expirations to load", min_value=2, max_value=12, value=6, step=1)
        peer_input = st.text_area(
            "Peer tickers (comma-separated)",
            value="MSFT, GOOGL, AMZN",
            help="Used for realized volatility comparison.",
        )

    if not ticker:
        st.info("Enter a ticker to begin.")
        return

    col_status, col_price = st.columns([2, 1])
    try:
        with col_status, st.spinner("Loading price history..."):
            history = load_history(ticker, period=history_period)
        spot = float(history["Close"].iloc[-1])
    except Exception as exc:
        st.error(f"Unable to load history for {ticker}: {exc}")
        return

    with col_price:
        st.metric("Last Close", f"${spot:,.2f}")

    # Historical volatility section
    vol_series_fig = historical_vol_chart(history, realized_window)
    vol_snapshot = realized_vol_snapshot(history, windows=[10, 20, 30, 60, 90, 180])
    snapshot_df = pd.DataFrame(
        [{"window": f"{w}d", "vol": vol_snapshot[w]} for w in sorted(vol_snapshot.keys())]
    )
    snapshot_df["vol_display"] = snapshot_df["vol"].apply(lambda v: format_pct(v))

    col_hist_chart, col_hist_table = st.columns([2.5, 1])
    with col_hist_chart:
        st.plotly_chart(vol_series_fig, use_container_width=True)
    with col_hist_table:
        st.subheader("Realized Volatility")
        st.dataframe(snapshot_df[["window", "vol_display"]].set_index("window"), use_container_width=True, height=260)

    # Options surface and expected move
    try:
        with st.spinner("Loading options surface..."):
            surface = load_surface(ticker, max_expirations=max_expirations)
    except Exception as exc:
        st.error(f"Unable to load options surface for {ticker}: {exc}")
        surface = pd.DataFrame()

    expected_move = None
    if not surface.empty:
        expected_move = estimate_expected_move(surface, spot)

    col_expected, col_surface = st.columns([1.1, 1.9])
    with col_expected:
        st.subheader("Expected Move")
        if expected_move:
            st.markdown(
                f"- Expiration: **{expected_move.expiration.date()}**\n"
                f"- ATM strike: **{expected_move.strike:.2f}**\n"
                f"- Move: **${expected_move.move:,.2f}** ({format_pct(expected_move.move_pct)})"
            )
            st.plotly_chart(expected_move_chart(spot, expected_move), use_container_width=True)
        else:
            st.info("Expected move unavailable (missing ATM straddle data).")

    with col_surface:
        st.subheader("Implied Volatility Surface")
        st.plotly_chart(iv_surface_chart(surface), use_container_width=True)
        st.plotly_chart(term_structure_chart(surface, spot), use_container_width=True)

    # Peer comparison
    peer_list: List[str] = [p.strip().upper() for p in peer_input.split(",") if p.strip()]
    if peer_list:
        with st.spinner("Computing peer volatilities..."):
            peers_df = peer_realized_vols(peer_list, window=realized_window, period=history_period)
        st.subheader("Peer Comparison")
        st.plotly_chart(peer_chart(peers_df), use_container_width=True)
        st.dataframe(peers_df.set_index("ticker"), use_container_width=True, height=240)

    st.caption(
        "Data sourced via Yahoo Finance (yfinance). Implied moves use the nearest expiry ATM straddle mid-price."
    )


if __name__ == "__main__":
    main()
