import math
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import (
    ExpectedMove,
    atm_iv_by_expiration,
    estimate_expected_move,
    fetch_options_surface,
    fetch_price_history,
    peer_atm_iv,
    peer_realized_vols,
    realized_vol_series,
    realized_vol_snapshot,
    skew_metrics,
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
        hover_data={
            "impliedVolatility": ":.2%",
            "bid": True,
            "ask": True,
            "volume": True,
            "openInterest": True,
        },
    )
    fig.update_layout(yaxis_title="Expiration", xaxis_title="Strike", coloraxis_colorbar_title="IV")
    return fig


def term_structure_chart(surface: pd.DataFrame, spot: float, realized_anchor: Optional[float] = None) -> go.Figure:
    if surface.empty:
        return go.Figure()
    term_df = atm_iv_by_expiration(surface, spot)
    term_df_display = term_df.copy()
    term_df_display["expiration"] = term_df_display["expiration"].dt.date
    fig = px.line(
        term_df_display,
        x="expiration",
        y="atm_iv",
        markers=True,
        title="ATM Implied Volatility Term Structure",
    )
    if realized_anchor is not None and np.isfinite(realized_anchor):
        fig.add_hline(y=realized_anchor, line_dash="dash", line_color="gray", annotation_text="Realized vol (window)")
    fig.update_layout(xaxis_title="Expiration", yaxis_title="IV (ATM)")
    return fig


def historical_vol_chart(history: pd.DataFrame, window: int, iv_overlay: Optional[float] = None) -> go.Figure:
    series = realized_vol_series(history, window=window)
    fig = px.line(series, labels={"value": "Realized Vol", "index": "Date"})
    if iv_overlay is not None and np.isfinite(iv_overlay):
        fig.add_hline(y=iv_overlay, line_dash="dash", line_color="crimson", annotation_text="Current ATM IV")
    fig.update_layout(title=f"Rolling Realized Volatility (window={window}d)", yaxis_tickformat=".0%")
    return fig


def peer_chart(peers: pd.DataFrame) -> go.Figure:
    if peers.empty:
        return go.Figure()
    peers = peers.copy()
    fig = px.bar(peers, x="ticker", y=["realized_vol", "atm_iv"], barmode="group", title="Peer Volatility")
    fig.update_layout(yaxis_tickformat=".1%", legend_title="Metric")
    return fig


def skew_chart(skews: List, title: str) -> go.Figure:
    if not skews:
        return go.Figure()
    df = pd.DataFrame(
        {
            "expiration": [s.expiration.date() for s in skews],
            "risk_reversal": [s.risk_reversal for s in skews],
            "butterfly": [s.butterfly for s in skews],
        }
    )
    fig = px.bar(df, x="expiration", y=["risk_reversal", "butterfly"], barmode="group", title=title)
    fig.update_layout(yaxis_title="IV points")
    return fig


def generate_surface_insights(term_df: pd.DataFrame, skews: List, spot: float) -> List[str]:
    insights = []
    if term_df is not None and not term_df.empty:
        front_iv = term_df["atm_iv"].iloc[0]
        insights.append(f"Front-month ATM IV: {format_pct(front_iv)}.")
    if skews:
        first = skews[0]
        if np.isfinite(first.risk_reversal):
            if first.risk_reversal > 0:
                insights.append("Upside skewed: calls trade over puts (positive risk reversal).")
            elif first.risk_reversal < 0:
                insights.append("Downside skewed: puts rich to calls (negative risk reversal).")
    if not insights:
        insights.append("Not enough data to summarize skew/term insights.")
    return insights


def term_insights(term_df: pd.DataFrame, realized_anchor: Optional[float]) -> List[str]:
    insights = []
    if term_df is None or term_df.empty:
        return ["Not enough data to read the term structure."]
    if len(term_df) >= 2:
        slope = term_df["atm_iv"].iloc[-1] - term_df["atm_iv"].iloc[0]
        if slope > 0.02:
            insights.append("Upward sloping term structure; back months carry higher IV.")
        elif slope < -0.02:
            insights.append("Inverted term structure; near-dated IV is richer than back months.")
    if realized_anchor is not None and np.isfinite(realized_anchor):
        front_iv = term_df["atm_iv"].iloc[0]
        diff = front_iv - realized_anchor
        insights.append(f"Front ATM IV vs realized: {format_pct(diff)} differential at current window.")
    return insights or ["Term structure looks flat relative to current anchor."]


def skew_insights(skews: List) -> List[str]:
    if not skews:
        return ["Skew snapshot unavailable for this name/expiry set."]
    first = skews[0]
    insights = []
    if np.isfinite(first.risk_reversal):
        if first.risk_reversal < 0:
            insights.append("Downside puts are richer than upside calls (negative risk reversal).")
        elif first.risk_reversal > 0:
            insights.append("Upside calls trade over downside puts (positive risk reversal).")
    if np.isfinite(first.butterfly):
        if first.butterfly > 0:
            insights.append("Wings price over ATM (smiley skew).")
        elif first.butterfly < 0:
            insights.append("ATM trades rich vs wings (frowny skew).")
    return insights or ["Skew appears flat near ATM."]


def main():
    st.title("Options Volatility Lab")
    st.caption("Visualize implied and historical volatility from an options perspective.")

    with st.sidebar:
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        history_period = st.selectbox("Historical window", options=["6mo", "1y", "2y", "5y"], index=1)
        realized_window = st.select_slider("Realized vol lookback (days)", options=[10, 20, 30, 60, 90], value=30)
        max_expirations = st.slider("Max expirations to load", min_value=2, max_value=20, value=10, step=1)
        peer_input = st.text_area(
            "Peer tickers (comma-separated)",
            value="MSFT, GOOGL, AMZN",
            help="Used for realized/implied volatility comparison.",
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
        st.metric(f"{ticker} Last Close", f"${spot:,.2f}")

    # Options surface and expected move
    try:
        with st.spinner("Loading options surface..."):
            surface = load_surface(ticker, max_expirations=max_expirations)
    except Exception as exc:
        st.error(f"Unable to load options surface for {ticker}: {exc}")
        surface = pd.DataFrame()
    exp_selection = None
    expected_move = None
    term_df = pd.DataFrame()
    skews = []
    atm_iv_front = None
    expirations = []
    if not surface.empty:
        expirations = sorted(surface["expiration"].unique())
        term_df = atm_iv_by_expiration(surface, spot)
        atm_iv_front = term_df["atm_iv"].iloc[0] if not term_df.empty else None
        skews = skew_metrics(surface, spot)
        iv_overlay = atm_iv_front
    else:
        iv_overlay = None

    # Historical volatility section (with optional IV overlay)
    vol_series_fig = historical_vol_chart(history, realized_window, iv_overlay=iv_overlay)
    vol_snapshot = realized_vol_snapshot(history, windows=[10, 20, 30, 60, 90, 180])
    snapshot_df = pd.DataFrame(
        [{"window": f"{w}d", "vol": vol_snapshot[w]} for w in sorted(vol_snapshot.keys())]
    )
    snapshot_df["vol_display"] = snapshot_df["vol"].apply(lambda v: format_pct(v))

    col_hist_chart, col_hist_table = st.columns([2.5, 1])
    with col_hist_chart:
        st.plotly_chart(vol_series_fig, use_container_width=True)
        st.caption("Note: Historical IV time series not provided by Yahoo Finance; overlay uses current ATM IV only.")
    with col_hist_table:
        st.subheader("Realized Volatility")
        st.dataframe(snapshot_df[["window", "vol_display"]].set_index("window"), use_container_width=True, height=260)

    col_expected, col_surface = st.columns([1.1, 1.9])
    with col_expected:
        st.subheader("Expected Move")
        if expirations:
            exp_indices = list(range(len(expirations)))
            exp_selection_idx = st.selectbox(
                "Expected move expiry",
                options=exp_indices,
                format_func=lambda i: expirations[i].date().isoformat(),
                help="Choose expiration to anchor the ATM straddle expected move.",
            )
            exp_selection = expirations[exp_selection_idx]
            expected_move = estimate_expected_move(surface, spot, expiration=exp_selection)
        if expected_move:
            st.markdown(
                f"- Expiration: **{expected_move.expiration.date()}**\n"
                f"- ATM strike: **{expected_move.strike:.2f}**\n"
                f"- Move: **${expected_move.move:,.2f}** ({format_pct(expected_move.move_pct)})"
            )
            st.plotly_chart(expected_move_chart(spot, expected_move), use_container_width=True)
            st.caption("Distribution width scales with the selected expiry's ATM straddle price (1σ range highlighted).")
        else:
            st.info("Expected move unavailable (missing ATM straddle data).")

    with col_surface:
        st.subheader("Implied Volatility Surface")
        st.plotly_chart(iv_surface_chart(surface), use_container_width=True)
        st.markdown("**Surface notes**")
        for line in generate_surface_insights(term_df, skews, spot):
            st.write(f"- {line}")
        st.plotly_chart(
            term_structure_chart(surface, spot, realized_anchor=vol_snapshot.get(realized_window)),
            use_container_width=True,
        )
        st.markdown("**Term structure insights**")
        for line in term_insights(term_df, realized_anchor=vol_snapshot.get(realized_window)):
            st.write(f"- {line}")
        if not surface.empty:
            st.plotly_chart(skew_chart(skews, "Skew (Risk Reversal / Butterfly)"), use_container_width=True)
            st.markdown("**Skew insights**")
            for line in skew_insights(skews):
                st.write(f"- {line}")
            if term_df is not None and not term_df.empty:
                iv_term_table = term_df.copy()
                iv_term_table["expiration"] = iv_term_table["expiration"].dt.date
                iv_term_table["atm_iv"] = iv_term_table["atm_iv"].apply(format_pct)
                st.dataframe(iv_term_table.set_index("expiration"), use_container_width=True, height=200)

        # Insights
        insight_lines = generate_surface_insights(term_df, skews, spot)
        st.markdown("**Insights**")
        for line in insight_lines:
            st.write(f"- {line}")

    # Option liquidity snapshot around ATM for selected expiry
    if exp_selection is not None:
        st.subheader("Bid/Ask Snapshot (near ATM)")
        exp_slice = surface[surface["expiration"] == exp_selection].copy()
        exp_slice["distance"] = (exp_slice["strike"] - spot).abs()
        exp_slice = exp_slice.sort_values("distance").head(12)
        requested_cols = ["option_type", "strike", "bid", "ask", "bidSize", "askSize", "volume", "openInterest", "impliedVolatility", "lastPrice"]
        display_cols = [c for c in requested_cols if c in exp_slice.columns]
        exp_slice = exp_slice[display_cols]
        exp_slice["impliedVolatility"] = exp_slice["impliedVolatility"].apply(format_pct)
        st.dataframe(exp_slice, use_container_width=True, height=260)

    # Peer comparison
    peer_list: List[str] = [p.strip().upper() for p in peer_input.split(",") if p.strip()]
    if peer_list:
        with st.spinner("Computing peer volatilities..."):
            peers_real = peer_realized_vols(peer_list, window=realized_window, period=history_period)
            peers_iv = peer_atm_iv(peer_list)
            peers_df = peers_real.merge(peers_iv, on="ticker", how="outer")
        st.subheader("Peer Comparison")
        st.plotly_chart(peer_chart(peers_df), use_container_width=True)
        display_df = peers_df.copy()
        display_df["realized_vol"] = display_df["realized_vol"].apply(format_pct)
        display_df["atm_iv"] = display_df["atm_iv"].apply(format_pct)
        st.dataframe(display_df.set_index("ticker"), use_container_width=True, height=240)

    st.caption(
        "Data sourced via Yahoo Finance (yfinance). Implied moves use the selected expiry ATM straddle mid-price."
    )


if __name__ == "__main__":
    main()
