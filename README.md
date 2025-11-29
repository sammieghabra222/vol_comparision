# Options Volatility Lab

Visual Streamlit app to explore a stock's volatility from an options perspective. Enter a ticker to see implied volatility surfaces, expected move from the front-month straddle, historical/realized volatility, and peer comparisons.

## Features
- Expected move visualization from the nearest expiry ATM straddle (price distribution chart).
- Rolling realized volatility with quick toggles for common lookback windows.
- Implied volatility surface across expirations/strikes plus ATM term structure.
- Peer volatility comparison via user-provided tickers.

## Getting Started
1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. In the sidebar, enter a ticker (default: `AAPL`), choose the historical window, realized vol lookback, and peer list. Adjust how many expirations to load to trade off speed vs. detail.

## Notes and Data Sources
- Market and options data are pulled via Yahoo Finance (`yfinance`); an internet connection is required when running the app.
- Expected move uses the front-month ATM call/put mid-price (or last price if bid/ask unavailable).
- The IV surface heatmap averages IV by strike/expiry; the term-structure chart selects the ATM IV per expiry.
- Realized volatility is annualized log-return volatility; peer comparison uses the same lookback window you select.

## Next Ideas
- Add caching/persistence for frequently viewed tickers.
- Allow filtering to specific moneyness buckets for the IV surface.
- Include skew metrics and realized vs implied overlays.
