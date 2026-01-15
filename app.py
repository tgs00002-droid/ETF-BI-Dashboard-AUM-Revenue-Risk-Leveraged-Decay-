import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# -----------------------------------
# App setup
# -----------------------------------
st.set_page_config(page_title="ProShares Single Stock ETF Dashboard", layout="wide")
st.title("ProShares Single Stock ETF Dashboard (Simple)")
st.caption("Pulls the official ProShares table → filters to Single Stock → charts performance/risk using real prices.")

# -----------------------------------
# Only 2 simple controls
# -----------------------------------
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start date", value=pd.to_datetime("2024-01-01"))
with col2:
    end = st.date_input("End date", value=pd.to_datetime("today"))

TOP_N = 8  # keep visuals clean

# -----------------------------------
# Source: ProShares screener table
# (We scrape the Product Overview table and filter Fund Type == "Single Stock")
# -----------------------------------
PROSHARES_URL = "https://www.proshares.com/our-etfs/find-leveraged-and-inverse-etfs?benchmark=&etftype=&product=Product+Overview+&search=&strategy=Single+Stock"

@st.cache_data(show_spinner=False)
def load_proshares_single_stock(url: str) -> pd.DataFrame:
    # read_html finds tables in the page HTML
    tables = pd.read_html(url)
    if not tables:
        return pd.DataFrame()

    # The big product table is typically the one with "Ticker" / "Fund Name"
    # We'll search for a table containing those columns.
    target = None
    for t in tables:
        cols = [c.strip() for c in t.columns.astype(str).tolist()]
        if "Ticker" in cols and "Fund Name" in cols:
            target = t.copy()
            break

    if target is None:
        return pd.DataFrame()

    # Normalize column names
    target.columns = [str(c).strip() for c in target.columns]

    # Filter to Single Stock
    if "Fund Type" not in target.columns:
        return pd.DataFrame()

    df = target[target["Fund Type"].astype(str).str.strip().str.lower() == "single stock"].copy()

    # Clean up Net Assets if present
    if "Net Assets" in df.columns:
        # Convert "$1,234,567" → 1234567
        df["Net Assets Numeric"] = (
            df["Net Assets"].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["Net Assets Numeric"] = pd.to_numeric(df["Net Assets Numeric"], errors="coerce")
    else:
        df["Net Assets Numeric"] = np.nan

    # Keep only useful columns if they exist
    keep_cols = [c for c in ["Ticker", "Fund Name", "Daily Objective", "Net Assets", "Net Assets Numeric", "Index/Benchmark"] if c in df.columns]
    df = df[keep_cols].dropna(subset=["Ticker"])

    # Sort biggest funds first (if we have net assets)
    if df["Net Assets Numeric"].notna().any():
        df = df.sort_values("Net Assets Numeric", ascending=False)

    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def fetch_prices(tickers, start_date, end_date) -> pd.DataFrame:
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"].copy()
    else:
        prices = data[["Adj Close"]].copy()
        prices.columns = tickers
    return prices.dropna(how="all")

def annual_volatility(ret_series: pd.Series) -> float:
    return float(ret_series.std() * np.sqrt(252))

def max_drawdown(price_series: pd.Series) -> float:
    peak = price_series.cummax()
    dd = price_series / peak - 1.0
    return float(dd.min())

# -----------------------------------
# Load ProShares table
# -----------------------------------
with st.spinner("Loading ProShares Single Stock ETF list..."):
    fund_table = load_proshares_single_stock(PROSHARES_URL)

if fund_table.empty:
    st.error("Could not load the ProShares table (site layout may have changed). Try again later.")
    st.stop()

st.subheader("ProShares Single Stock ETFs (from ProShares table)")
st.dataframe(
    fund_table.drop(columns=["Net Assets Numeric"], errors="ignore"),
    use_container_width=True,
    hide_index=True
)

# Pick top N tickers automatically (no user editing)
tickers = fund_table["Ticker"].head(TOP_N).astype(str).str.strip().tolist()

st.divider()
st.subheader(f"Charts for Top {min(TOP_N, len(tickers))} funds (by Net Assets when available)")

with st.spinner("Downloading price history for charting..."):
    prices = fetch_prices(tickers, start, end)

if prices.empty or prices.shape[0] < 10:
    st.error("Not enough price data for the selected date range. Try a wider range.")
    st.stop()

prices = prices.dropna(axis=1, how="all")
rets = prices.pct_change().dropna()
growth_100 = (1 + rets).cumprod() * 100

# -----------------------------------
# Metrics + “answers”
# -----------------------------------
rows = []
for t in prices.columns:
    total_ret = float(growth_100[t].iloc[-1] / growth_100[t].iloc[0] - 1)
    vol = annual_volatility(rets[t])
    mdd = max_drawdown(prices[t])
    rows.append({"Ticker": t, "Total Return": total_ret, "Volatility (ann.)": vol, "Max Drawdown": mdd})
metrics = pd.DataFrame(rows)

best = metrics.sort_values("Total Return", ascending=False).iloc[0]
riskiest = metrics.sort_values("Volatility (ann.)", ascending=False).iloc[0]
worstdd = metrics.sort_values("Max Drawdown").iloc[0]  # most negative

c1, c2, c3 = st.columns(3)
c1.metric("Best performer", best["Ticker"], f"{best['Total Return']:.1%}")
c2.metric("Riskiest (volatility)", riskiest["Ticker"], f"{riskiest['Volatility (ann.)']:.1%}")
c3.metric("Worst drawdown", worstdd["Ticker"], f"{worstdd['Max Drawdown']:.1%}")

st.subheader("Summary Metrics (Top funds)")
show = metrics.copy()
show["Total Return"] = show["Total Return"].map(lambda x: f"{x:.2%}")
show["Volatility (ann.)"] = show["Volatility (ann.)"].map(lambda x: f"{x:.2%}")
show["Max Drawdown"] = show["Max Drawdown"].map(lambda x: f"{x:.2%}")
st.dataframe(show, use_container_width=True, hide_index=True)

st.subheader("Growth of $100 (Real prices)")
st.line_chart(growth_100, use_container_width=True)

st.download_button(
    "Download metrics CSV",
    data=metrics.to_csv(index=False).encode("utf-8"),
    file_name="proshares_single_stock_metrics.csv",
    mime="text/csv",
)
