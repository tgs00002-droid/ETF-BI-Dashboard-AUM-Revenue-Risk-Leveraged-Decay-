import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from io import StringIO

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="ProShares Single Stock Dashboard", layout="wide")
st.title("ProShares Single Stock ETF Dashboard (Simple)")
st.caption("Uses ProShares Single Stock list + real prices to answer performance/risk questions.")

# ----------------------------
# Minimal controls: just dates
# ----------------------------
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start date", value=pd.to_datetime("2024-01-01"))
with col2:
    end = st.date_input("End date", value=pd.to_datetime("today"))

# ProShares page you provided
PROSHARES_URL = "https://www.proshares.com/our-etfs/find-leveraged-and-inverse-etfs?etftype=&strategy=Single+Stock&benchmark=&product=Product+Overview+&search="

# Keep chart readable
TOP_N = 10  # show top 10 Single Stock funds by Net Assets (from the table)


# ----------------------------
# Load ProShares table (fix for HTTPError)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_proshares_single_stock(url: str) -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    html = r.text

    # Parse tables from HTML text (not from URL)
    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame()

    # Find the table with the product overview columns
    target = None
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if "Ticker" in cols and "Fund Name" in cols and "Fund Type" in cols:
            target = t.copy()
            break

    if target is None:
        return pd.DataFrame()

    # Filter to Single Stock
    target["Fund Type"] = target["Fund Type"].astype(str).str.strip()
    df = target[target["Fund Type"].str.lower() == "single stock"].copy()

    # Clean Net Assets for sorting
    if "Net Assets" in df.columns:
        df["Net Assets Numeric"] = (
            df["Net Assets"].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["Net Assets Numeric"] = pd.to_numeric(df["Net Assets Numeric"], errors="coerce")
    else:
        df["Net Assets Numeric"] = np.nan

    keep = [c for c in ["Ticker", "Fund Name", "Daily Objective", "Net Assets", "Net Assets Numeric", "Index/Benchmark"] if c in df.columns]
    df = df[keep].dropna(subset=["Ticker"]).reset_index(drop=True)

    # Sort biggest funds first (if available)
    if df["Net Assets Numeric"].notna().any():
        df = df.sort_values("Net Assets Numeric", ascending=False).reset_index(drop=True)

    return df


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


# ----------------------------
# Pull ProShares Single Stock list
# ----------------------------
with st.spinner("Loading ProShares Single Stock ETF list..."):
    fund_table = load_proshares_single_stock(PROSHARES_URL)

if fund_table.empty:
    st.error("Could not load the ProShares table. (Site may have changed or blocked requests.)")
    st.stop()

st.subheader("ProShares Single Stock ETFs (from ProShares)")
# show a clean table (no extra clutter)
show_cols = [c for c in ["Ticker", "Fund Name", "Daily Objective", "Net Assets", "Index/Benchmark"] if c in fund_table.columns]
st.dataframe(fund_table[show_cols], use_container_width=True, hide_index=True)

# Select top N tickers automatically
tickers = fund_table["Ticker"].head(TOP_N).astype(str).str.strip().tolist()

st.divider()

# ----------------------------
# Pull prices + compute metrics
# ----------------------------
with st.spinner("Downloading price history..."):
    prices = fetch_prices(tickers, start, end)

if prices.empty or prices.shape[0] < 10:
    st.error("Not enough price data for this date range. Try a wider range.")
    st.stop()

prices = prices.dropna(axis=1, how="all")
rets = prices.pct_change().dropna()
growth_100 = (1 + rets).cumprod() * 100

rows = []
for t in prices.columns:
    total_ret = float(growth_100[t].iloc[-1] / growth_100[t].iloc[0] - 1)
    vol = annual_volatility(rets[t])
    mdd = max_drawdown(prices[t])
    rows.append({"Ticker": t, "Total Return": total_ret, "Volatility (ann.)": vol, "Max Drawdown": mdd})

metrics = pd.DataFrame(rows)

# Key answers
best = metrics.sort_values("Total Return", ascending=False).iloc[0]
riskiest = metrics.sort_values("Volatility (ann.)", ascending=False).iloc[0]
worstdd = metrics.sort_values("Max Drawdown").iloc[0]  # most negative

st.subheader("Key Answers")
c1, c2, c3 = st.columns(3)
c1.metric("Best performer", best["Ticker"], f"{best['Total Return']:.1%}")
c2.metric("Riskiest (volatility)", riskiest["Ticker"], f"{riskiest['Volatility (ann.)']:.1%}")
c3.metric("Worst drawdown", worstdd["Ticker"], f"{worstdd['Max Drawdown']:.1%}")

st.subheader("Summary Metrics (Top Single Stock funds)")
mshow = metrics.copy()
mshow["Total Return"] = mshow["Total Return"].map(lambda x: f"{x:.2%}")
mshow["Volatility (ann.)"] = mshow["Volatility (ann.)"].map(lambda x: f"{x:.2%}")
mshow["Max Drawdown"] = mshow["Max Drawdown"].map(lambda x: f"{x:.2%}")
st.dataframe(mshow, use_container_width=True, hide_index=True)

st.subheader("Growth of $100 (Real prices)")
st.line_chart(growth_100, use_container_width=True)

st.download_button(
    "Download metrics CSV",
    data=metrics.to_csv(index=False).encode("utf-8"),
    file_name="proshares_single_stock_metrics.csv",
    mime="text/csv",
)
