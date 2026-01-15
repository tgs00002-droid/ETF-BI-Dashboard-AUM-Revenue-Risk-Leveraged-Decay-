import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="ProShares ETF Dashboard", layout="wide")
st.title("ProShares ETF Dashboard (Performance • Risk • Drawdown • Daily Reset)")
st.caption(
    "Uses real adjusted close prices for ProShares tickers only. "
    "Answers: best performer, riskiest, worst drawdown, and a simple daily-reset illustration."
)

# ----------------------------
# Minimal controls
# ----------------------------
colA, colB = st.columns(2)
with colA:
    start = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
with colB:
    end = st.date_input("End date", value=pd.to_datetime("today"))

# ProShares-only universe (coherent set)
TICKERS = ["NOBL", "SSO", "UPRO", "SH", "SDS", "SPXU"]
BENCH = "NOBL"  # ProShares benchmark-like equity ETF

st.write(f"**Tickers:** {', '.join(TICKERS)}  |  **Benchmark (ProShares):** {BENCH}")

# ----------------------------
# Data fetch
# ----------------------------
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
    prices = prices.dropna(how="all")
    return prices

with st.spinner("Downloading price data..."):
    prices = fetch_prices(TICKERS, start, end)

if prices.empty or prices.shape[0] < 10:
    st.error("Not enough data returned. Try a wider date range (example: 2020-01-01 to today).")
    st.stop()

# Keep only tickers that actually returned data
available = [t for t in TICKERS if t in prices.columns and prices[t].notna().sum() > 10]
prices = prices[available].dropna(how="all")

if BENCH not in prices.columns:
    st.error(f"Benchmark {BENCH} has no data in this date range. Try a wider range.")
    st.stop()

rets = prices.pct_change().dropna()

# ----------------------------
# Metrics
# ----------------------------
def annual_volatility(ret_series: pd.Series) -> float:
    return float(ret_series.std() * np.sqrt(252))

def max_drawdown(price_series: pd.Series) -> float:
    peak = price_series.cummax()
    dd = price_series / peak - 1.0
    return float(dd.min())

growth_100 = (1 + rets).cumprod() * 100

metrics = []
for t in prices.columns:
    total_ret = float(growth_100[t].iloc[-1] / growth_100[t].iloc[0] - 1)
    vol = annual_volatility(rets[t])
    mdd = max_drawdown(prices[t])
    metrics.append({"Ticker": t, "Total Return": total_ret, "Volatility (ann.)": vol, "Max Drawdown": mdd})

metrics_df = pd.DataFrame(metrics)

# Key answers
best = metrics_df.sort_values("Total Return", ascending=False).iloc[0]
risk = metrics_df.sort_values("Volatility (ann.)", ascending=False).iloc[0]
worst_dd = metrics_df.sort_values("Max Drawdown").iloc[0]  # most negative

# ----------------------------
# Key answers (clean)
# ----------------------------
st.subheader("Key Answers")
c1, c2, c3 = st.columns(3)
c1.metric("Best performer", f"{best['Ticker']}", f"{best['Total Return']:.1%}")
c2.metric("Riskiest (highest volatility)", f"{risk['Ticker']}", f"{risk['Volatility (ann.)']:.1%}")
c3.metric("Worst drawdown", f"{worst_dd['Ticker']}", f"{worst_dd['Max Drawdown']:.1%}")

st.divider()

# ----------------------------
# Simple table
# ----------------------------
st.subheader("Summary Table")
show = metrics_df.copy()
show["Total Return"] = show["Total Return"].map(lambda x: f"{x:.2%}")
show["Volatility (ann.)"] = show["Volatility (ann.)"].map(lambda x: f"{x:.2%}")
show["Max Drawdown"] = show["Max Drawdown"].map(lambda x: f"{x:.2%}")
st.dataframe(show, use_container_width=True, hide_index=True)

# ----------------------------
# Graph 1: Growth of $100
# ----------------------------
st.subheader("Growth of $100 (ProShares tickers)")
st.line_chart(growth_100, use_container_width=True)

st.divider()

# ----------------------------
# Daily reset illustration (simple)
# Use benchmark daily returns to simulate what 2x and -1x daily-reset returns would look like
# ----------------------------
st.subheader("Daily Reset Illustration (Benchmark vs 2x vs -1x)")

bench_rets = rets[BENCH].dropna()

def simulate_daily_reset(bench_returns: pd.Series, leverage: float, start_value: float = 100.0) -> pd.Series:
    vals = [start_value]
    idx = bench_returns.index
    for r in bench_returns.values:
        vals.append(vals[-1] * (1 + leverage * r))
    return pd.Series(vals[1:], index=idx)

sim_2x = simulate_daily_reset(bench_rets, 2.0, 100.0)
sim_m1 = simulate_daily_reset(bench_rets, -1.0, 100.0)

compare = pd.DataFrame({
    f"{BENCH} Benchmark": growth_100[BENCH].reindex(sim_2x.index),
    "Simulated 2x (daily reset)": sim_2x,
    "Simulated -1x (daily reset)": sim_m1,
}).dropna()

st.line_chart(compare, use_container_width=True)

st.info(
    "What this means (simple): leveraged/inverse ETFs target DAILY returns. "
    "Over time, compounding can create drift, especially when markets bounce around."
)

# ----------------------------
# Download
# ----------------------------
st.download_button(
    "Download metrics CSV",
    data=metrics_df.to_csv(index=False).encode("utf-8"),
    file_name="proshares_metrics.csv",
    mime="text/csv",
)
