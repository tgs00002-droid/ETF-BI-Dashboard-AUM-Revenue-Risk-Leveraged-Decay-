import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Simple ETF Dashboard", layout="wide")
st.title("Simple ETF Dashboard (Performance • Risk • Drawdown • Daily Reset)")

st.caption(
    "Uses real adjusted close prices. Answers: best performer, riskiest, worst drawdown, "
    "and a simple daily-reset leverage illustration."
)

# ----------------------------
# Controls (minimal)
# ----------------------------
# Keep it simple: only date range.
colA, colB = st.columns(2)
with colA:
    start = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
with colB:
    end = st.date_input("End date", value=pd.to_datetime("today"))

# Fixed tickers (simple + relevant)
# SPY benchmark + ProShares examples + Nasdaq for variety
TICKERS = ["SPY", "QQQ", "SSO", "SH", "TLT"]  # ProShares: SSO (2x), SH (-1x)
BENCH = "SPY"

st.write(f"**Tracking:** {', '.join(TICKERS)}  |  **Benchmark:** {BENCH}")

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
        # single ticker case
        prices = data[["Adj Close"]].copy()
        prices.columns = tickers
    prices = prices.dropna(how="all")
    return prices

with st.spinner("Downloading price data..."):
    prices = fetch_prices(TICKERS, start, end)

if prices.empty or prices.shape[0] < 5:
    st.error("Not enough data returned. Try a wider date range.")
    st.stop()

# Ensure order and remove missing columns (if any ticker fails)
prices = prices[[t for t in TICKERS if t in prices.columns]].dropna(how="all")
rets = prices.pct_change().dropna()

# ----------------------------
# Metrics (real)
# ----------------------------
def total_return(series: pd.Series) -> float:
    return float(series.iloc[-1] / series.iloc[0] - 1)

def annual_volatility(ret_series: pd.Series) -> float:
    return float(ret_series.std() * np.sqrt(252))

def max_drawdown(price_series: pd.Series) -> float:
    peak = price_series.cummax()
    dd = price_series / peak - 1.0
    return float(dd.min())

growth_100 = (1 + rets).cumprod() * 100

metrics = []
for t in prices.columns:
    metrics.append({
        "Ticker": t,
        "Total Return": total_return(growth_100[t]),
        "Volatility (ann.)": annual_volatility(rets[t]),
        "Max Drawdown": max_drawdown(prices[t]),
    })
metrics_df = pd.DataFrame(metrics)

# "Answers"
best = metrics_df.sort_values("Total Return", ascending=False).iloc[0]
risk = metrics_df.sort_values("Volatility (ann.)", ascending=False).iloc[0]
worst_dd = metrics_df.sort_values("Max Drawdown").iloc[0]  # most negative = worst

# ----------------------------
# Top answers (clean)
# ----------------------------
st.subheader("Key Answers")
c1, c2, c3 = st.columns(3)
c1.metric("Best performer", f"{best['Ticker']}", f"{best['Total Return']:.1%}")
c2.metric("Riskiest (highest volatility)", f"{risk['Ticker']}", f"{risk['Volatility (ann.)']:.1%}")
c3.metric("Worst drawdown", f"{worst_dd['Ticker']}", f"{worst_dd['Max Drawdown']:.1%}")

st.divider()

# ----------------------------
# Simple table (no clutter)
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
st.subheader("Growth of $100 (Real prices)")
st.line_chart(growth_100, use_container_width=True)

st.divider()

# ----------------------------
# Daily reset illustration (simple + clear)
# Uses benchmark returns to simulate what 2x and -1x *would* do with daily reset.
# This illustrates the concept without extra controls.
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
    "What this shows: leveraged/inverse targets are DAILY. Over longer, volatile periods, compounding can make results "
    "drift from what people expect if they think in long-term multiples. That’s the key 'daily reset' concept."
)

# ----------------------------
# Download
# ----------------------------
st.download_button(
    "Download metrics CSV",
    data=metrics_df.to_csv(index=False).encode("utf-8"),
    file_name="etf_metrics.csv",
    mime="text/csv",
)
