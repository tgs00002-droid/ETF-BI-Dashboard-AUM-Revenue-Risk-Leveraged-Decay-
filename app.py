import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Simple ETF BI Dashboard", layout="wide")
st.title("Simple ETF BI Dashboard (ProShares-style)")
st.caption("Answers: performance, risk, estimated fee revenue, and daily-reset leverage effect.")

# -------------------------
# 1) Pick tickers (simple)
# -------------------------
st.sidebar.header("Settings")

default = "SPY, QQQ, TLT, SSO, SH"  # SPY/QQQ/TLT + ProShares examples (SSO, SH)
tickers = st.sidebar.text_input("Tickers (comma-separated)", default)
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

benchmark = st.sidebar.selectbox("Benchmark", tickers, index=0)
start = st.sidebar.date_input("Start date", value=pd.to_datetime("2021-01-01"))
end = st.sidebar.date_input("End date", value=pd.to_datetime("today"))

# -------------------------
# 2) Simple AUM + fee assumptions
# -------------------------
st.sidebar.header("AUM & Expense Ratio (simple inputs)")
st.sidebar.caption("These are estimates for the BI model. You can edit anytime.")

# Simple default assumptions (you can change)
defaults = {
    "SPY": (500e9, 0.0009),
    "QQQ": (250e9, 0.0020),
    "TLT": (40e9, 0.0015),
    "SSO": (2e9, 0.0091),
    "SH":  (3e9, 0.0089),
}

aum = {}
fee = {}
for t in tickers:
    a0, f0 = defaults.get(t, (1e9, 0.0050))
    aum[t] = st.sidebar.number_input(f"{t} AUM ($)", value=float(a0), step=1e8, format="%.0f")
    fee[t] = st.sidebar.number_input(f"{t} Expense Ratio (decimal)", value=float(f0), step=0.0001, format="%.4f")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def get_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)
    if data.empty:
        return pd.DataFrame()
    prices = data["Adj Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Adj Close"]]
    if not isinstance(prices, pd.DataFrame):
        prices = prices.to_frame()
    prices = prices.dropna(how="all")
    return prices

def calc_metrics(prices, benchmark_ticker):
    rets = prices.pct_change().dropna()
    growth = (1 + rets).cumprod() * 100

    # Performance (total return)
    total_return = (growth.iloc[-1] / growth.iloc[0]) - 1

    # Risk (annualized volatility)
    vol = rets.std() * np.sqrt(252)

    # Max drawdown
    peak = prices.cummax()
    dd = prices / peak - 1
    mdd = dd.min()

    # Tracking error vs benchmark
    bench = rets[benchmark_ticker]
    te = {}
    for c in rets.columns:
        diff = (rets[c] - bench).dropna()
        te[c] = float(diff.std() * np.sqrt(252)) if len(diff) else np.nan
    te = pd.Series(te)

    return rets, growth, total_return, vol, mdd, te

def simulate_daily_reset(bench_returns, leverage, start_value=100):
    vals = [start_value]
    for r in bench_returns.dropna():
        vals.append(vals[-1] * (1 + leverage * r))
    return pd.Series(vals[1:], index=bench_returns.dropna().index)

# -------------------------
# 3) Load data
# -------------------------
with st.spinner("Downloading price data..."):
    prices = get_prices(tickers, start, end)

if prices.empty:
    st.error("No data returned. Check tickers or date range.")
    st.stop()

# Make sure all tickers exist
prices = prices[[c for c in tickers if c in prices.columns]]

rets, growth, total_return, vol, mdd, te = calc_metrics(prices, benchmark)

# -------------------------
# 4) Executive summary table (answers the key questions)
# -------------------------
summary = pd.DataFrame({
    "AUM ($)": pd.Series({t: aum[t] for t in prices.columns}),
    "Expense Ratio": pd.Series({t: fee[t] for t in prices.columns}),
    "Est. Fee Revenue ($/yr)": pd.Series({t: aum[t] * fee[t] for t in prices.columns}),
    "Total Return": total_return,
    "Volatility (ann.)": vol,
    "Max Drawdown": mdd,
    f"Tracking Error vs {benchmark}": te
}).reset_index().rename(columns={"index": "Ticker"})

# Rank-style “answer” columns
best_perf = summary.sort_values("Total Return", ascending=False).iloc[0]
most_risk = summary.sort_values("Volatility (ann.)", ascending=False).iloc[0]
most_rev = summary.sort_values("Est. Fee Revenue ($/yr)", ascending=False).iloc[0]

# -------------------------
# 5) Top answers (super simple)
# -------------------------
st.subheader("Key Answers (what matters)")
c1, c2, c3 = st.columns(3)
c1.metric("Best performance", f"{best_perf['Ticker']} ({best_perf['Total Return']:.1%})")
c2.metric("Highest risk (volatility)", f"{most_risk['Ticker']} ({most_risk['Volatility (ann.)']:.1%})")
c3.metric("Highest fee revenue (model)", f"{most_rev['Ticker']} (${most_rev['Est. Fee Revenue ($/yr)']:,.0f}/yr)")

st.divider()

# -------------------------
# 6) Simple table + chart
# -------------------------
st.subheader("Executive Summary Table")
show = summary.copy()
show["AUM ($)"] = show["AUM ($)"].map(lambda x: f"${x:,.0f}")
show["Est. Fee Revenue ($/yr)"] = show["Est. Fee Revenue ($/yr)"].map(lambda x: f"${x:,.0f}")
show["Expense Ratio"] = show["Expense Ratio"].map(lambda x: f"{x:.4f}")
show["Total Return"] = show["Total Return"].map(lambda x: f"{x:.2%}")
show["Volatility (ann.)"] = show["Volatility (ann.)"].map(lambda x: f"{x:.2%}")
show["Max Drawdown"] = show["Max Drawdown"].map(lambda x: f"{x:.2%}")
show[f"Tracking Error vs {benchmark}"] = show[f"Tracking Error vs {benchmark}"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "NA")
st.dataframe(show, use_container_width=True)

st.subheader("Growth of $100")
st.line_chart(growth, use_container_width=True)

st.divider()

# -------------------------
# 7) Simple daily reset simulation (one chart, one message)
# -------------------------
st.subheader("Daily Reset (why leverage can decay)")
lev = st.slider("Leverage for simulation (uses benchmark daily returns)", -3.0, 3.0, 2.0, 0.5)

bench_rets = rets[benchmark]
sim = simulate_daily_reset(bench_rets, leverage=lev, start_value=100)

compare = pd.DataFrame({
    f"Simulated {lev}x (Daily Reset)": sim,
    f"{benchmark} Benchmark": growth[benchmark].reindex(sim.index)
}).dropna()

st.line_chart(compare, use_container_width=True)

st.info(
    "Why this matters: leveraged/inverse ETFs target DAILY returns. Over time, compounding in volatile markets can make "
    "performance drift from what people expect long-term. This is the key 'daily reset / decay' concept."
)

# -------------------------
# 8) Download summary
# -------------------------
st.download_button(
    "Download summary as CSV",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name="etf_summary.csv",
    mime="text/csv"
)
