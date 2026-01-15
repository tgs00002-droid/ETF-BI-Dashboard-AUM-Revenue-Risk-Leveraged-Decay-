import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="ETF BI Dashboard (ProShares-style)", layout="wide")

st.title("ETF BI Dashboard (AUM • Revenue • Risk • Leveraged Decay)")
st.caption("Portfolio-style BI project: KPI table + performance/risk + daily reset simulation (educational).")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Controls")

default_tickers = ["SPY", "QQQ", "TLT", "SSO", "SH"]  # includes ProShares examples
tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value=",".join(default_tickers)
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

benchmark_ticker = st.sidebar.selectbox("Benchmark ticker", options=tickers, index=0)

start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))

st.sidebar.subheader("Leveraged / Inverse Simulation")
lev_long = st.sidebar.slider("Leverage (long)", min_value=1.0, max_value=3.0, value=2.0, step=0.5)
lev_inv = st.sidebar.slider("Leverage (inverse)", min_value=-3.0, max_value=-1.0, value=-1.0, step=0.5)

st.sidebar.subheader("AUM + Fees (you can edit)")
st.sidebar.caption("These are placeholders for BI modeling. You can update to real values later.")

# Default placeholder AUM + expense ratios (you can change)
DEFAULT_META = {
    "SPY": {"name": "S&P 500", "expense_ratio": 0.0009, "aum": 500e9},
    "QQQ": {"name": "Nasdaq 100", "expense_ratio": 0.0020, "aum": 250e9},
    "TLT": {"name": "20Y+ Treasuries", "expense_ratio": 0.0015, "aum": 40e9},
    "SSO": {"name": "ProShares Ultra S&P500 (2x)", "expense_ratio": 0.0091, "aum": 2e9},
    "SH":  {"name": "ProShares Short S&P500 (-1x)", "expense_ratio": 0.0089, "aum": 3e9},
}

# Let user override AUM/fee quickly for the selected tickers
meta = {}
for t in tickers:
    base = DEFAULT_META.get(t, {"name": t, "expense_ratio": 0.0050, "aum": 1e9})
    st.sidebar.markdown(f"**{t}**")
    name = st.sidebar.text_input(f"{t} name", value=base["name"], key=f"name_{t}")
    exp = st.sidebar.number_input(f"{t} expense ratio (decimal)", value=float(base["expense_ratio"]), step=0.0001, format="%.4f", key=f"exp_{t}")
    aum = st.sidebar.number_input(f"{t} AUM (USD)", value=float(base["aum"]), step=1e8, format="%.0f", key=f"aum_{t}")
    meta[t] = {"ticker": t, "name": name, "expense_ratio": exp, "aum": aum}

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def fetch_adj_close(tickers_list, start, end) -> pd.DataFrame:
    data = yf.download(tickers_list, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"].copy()
    else:
        prices = data[["Adj Close"]].copy()
        prices.columns = tickers_list
    prices = prices.dropna(how="all")
    return prices

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")

def cumulative_return(rets: pd.DataFrame) -> pd.Series:
    return (1 + rets).prod() - 1

def annualized_volatility(rets: pd.DataFrame, trading_days=252) -> pd.Series:
    return rets.std() * np.sqrt(trading_days)

def max_drawdown(prices: pd.DataFrame) -> pd.Series:
    peak = prices.cummax()
    dd = prices / peak - 1.0
    return dd.min()

def tracking_error(etf_ret: pd.Series, bench_ret: pd.Series, trading_days=252) -> float:
    diff = (etf_ret - bench_ret).dropna()
    if diff.empty:
        return float("nan")
    return float(diff.std() * np.sqrt(trading_days))

def simulate_daily_reset(bench_returns: pd.Series, leverage: float, start_value: float = 100.0) -> pd.Series:
    vals = [start_value]
    idx = bench_returns.dropna().index
    for r in bench_returns.dropna():
        vals.append(vals[-1] * (1 + leverage * r))
    return pd.Series(vals[1:], index=idx)

def build_summary(prices: pd.DataFrame, meta_dict: dict, bench_ticker: str) -> pd.DataFrame:
    rets = daily_returns(prices)
    bench = rets[bench_ticker]

    cum = cumulative_return(rets)
    vol = annualized_volatility(rets)
    mdd = max_drawdown(prices)

    rows = []
    for t, m in meta_dict.items():
        if t not in prices.columns:
            continue

        te = 0.0 if t == bench_ticker else tracking_error(rets[t], bench)
        fee_rev = m["aum"] * m["expense_ratio"]

        rows.append({
            "Ticker": t,
            "Name": m["name"],
            "AUM ($)": m["aum"],
            "Expense Ratio": m["expense_ratio"],
            "Est. Annual Fee Revenue ($)": fee_rev,
            "Cumulative Return": float(cum.get(t, np.nan)),
            "Annualized Volatility": float(vol.get(t, np.nan)),
            "Max Drawdown": float(mdd.get(t, np.nan)),
            f"Tracking Error vs {bench_ticker}": float(te),
        })

    out = pd.DataFrame(rows).sort_values("AUM ($)", ascending=False)
    return out

# ---------------------------
# Load data
# ---------------------------
with st.spinner("Downloading ETF price history..."):
    prices = fetch_adj_close(tickers, start_date, end_date)

if prices.empty:
    st.error("No price data returned. Check tickers or date range.")
    st.stop()

# Align columns to requested tickers order (if possible)
prices = prices[[c for c in tickers if c in prices.columns]]

rets = daily_returns(prices)
growth_100 = (1 + rets).cumprod() * 100

# ---------------------------
# Executive KPIs (top cards)
# ---------------------------
summary_df = build_summary(prices, meta, benchmark_ticker)

col1, col2, col3, col4 = st.columns(4)
total_aum = summary_df["AUM ($)"].sum()
total_fee_rev = summary_df["Est. Annual Fee Revenue ($)"].sum()

col1.metric("Tickers", len(prices.columns))
col2.metric("Total AUM (model)", f"${total_aum:,.0f}")
col3.metric("Est. Annual Fee Revenue (model)", f"${total_fee_rev:,.0f}")
col4.metric("Benchmark", benchmark_ticker)

st.divider()

# ---------------------------
# Main layout
# ---------------------------
left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Executive Summary Table")
    fmt = summary_df.copy()
    # Friendly formatting
    money_cols = ["AUM ($)", "Est. Annual Fee Revenue ($)"]
    for c in money_cols:
        fmt[c] = fmt[c].map(lambda x: f"${x:,.0f}")
    fmt["Expense Ratio"] = fmt["Expense Ratio"].map(lambda x: f"{x:.4f}")
    fmt["Cumulative Return"] = fmt["Cumulative Return"].map(lambda x: f"{x:.2%}")
    fmt["Annualized Volatility"] = fmt["Annualized Volatility"].map(lambda x: f"{x:.2%}")
    fmt["Max Drawdown"] = fmt["Max Drawdown"].map(lambda x: f"{x:.2%}")
    te_col = f"Tracking Error vs {benchmark_ticker}"
    fmt[te_col] = fmt[te_col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "NA")

    st.dataframe(fmt, use_container_width=True)

    st.subheader("Growth of $100 (Adjusted Close)")
    st.line_chart(growth_100, use_container_width=True)

with right:
    st.subheader("Daily Reset Simulation (Volatility Decay Illustration)")
    bench_series = rets[benchmark_ticker].dropna()

    sim_long = simulate_daily_reset(bench_series, leverage=lev_long, start_value=100)
    sim_inv = simulate_daily_reset(bench_series, leverage=lev_inv, start_value=100)

    # Align benchmark growth to sim index
    bench_growth = growth_100[benchmark_ticker].reindex(sim_long.index).dropna()

    sim_df = pd.DataFrame({
        f"Simulated {lev_long}x Daily Reset": sim_long.reindex(bench_growth.index),
        f"Simulated {lev_inv}x Daily Reset": sim_inv.reindex(bench_growth.index),
        f"{benchmark_ticker} Benchmark": bench_growth
    }).dropna()

    st.line_chart(sim_df, use_container_width=True)

    st.markdown("**What to say in the interview:**")
    st.write(
        "“This panel shows daily reset compounding. In volatile markets, leveraged/inverse products can drift away from "
        "simple long-term expectations, which is why they’re typically positioned for short-term tactical use.”"
    )

st.divider()

# ---------------------------
# Quick download
# ---------------------------
st.subheader("Download Summary")
csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
st.download_button("Download summary.csv", data=csv_bytes, file_name="summary.csv", mime="text/csv")
