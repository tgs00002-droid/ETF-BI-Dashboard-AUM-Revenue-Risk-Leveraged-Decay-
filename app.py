import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from io import StringIO

st.set_page_config(page_title="ProShares Single Stock Dashboard", layout="wide")
st.title("ProShares Single Stock ETF Dashboard (Simple + Price/NAV)")
st.caption(
    "Uses ProShares screener tables (Single Stock + Price/NAV) and real price history "
    "to answer: premium/discount vs NAV + performance/risk."
)

# Minimal controls: only date range
c1, c2 = st.columns(2)
with c1:
    start = st.date_input("Start date", value=pd.to_datetime("2024-01-01"))
with c2:
    end = st.date_input("End date", value=pd.to_datetime("today"))

# Your ProShares links
URL_SINGLE_STOCK = "https://www.proshares.com/our-etfs/find-leveraged-and-inverse-etfs?etftype=&strategy=Single+Stock&benchmark=&product=Product+Overview+&search="
URL_PRICE_NAV    = "https://www.proshares.com/our-etfs/find-leveraged-and-inverse-etfs?etftype=&strategy=&benchmark=&product=Price%2FNAV&search="

TOP_N = 10  # keep charts readable


def _get_html(url: str) -> str:
    """Fetch HTML with headers so Streamlit Cloud doesn’t get blocked."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text


@st.cache_data(show_spinner=False)
def load_single_stock_table(url: str) -> pd.DataFrame:
    html = _get_html(url)
    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame()

    # Find a table containing these columns
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

    # Clean net assets if present
    if "Net Assets" in df.columns:
        df["Net Assets Numeric"] = (
            df["Net Assets"].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["Net Assets Numeric"] = pd.to_numeric(df["Net Assets Numeric"], errors="coerce")
    else:
        df["Net Assets Numeric"] = np.nan

    keep_cols = [c for c in ["Ticker", "Fund Name", "Daily Objective", "Net Assets", "Net Assets Numeric", "Index/Benchmark"] if c in df.columns]
    df = df[keep_cols].dropna(subset=["Ticker"]).reset_index(drop=True)

    # sort by net assets if available
    if df["Net Assets Numeric"].notna().any():
        df = df.sort_values("Net Assets Numeric", ascending=False).reset_index(drop=True)

    return df


@st.cache_data(show_spinner=False)
def load_price_nav_table(url: str) -> pd.DataFrame:
    """
    Parse the ProShares Price/NAV table.
    The exact column names can vary (Market Price vs NAV month-end/quarter-end),
    so we search for a table that includes 'Ticker' plus some NAV/Price columns.
    """
    html = _get_html(url)
    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame()

    candidate = None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "ticker" in cols and (any("nav" in c for c in cols) or any("market" in c for c in cols) or any("price" in c for c in cols)):
            candidate = t.copy()
            break

    if candidate is None:
        return pd.DataFrame()

    candidate.columns = [str(c).strip() for c in candidate.columns]
    return candidate


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
# Load ProShares tables
# ----------------------------
with st.spinner("Loading ProShares Single Stock list..."):
    single_stock = load_single_stock_table(URL_SINGLE_STOCK)

if single_stock.empty:
    st.error("Could not load the ProShares Single Stock table.")
    st.stop()

with st.spinner("Loading ProShares Price/NAV table..."):
    price_nav = load_price_nav_table(URL_PRICE_NAV)

if price_nav.empty:
    st.warning("Loaded Single Stock table, but could not parse Price/NAV table (site layout may have changed).")

# pick top tickers from ProShares single stock list
tickers = single_stock["Ticker"].head(TOP_N).astype(str).str.strip().tolist()

st.subheader("1) ProShares Single Stock ETFs (source table)")
show_cols = [c for c in ["Ticker", "Fund Name", "Daily Objective", "Net Assets", "Index/Benchmark"] if c in single_stock.columns]
st.dataframe(single_stock[show_cols], use_container_width=True, hide_index=True)

st.divider()

# ----------------------------
# Price/NAV “Premium/Discount” answers
# ----------------------------
st.subheader("2) Price vs NAV (Premium/Discount) — what this answers")

if not price_nav.empty and "Ticker" in price_nav.columns:
    pn = price_nav.copy()

    # Keep only tickers we’re charting (top N single stock)
    pn["Ticker"] = pn["Ticker"].astype(str).str.strip()
    pn = pn[pn["Ticker"].isin(tickers)].copy()

    # Find a likely Market Price column + NAV column (names vary)
    cols_lower = {c: c.lower() for c in pn.columns}

    price_col = None
    nav_col = None

    # Prefer quarter-end if present
    for c in pn.columns:
        lc = cols_lower[c]
        if price_col is None and ("market price" in lc or (("price" in lc) and ("market" in lc))):
            price_col = c
        if nav_col is None and "nav" in lc:
            nav_col = c

    # If still not found, just try common patterns
    if price_col is None:
        for c in pn.columns:
            if "Price" in c or "Market" in c:
                price_col = c
                break
    if nav_col is None:
        for c in pn.columns:
            if "NAV" in c:
                nav_col = c
                break

    if price_col and nav_col:
        def to_num(x):
            return pd.to_numeric(
                str(x).replace("$", "").replace(",", "").strip(),
                errors="coerce"
            )

        pn["Market Price"] = pn[price_col].map(to_num)
        pn["NAV"] = pn[nav_col].map(to_num)
        pn["Premium/Discount"] = (pn["Market Price"] - pn["NAV"]) / pn["NAV"]

        # Key answers
        pn_valid = pn.dropna(subset=["Premium/Discount"]).copy()
        if not pn_valid.empty:
            biggest_premium = pn_valid.sort_values("Premium/Discount", ascending=False).iloc[0]
            biggest_discount = pn_valid.sort_values("Premium/Discount", ascending=True).iloc[0]

            a, b = st.columns(2)
            a.metric("Largest Premium vs NAV", biggest_premium["Ticker"], f"{biggest_premium['Premium/Discount']:.2%}")
            b.metric("Largest Discount vs NAV", biggest_discount["Ticker"], f"{biggest_discount['Premium/Discount']:.2%}")

            # Clean table
            out = pn_valid[["Ticker", "Market Price", "NAV", "Premium/Discount"]].copy()
            out["Market Price"] = out["Market Price"].map(lambda v: f"${v:,.2f}")
            out["NAV"] = out["NAV"].map(lambda v: f"${v:,.2f}")
            out["Premium/Discount"] = out["Premium/Discount"].map(lambda v: f"{v:.2%}")

            st.dataframe(out, use_container_width=True, hide_index=True)
        else:
            st.info("Price/NAV parsed, but premium/discount could not be computed for these tickers.")
    else:
        st.info("Price/NAV table loaded, but the exact Price and NAV columns weren’t detected.")
else:
    st.info("Price/NAV table not available right now, but the performance section below still works.")

st.divider()

# ----------------------------
# Performance/Risk section (real market history)
# ----------------------------
st.subheader("3) Performance & Risk (real price history)")

with st.spinner("Downloading price history for performance/risk..."):
    prices = fetch_prices(tickers, start, end)

if prices.empty or prices.shape[0] < 10:
    st.error("Not enough Yahoo Finance data for these tickers in this date range. Try a wider range.")
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

best = metrics.sort_values("Total Return", ascending=False).iloc[0]
riskiest = metrics.sort_values("Volatility (ann.)", ascending=False).iloc[0]
worstdd = metrics.sort_values("Max Drawdown").iloc[0]  # most negative

x1, x2, x3 = st.columns(3)
x1.metric("Best performer", best["Ticker"], f"{best['Total Return']:.1%}")
x2.metric("Riskiest (volatility)", riskiest["Ticker"], f"{riskiest['Volatility (ann.)']:.1%}")
x3.metric("Worst drawdown", worstdd["Ticker"], f"{worstdd['Max Drawdown']:.1%}")

st.subheader("Growth of $100 (real prices)")
st.line_chart(growth_100, use_container_width=True)

st.download_button(
    "Download performance metrics CSV",
    data=metrics.to_csv(index=False).encode("utf-8"),
    file_name="proshares_single_stock_performance_metrics.csv",
    mime="text/csv",
)
