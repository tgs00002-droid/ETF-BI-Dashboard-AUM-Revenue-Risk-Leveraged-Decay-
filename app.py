import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from io import StringIO

# =========================================================
# CONFIG
# =========================================================
URL_SINGLE_STOCK = (
    "https://www.proshares.com/our-etfs/find-leveraged-and-inverse-etfs"
    "?etftype=&strategy=Single+Stock&benchmark=&product=Product+Overview+&search="
)

URL_PRICE_NAV = (
    "https://www.proshares.com/our-etfs/find-leveraged-and-inverse-etfs"
    "?etftype=&strategy=&benchmark=&product=Price%2FNAV&search="
)

# Your requested performance view:
# product=Performance & month=Market Price, Month-End
URL_PERFORMANCE = (
    "https://www.proshares.com/our-etfs/find-leveraged-and-inverse-etfs"
    "?etftype=&strategy=&benchmark=&product=Performance&month=Market%20Price%2C%20Month-End&search="
)

TOP_N = 12  # keep charts readable; auto-select top by net assets (if available)

# =========================================================
# STREAMLIT PAGE
# =========================================================
st.set_page_config(page_title="ProShares Single Stock BI Dashboard", layout="wide")
st.title("ProShares Single Stock ETF BI Dashboard")
st.caption("Pulls ProShares tables (Product Overview, Price/NAV, Performance) + real price history.")

# Only controls: date range (for Yahoo Finance charts)
c1, c2 = st.columns(2)
with c1:
    start = st.date_input("Chart start date", value=pd.to_datetime("2024-01-01"))
with c2:
    end = st.date_input("Chart end date", value=pd.to_datetime("today"))

# =========================================================
# HELPERS
# =========================================================
def _get_html(url: str) -> str:
    """Fetch HTML in a Streamlit-Cloud-friendly way (User-Agent avoids 403)."""
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


def _to_num(x):
    """
    Convert values like "$184,642,499", "0.12%", "$4.53", "$-0.10 (-1.2%)" -> float.
    Keeps first numeric token.
    """
    s = str(x).strip()
    s = s.replace("$", "").replace(",", "")
    if not s:
        return np.nan
    token = s.split()[0]  # take first token
    token = token.replace("%", "")
    return pd.to_numeric(token, errors="coerce")


def _find_table_with_columns(tables, required_cols):
    """Find the first table containing all required columns."""
    req = set([c.strip().lower() for c in required_cols])
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if req.issubset(set(cols)):
            out = t.copy()
            out.columns = [str(c).strip() for c in out.columns]
            return out
    return None


def _pick_columns(df: pd.DataFrame, include_keywords):
    """Return columns whose name contains ANY keyword in include_keywords (case-insensitive)."""
    out = []
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in include_keywords):
            out.append(c)
    return out


@st.cache_data(show_spinner=False)
def load_single_stock_table(url: str) -> pd.DataFrame:
    html = _get_html(url)
    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame()

    # Product Overview should have: Ticker, Fund Name, Fund Type
    target = _find_table_with_columns(tables, ["Ticker", "Fund Name", "Fund Type"])
    if target is None:
        return pd.DataFrame()

    target["Fund Type"] = target["Fund Type"].astype(str).str.strip()
    df = target[target["Fund Type"].str.lower() == "single stock"].copy()

    # net assets numeric for sorting if present
    if "Net Assets" in df.columns:
        df["Net Assets Numeric"] = df["Net Assets"].map(_to_num)
    else:
        df["Net Assets Numeric"] = np.nan

    keep = [c for c in ["Ticker", "Fund Name", "Daily Objective", "Net Assets", "Net Assets Numeric", "Index/Benchmark"] if c in df.columns]
    df = df[keep].dropna(subset=["Ticker"]).reset_index(drop=True)

    if df["Net Assets Numeric"].notna().any():
        df = df.sort_values("Net Assets Numeric", ascending=False).reset_index(drop=True)

    return df


@st.cache_data(show_spinner=False)
def load_price_nav_table(url: str) -> pd.DataFrame:
    html = _get_html(url)
    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame()

    # Price/NAV usually includes: Ticker + NAV + Premium/Discount or Market Price
    # We'll look for Ticker plus any NAV-ish column.
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "ticker" in cols and any("nav" in c for c in cols) and (any("premium" in c for c in cols) or any("market" in c for c in cols) or any("price" in c for c in cols)):
            out = t.copy()
            out.columns = [str(c).strip() for c in out.columns]
            return out

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_performance_table(url: str) -> pd.DataFrame:
    html = _get_html(url)
    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame()

    # Performance table should have at least Ticker and some return columns
    # We'll pick the first table that has "Ticker" and contains any % / "Month" / "YTD" etc.
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "ticker" in cols:
            # Heuristic: if it has any performance-like columns
            if any(("month" in c) or ("ytd" in c) or ("1 year" in c) or ("3 year" in c) or ("return" in c) for c in cols):
                out = t.copy()
                out.columns = [str(c).strip() for c in out.columns]
                return out

    # fallback: just return first ticker table
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "ticker" in cols:
            out = t.copy()
            out.columns = [str(c).strip() for c in out.columns]
            return out

    return pd.DataFrame()


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


# =========================================================
# LOAD PROSHARES DATA
# =========================================================
with st.spinner("Loading ProShares Single Stock list (Product Overview)..."):
    single_stock = load_single_stock_table(URL_SINGLE_STOCK)

if single_stock.empty:
    st.error("Could not load ProShares Single Stock table. (Site layout/access may have changed.)")
    st.stop()

tickers = single_stock["Ticker"].head(TOP_N).astype(str).str.strip().tolist()

with st.spinner("Loading ProShares Price/NAV table..."):
    price_nav = load_price_nav_table(URL_PRICE_NAV)

with st.spinner("Loading ProShares Performance table (Market Price, Month-End)..."):
    perf = load_performance_table(URL_PERFORMANCE)

# =========================================================
# DISPLAY: FUND LIST
# =========================================================
st.subheader("1) ProShares Single Stock ETFs (Top by Net Assets)")
cols_show = [c for c in ["Ticker", "Fund Name", "Daily Objective", "Net Assets", "Index/Benchmark"] if c in single_stock.columns]
st.dataframe(single_stock[cols_show].head(TOP_N), use_container_width=True, hide_index=True)

st.divider()

# =========================================================
# DISPLAY: PRICE/NAV
# =========================================================
st.subheader("2) Price vs NAV (Premium/Discount + Spread)")

if price_nav.empty or "Ticker" not in price_nav.columns:
    st.warning("Price/NAV table could not be parsed right now.")
else:
    pn = price_nav.copy()
    pn["Ticker"] = pn["Ticker"].astype(str).str.strip()
    pn = pn[pn["Ticker"].isin(tickers)].copy()

    # Find likely columns
    # These names can vary, so we detect by keywords
    market_cols = _pick_columns(pn, ["market price", "market"])
    nav_cols = _pick_columns(pn, ["nav"])
    prem_cols = _pick_columns(pn, ["premium", "discount"])
    spread_cols = _pick_columns(pn, ["bid / ask", "bid/ask", "spread"])

    market_col = market_cols[0] if market_cols else None
    nav_col = nav_cols[0] if nav_cols else None
    prem_col = prem_cols[0] if prem_cols else None
    spread_col = spread_cols[0] if spread_cols else None

    if market_col and nav_col:
        out = pd.DataFrame({"Ticker": pn["Ticker"]})
        out["Market Price"] = pn[market_col].map(_to_num)
        out["NAV"] = pn[nav_col].map(_to_num)

        if prem_col:
            # ProShares often provides premium/discount already in %
            out["Premium/Discount (%)"] = pn[prem_col].map(_to_num)
        else:
            # Compute if not provided
            out["Premium/Discount (%)"] = ((out["Market Price"] - out["NAV"]) / out["NAV"]) * 100

        if spread_col:
            out["30D Bid/Ask Spread (%)"] = pn[spread_col].map(_to_num)

        # Key callouts
        c1, c2, c3 = st.columns(3)
        valid = out.dropna(subset=["Premium/Discount (%)"])
        if not valid.empty:
            prem = valid.sort_values("Premium/Discount (%)", ascending=False).iloc[0]
            disc = valid.sort_values("Premium/Discount (%)", ascending=True).iloc[0]
            c1.metric("Largest Premium", prem["Ticker"], f"{prem['Premium/Discount (%)']:.2f}%")
            c2.metric("Largest Discount", disc["Ticker"], f"{disc['Premium/Discount (%)']:.2f}%")
        else:
            c1.metric("Largest Premium", "N/A", "")
            c2.metric("Largest Discount", "N/A", "")

        if "30D Bid/Ask Spread (%)" in out.columns and out["30D Bid/Ask Spread (%)"].notna().any():
            tight = out.sort_values("30D Bid/Ask Spread (%)", ascending=True).iloc[0]
            c3.metric("Tightest Spread", tight["Ticker"], f"{tight['30D Bid/Ask Spread (%)']:.2f}%")
        else:
            c3.metric("Tightest Spread", "N/A", "")

        # Display table nicely
        disp = out.copy()
        disp["Market Price"] = disp["Market Price"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "")
        disp["NAV"] = disp["NAV"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "")
        disp["Premium/Discount (%)"] = disp["Premium/Discount (%)"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")
        if "30D Bid/Ask Spread (%)" in disp.columns:
            disp["30D Bid/Ask Spread (%)"] = disp["30D Bid/Ask Spread (%)"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")

        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Price/NAV table loaded but could not detect Market Price and NAV columns.")

st.divider()

# =========================================================
# DISPLAY: PERFORMANCE TABLE (ProShares)
# =========================================================
st.subheader("3) ProShares Performance (Market Price, Month-End)")

if perf.empty or "Ticker" not in perf.columns:
    st.warning("Performance table could not be parsed right now.")
else:
    pf = perf.copy()
    pf["Ticker"] = pf["Ticker"].astype(str).str.strip()
    pf = pf[pf["Ticker"].isin(tickers)].copy()

    # Show important columns first if they exist
    # We’ll keep it flexible because ProShares may change the exact headers.
    preferred = []
    for key in ["Ticker", "Fund Name", "1 Month", "3 Month", "6 Month", "YTD", "1 Year", "Since Inception"]:
        for c in pf.columns:
            if c not in preferred and key.lower() in str(c).lower():
                preferred.append(c)

    # Fallback to first 10 columns if we didn’t find much
    if len(preferred) < 2:
        preferred = pf.columns.tolist()[:10]

    st.dataframe(pf[preferred], use_container_width=True, hide_index=True)

st.divider()

# =========================================================
# MARKET HISTORY: PRICES + GROWTH
# =========================================================
st.subheader("4) Market History (Yahoo Finance): Performance & Risk")

with st.spinner("Downloading price history for charts..."):
    prices = fetch_prices(tickers, start, end)

if prices.empty or prices.shape[0] < 10:
    st.error("Not enough Yahoo Finance price history for the selected date range. Try a wider range.")
    st.stop()

prices = prices.dropna(axis=1, how="all")
rets = prices.pct_change().dropna()
growth_100 = (1 + rets).cumprod() * 100

# Metrics (based on market history)
rows = []
for t in prices.columns:
    total_ret = float(growth_100[t].iloc[-1] / growth_100[t].iloc[0] - 1)
    vol = annual_volatility(rets[t])
    mdd = max_drawdown(prices[t])
    rows.append({"Ticker": t, "Total Return": total_ret, "Volatility (ann.)": vol, "Max Drawdown": mdd})
metrics = pd.DataFrame(rows)

best = metrics.sort_values("Total Return", ascending=False).iloc[0]
riskiest = metrics.sort_values("Volatility (ann.)", ascending=False).iloc[0]
worstdd = metrics.sort_values("Max Drawdown").iloc[0]

m1, m2, m3 = st.columns(3)
m1.metric("Best Performer", best["Ticker"], f"{best['Total Return']:.1%}")
m2.metric("Riskiest (Volatility)", riskiest["Ticker"], f"{riskiest['Volatility (ann.)']:.1%}")
m3.metric("Worst Drawdown", worstdd["Ticker"], f"{worstdd['Max Drawdown']:.1%}")

st.subheader("Growth of $100 (Normalized Performance)")
st.caption("All lines start at 100. If CRCA shows 39, that means $100 invested became $39 (not the share price).")
st.line_chart(growth_100, use_container_width=True)

st.subheader("Actual ETF Prices (Real Market Prices)")
st.caption("This shows real adjusted close prices (e.g., CRCA could be around $4–$5).")
st.line_chart(prices, use_container_width=True)

# Download metrics for your project
st.download_button(
    "Download market history metrics (CSV)",
    data=metrics.to_csv(index=False).encode("utf-8"),
    file_name="proshares_single_stock_market_history_metrics.csv",
    mime="text/csv",
)
