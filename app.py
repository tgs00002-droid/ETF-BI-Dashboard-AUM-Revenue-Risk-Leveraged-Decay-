import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from io import StringIO

# -----------------------------
# CONFIG
# -----------------------------
URL_SINGLE_STOCK = (
    "https://www.proshares.com/our-etfs/find-leveraged-and-inverse-etfs"
    "?etftype=&strategy=Single+Stock&benchmark=&product=Product+Overview+&search="
)

URL_PRICE_NAV = (
    "https://www.proshares.com/our-etfs/find-leveraged-and-inverse-etfs"
    "?etftype=&strategy=&benchmark=&product=Price%2FNAV&search="
)

TOP_N = 12  # keep charts readable and fast


# -----------------------------
# STREAMLIT PAGE
# -----------------------------
st.set_page_config(page_title="ProShares Single Stock BI Dashboard", layout="wide")
st.title("ProShares Single Stock BI Dashboard")
st.caption(
    "ProShares source tables (Single Stock + Price/NAV) + real price history to answer key BI questions."
)

# Only control: date range for performance section
c1, c2 = st.columns(2)
with c1:
    start = st.date_input("Performance start date", value=pd.to_datetime("2024-01-01"))
with c2:
    end = st.date_input("Performance end date", value=pd.to_datetime("today"))


# -----------------------------
# HELPERS
# -----------------------------
def _get_html(url: str) -> str:
    # Streamlit Cloud often blocks bare requests. This header fixes it.
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


def _to_number(x):
    # handles "$266.56", "$-0.23 (-0.09%)", "0.08%", etc.
    s = str(x)
    s = s.replace("$", "").replace(",", "").strip()
    # keep only first token that looks like a number
    token = s.split()[0] if len(s.split()) else s
    token = token.replace("%", "")
    return pd.to_numeric(token, errors="coerce")


@st.cache_data(show_spinner=False)
def load_single_stock_funds() -> pd.DataFrame:
    html = _get_html(URL_SINGLE_STOCK)
    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame()

    # Find Product Overview table (has these columns)
    target = None
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if "Ticker" in cols and "Fund Name" in cols and "Fund Type" in cols:
            target = t.copy()
            break
    if target is None:
        return pd.DataFrame()

    target["Fund Type"] = target["Fund Type"].astype(str).str.strip()
    df = target[target["Fund Type"].str.lower() == "single stock"].copy()

    # Net Assets numeric for ranking
    if "Net Assets" in df.columns:
        df["Net Assets Numeric"] = df["Net Assets"].map(_to_number)
    else:
        df["Net Assets Numeric"] = np.nan

    keep = [c for c in ["Ticker", "Fund Name", "Daily Objective", "Net Assets", "Net Assets Numeric", "Index/Benchmark"] if c in df.columns]
    df = df[keep].dropna(subset=["Ticker"]).reset_index(drop=True)

    if df["Net Assets Numeric"].notna().any():
        df = df.sort_values("Net Assets Numeric", ascending=False).reset_index(drop=True)

    return df


@st.cache_data(show_spinner=False)
def load_price_nav_snapshot() -> pd.DataFrame:
    html = _get_html(URL_PRICE_NAV)
    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame()

    # Find the table that contains Premium / Discount
    target = None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "ticker" in cols and any("premium" in c for c in cols) and any("nav" in c for c in cols):
            target = t.copy()
            break

    if target is None:
        return pd.DataFrame()

    # Normalize column names
    target.columns = [str(c).strip() for c in target.columns]
    return target


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


def annual_vol(ret: pd.Series) -> float:
    return float(ret.std() * np.sqrt(252))


def max_drawdown(px: pd.Series) -> float:
    peak = px.cummax()
    dd = px / peak - 1.0
    return float(dd.min())


# -----------------------------
# LOAD DATA
# -----------------------------
with st.spinner("Loading ProShares Single Stock list..."):
    single_stock = load_single_stock_funds()

if single_stock.empty:
    st.error("Could not load Single Stock table from ProShares. (Site layout or access blocked.)")
    st.stop()

tickers = single_stock["Ticker"].head(TOP_N).astype(str).str.strip().tolist()

with st.spinner("Loading ProShares Price/NAV snapshot..."):
    price_nav = load_price_nav_snapshot()

# -----------------------------
# SECTION 1: WHAT FUNDS EXIST
# -----------------------------
st.subheader("1) ProShares Single Stock ETFs (Top by Net Assets)")
show_cols = [c for c in ["Ticker", "Fund Name", "Daily Objective", "Net Assets", "Index/Benchmark"] if c in single_stock.columns]
st.dataframe(single_stock[show_cols].head(TOP_N), use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# SECTION 2: PRICE vs NAV (SHOWCASE)
# -----------------------------
st.subheader("2) Price vs NAV (Premium/Discount + Bid/Ask Spread)")

if price_nav.empty or "Ticker" not in price_nav.columns:
    st.warning("Could not parse the ProShares Price/NAV table right now. Performance section below still works.")
else:
    pn = price_nav.copy()
    pn["Ticker"] = pn["Ticker"].astype(str).str.strip()
    pn = pn[pn["Ticker"].isin(tickers)].copy()

    # Detect likely columns (these exist on the page)
    # Example header includes: Market Price (Change), NAV (Change), 30 day bid / ask spread, Premium / Discount
    colmap = {c.lower(): c for c in pn.columns}
    market_col = None
    nav_col = None
    spread_col = None
    prem_col = None

    for c in pn.columns:
        lc = c.lower()
        if market_col is None and "market price" in lc:
            market_col = c
        if nav_col is None and lc == "nav (change)" or ("nav" in lc and nav_col is None):
            nav_col = c
        if spread_col is None and "bid / ask" in lc:
            spread_col = c
        if prem_col is None and "premium" in lc:
            prem_col = c

    if market_col is None or nav_col is None or prem_col is None:
        st.warning("Price/NAV columns changed on the website. We loaded the table but couldn't detect the key columns.")
    else:
        pn_out = pd.DataFrame({
            "Ticker": pn["Ticker"],
            "Fund Name": pn["Fund Name"] if "Fund Name" in pn.columns else "",
            "Market Price": pn[market_col].map(_to_number),
            "NAV": pn[nav_col].map(_to_number),
            "30D Bid/Ask Spread (%)": pn[spread_col].map(_to_number) if spread_col else np.nan,
            "Premium/Discount (%)": pn[prem_col].map(_to_number),
        }).dropna(subset=["Ticker"])

        # Convert percent columns (if already in % units, keep as percent)
        # Premium/Discount comes like "-0.09%" on site -> _to_number gives -0.09 (percent units)
        # We'll keep it in percent units for display.
        # Same for spread.
        # Add “absolute spread” for ranking efficiency
        pn_out["Abs Premium/Discount (%)"] = pn_out["Premium/Discount (%)"].abs()

        # Key answers (what you can say you built)
        k1, k2, k3 = st.columns(3)
        biggest_premium = pn_out.sort_values("Premium/Discount (%)", ascending=False).head(1).iloc[0]
        biggest_discount = pn_out.sort_values("Premium/Discount (%)", ascending=True).head(1).iloc[0]

        # Tightest spread: smallest 30D spread
        if pn_out["30D Bid/Ask Spread (%)"].notna().any():
            tightest = pn_out.sort_values("30D Bid/Ask Spread (%)", ascending=True).head(1).iloc[0]
            k3.metric("Tightest Bid/Ask Spread", tightest["Ticker"], f"{tightest['30D Bid/Ask Spread (%)']:.2f}%")
        else:
            k3.metric("Tightest Bid/Ask Spread", "N/A", "No data")

        k1.metric("Largest Premium vs NAV", biggest_premium["Ticker"], f"{biggest_premium['Premium/Discount (%)']:.2f}%")
        k2.metric("Largest Discount vs NAV", biggest_discount["Ticker"], f"{biggest_discount['Premium/Discount (%)']:.2f}%")

        # Clean display table
        display = pn_out[["Ticker", "Market Price", "NAV", "30D Bid/Ask Spread (%)", "Premium/Discount (%)"]].copy()
        display["Market Price"] = display["Market Price"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "")
        display["NAV"] = display["NAV"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "")
        display["30D Bid/Ask Spread (%)"] = display["30D Bid/Ask Spread (%)"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")
        display["Premium/Discount (%)"] = display["Premium/Discount (%)"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")

        st.dataframe(display, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# SECTION 3: PERFORMANCE / RISK (REAL HISTORY)
# -----------------------------
st.subheader("3) Performance & Risk (Real Price History)")

with st.spinner("Downloading historical prices for performance metrics..."):
    prices = fetch_prices(tickers, start, end)

if prices.empty or prices.shape[0] < 10:
    st.error("Not enough price data for this date range. Try a wider range (example: start 2023-01-01).")
    st.stop()

prices = prices.dropna(axis=1, how="all")
rets = prices.pct_change().dropna()
growth_100 = (1 + rets).cumprod() * 100

rows = []
for t in prices.columns:
    total_ret = float(growth_100[t].iloc[-1] / growth_100[t].iloc[0] - 1)
    vol = annual_vol(rets[t])
    mdd = max_drawdown(prices[t])
    rows.append({"Ticker": t, "Total Return": total_ret, "Volatility (ann.)": vol, "Max Drawdown": mdd})

metrics = pd.DataFrame(rows)

best = metrics.sort_values("Total Return", ascending=False).iloc[0]
riskiest = metrics.sort_values("Volatility (ann.)", ascending=False).iloc[0]
worstdd = metrics.sort_values("Max Drawdown").iloc[0]  # most negative

m1, m2, m3 = st.columns(3)
m1.metric("Best Performer", best["Ticker"], f"{best['Total Return']:.1%}")
m2.metric("Riskiest (Volatility)", riskiest["Ticker"], f"{riskiest['Volatility (ann.)']:.1%}")
m3.metric("Worst Drawdown", worstdd["Ticker"], f"{worstdd['Max Drawdown']:.1%}")

# Chart answers: “How have these moved over time?”
st.subheader("Growth of $100")
st.line_chart(growth_100, use_container_width=True)

# Download button for recruiter/interview
st.download_button(
    "Download Performance Metrics CSV",
    data=metrics.to_csv(index=False).encode("utf-8"),
    file_name="proshares_single_stock_performance_metrics.csv",
    mime="text/csv",
)
