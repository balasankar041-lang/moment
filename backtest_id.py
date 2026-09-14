
import pandas as pd
import numpy as np
import yfinance as yf

# ============================================================
# PLAN 2 + ATR/VOLATILITY FILTER EXPERIMENT
#
# Core:
# 12-2 momentum -> Top 100 -> lowest ID Top 50 -> Top 15
#
# New filter:
# ATR(14) / Price <= 5%
#
# This rejects extremely volatile stocks while keeping the
# original momentum + Information Discretion ranking intact.
#
# Research only. No trading.
# ============================================================

MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"
START = "2019-01-01"
END = "2026-09-30"

TOP_MOMENTUM = 100
TOP_ID = 50
PORTFOLIO_SIZE = 15

COST = 0.005
ATR_PERIOD = 14
MAX_ATR_PCT = 0.05       # 5% daily ATR relative to price

# ------------------------------------------------------------
# Membership
# ------------------------------------------------------------

membership = pd.read_csv(MEMBERSHIP_FILE)
membership["effective_date"] = pd.to_datetime(membership["effective_date"])
membership["symbol"] = membership["symbol"].astype(str).str.strip().str.upper()
membership = membership.drop_duplicates(["effective_date", "symbol"])

membership_by_date = {
    d: set(g["symbol"])
    for d, g in membership.groupby("effective_date")
}
membership_dates = sorted(membership_by_date)

def get_members_for_date(date):
    valid = [d for d in membership_dates if d <= date]
    return membership_by_date[valid[-1]] if valid else set()

symbols = sorted(membership["symbol"].unique())
tickers = [s + ".NS" for s in symbols]

# ------------------------------------------------------------
# Download OHLC data
# ------------------------------------------------------------

print("Downloading OHLC data...")

frames = []

for i in range(0, len(tickers), 25):
    batch = tickers[i:i+25]

    try:
        data = yf.download(
            batch,
            start=START,
            end=END,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column"
        )

        if data.empty:
            continue

        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_product(
                [data.columns, batch]
            )

        frames.append(data)

        print(f"Downloaded {min(i+25, len(tickers))}/{len(tickers)}")

    except Exception as e:
        print(f"Batch failed: {e}")

if not frames:
    raise RuntimeError("No price data downloaded.")

data = pd.concat(frames, axis=1)

# Remove duplicate ticker/field columns.
data = data.loc[:, ~data.columns.duplicated()]

close = data["Close"].copy()
high = data["High"].copy()
low = data["Low"].copy()

close = close.dropna(axis=1, how="all")
high = high.reindex(columns=close.columns)
low = low.reindex(columns=close.columns)

print(f"Stocks with usable data: {close.shape[1]}")
print(f"Trading days: {len(close)}")

# ------------------------------------------------------------
# ATR(14)
# ------------------------------------------------------------

atr_pct = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)

for ticker in close.columns:
    h = high[ticker]
    l = low[ticker]
    c = close[ticker]

    previous_close = c.shift(1)

    tr = pd.concat(
        [
            h - l,
            (h - previous_close).abs(),
            (l - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()

    atr_pct[ticker] = atr / c

# ------------------------------------------------------------
# Month ends
# ------------------------------------------------------------

month_ends = close.resample("ME").last().index

last_completed_month = (
    pd.Timestamp.today().to_period("M").start_time
    - pd.Timedelta(days=1)
)

month_ends = month_ends[month_ends <= last_completed_month]
month_ends = month_ends[month_ends >= pd.Timestamp("2020-02-29")]

portfolio_returns = []
portfolio_dates = []
previous_portfolio = set()

rebalance_count = 0
skipped_count = 0

# ------------------------------------------------------------
# Backtest
# ------------------------------------------------------------

for month_end in month_ends:

    members = get_members_for_date(month_end)

    if not members:
        skipped_count += 1
        continue

    momentum_values = {}
    id_values = {}
    atr_values = {}

    end_2m = month_end - pd.DateOffset(months=2)
    end_12m = month_end - pd.DateOffset(months=12)

    for symbol in members:

        ticker = symbol + ".NS"

        if ticker not in close.columns:
            continue

        s = close[ticker].dropna()

        if len(s) < 270:
            continue

        p12_dates = s.index[s.index <= end_12m]
        p2_dates = s.index[s.index <= end_2m]

        if len(p12_dates) == 0 or len(p2_dates) == 0:
            continue

        d12 = p12_dates[-1]
        d2 = p2_dates[-1]

        p12 = s.loc[d12]
        p2 = s.loc[d2]

        if p12 <= 0:
            continue

        momentum = p2 / p12 - 1

        path = s.loc[d12:d2].pct_change().dropna()

        positive = (path > 0).sum()
        negative = (path < 0).sum()

        directional = positive + negative

        if directional == 0:
            continue

        positive_pct = positive / directional
        negative_pct = negative / directional

        id_value = np.sign(momentum) * (
            negative_pct - positive_pct
        )

        # ATR at rebalance date.
        a = atr_pct[ticker].dropna()
        a_dates = a.index[a.index <= month_end]

        if len(a_dates) == 0:
            continue

        atr_value = a.loc[a_dates[-1]]

        if pd.isna(atr_value):
            continue

        momentum_values[symbol] = momentum
        id_values[symbol] = id_value
        atr_values[symbol] = atr_value

    if len(momentum_values) < TOP_MOMENTUM:
        skipped_count += 1
        continue

    # Top 100 momentum.
    momentum_series = pd.Series(momentum_values).sort_values(
        ascending=False
    )

    top100 = momentum_series.head(TOP_MOMENTUM)
    top100 = top100[top100 > 0]

    if len(top100) < PORTFOLIO_SIZE:
        skipped_count += 1
        continue

    # Lowest ID -> Top 50.
    id_series = pd.Series({
        s: id_values[s]
        for s in top100.index
        if s in id_values
    }).sort_values()

    top50 = id_series.head(TOP_ID)

    if len(top50) < PORTFOLIO_SIZE:
        skipped_count += 1
        continue

    # --------------------------------------------------------
    # ATR filter
    # Keep only stocks with ATR/Price <= 5%.
    # --------------------------------------------------------

    atr_series = pd.Series({
        s: atr_values[s]
        for s in top50.index
        if s in atr_values
    })

    atr_pass = atr_series[atr_series <= MAX_ATR_PCT].index

    selected_candidates = top50.index.intersection(atr_pass)

    if len(selected_candidates) < PORTFOLIO_SIZE:
        skipped_count += 1
        continue

    selected = top50.loc[selected_candidates].head(PORTFOLIO_SIZE)
    portfolio = set(selected.index)

    # --------------------------------------------------------
    # Next-month returns
    # --------------------------------------------------------

    next_month = month_end + pd.offsets.MonthEnd(1)

    if next_month not in month_ends:
        continue

    stock_returns = []

    for symbol in portfolio:

        ticker = symbol + ".NS"
        s = close[ticker].dropna()

        start_dates = s.index[s.index > month_end]
        end_dates = s.index[s.index <= next_month]

        if len(start_dates) == 0 or len(end_dates) == 0:
            continue

        start_date = start_dates[0]
        end_date = end_dates[-1]

        if end_date <= start_date:
            continue

        ret = s.loc[end_date] / s.loc[start_date] - 1

        if pd.notna(ret):
            stock_returns.append(ret)

    if len(stock_returns) < max(5, PORTFOLIO_SIZE // 2):
        skipped_count += 1
        continue

    gross_return = np.mean(stock_returns)

    # Transaction cost.
    turnover = len(portfolio.symmetric_difference(previous_portfolio))

    if previous_portfolio:
        turnover_fraction = turnover / (
            len(portfolio) + len(previous_portfolio)
        )
    else:
        turnover_fraction = 1.0

    net_return = gross_return - COST * turnover_fraction

    portfolio_returns.append(net_return)
    portfolio_dates.append(next_month)

    previous_portfolio = portfolio
    rebalance_count += 1

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

returns = pd.Series(
    portfolio_returns,
    index=pd.DatetimeIndex(portfolio_dates)
).sort_index()

if returns.empty:
    raise RuntimeError("No strategy returns were produced.")

equity = (1 + returns).cumprod()

total_return = equity.iloc[-1] - 1
years = len(returns) / 12
cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

volatility = returns.std(ddof=1) * np.sqrt(12)

sharpe = (
    returns.mean() / returns.std(ddof=1) * np.sqrt(12)
    if returns.std(ddof=1) > 0 else np.nan
)

drawdown = equity / equity.cummax() - 1
max_drawdown = drawdown.min()
winning_months = (returns > 0).mean()

print()
print("=" * 60)
print("PLAN 2 + ATR FILTER RESULTS")
print("=" * 60)

print(f"ATR period: {ATR_PERIOD}")
print(f"Maximum ATR/Price: {MAX_ATR_PCT:.2%}")
print(f"Cost assumption: {COST:.2%}")

print(f"Total return: {total_return:.2%}")
print(f"CAGR: {cagr:.2%}")
print(f"Annual volatility: {volatility:.2%}")
print(f"Sharpe ratio: {sharpe:.2f}")
print(f"Maximum drawdown: {max_drawdown:.2%}")
print(f"Winning months: {winning_months:.2%}")
print(f"Months tested: {len(returns)}")
print(f"Rebalances: {rebalance_count}")
print(f"Skipped months: {skipped_count}")

pd.DataFrame({
    "date": returns.index,
    "monthly_return": returns.values,
    "equity": equity.values
}).to_csv("plan2_atr_monthly_returns.csv", index=False)

print("Saved: plan2_atr_monthly_returns.csv")
