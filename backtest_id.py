
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

# ============================================================
# PLAN 2 + RSI EXPERIMENT
#
# Core strategy:
# 1. Historical Nifty 500 membership
# 2. 12-2 momentum: 12-month return excluding latest 2 months
# 3. Rank entire universe by momentum -> Top 100
# 4. Information Discretion (ID): lower is better
# 5. Select Top 50 by lowest ID
# 6. Select Top 15 for the portfolio
#
# New experiment:
# 7. RSI(14) is used only as a confirmation filter.
#    We test RSI >= 50.
#
# IMPORTANT:
# This is a research/backtest experiment. It does NOT place trades.
# ============================================================

MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"

START = "2019-01-01"
END = "2026-09-30"

TOP_MOMENTUM = 100
TOP_ID = 50
PORTFOLIO_SIZE = 15

COST = 0.005          # 0.50% transaction-cost assumption
RSI_PERIOD = 14
RSI_MIN = 50

# ------------------------------------------------------------
# Load historical membership
# ------------------------------------------------------------

membership = pd.read_csv(MEMBERSHIP_FILE)
membership["effective_date"] = pd.to_datetime(membership["effective_date"])

membership["symbol"] = (
    membership["symbol"]
    .astype(str)
    .str.strip()
    .str.upper()
)

membership = membership.drop_duplicates(["effective_date", "symbol"])

symbols = sorted(membership["symbol"].unique())

print(f"Historical membership records: {len(membership)}")
print(f"Historical symbols: {len(symbols)}")

# ------------------------------------------------------------
# Download prices
# ------------------------------------------------------------

tickers = [s + ".NS" for s in symbols]

print("Downloading price data...")

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

        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            # Single ticker case
            close = data[["Close"]]
            close.columns = batch

        frames.append(close)

        print(f"Downloaded {min(i+25, len(tickers))}/{len(tickers)}")

    except Exception as e:
        print(f"Batch failed {i}-{i+25}: {e}")

if not frames:
    raise RuntimeError("No price data downloaded.")

prices = pd.concat(frames, axis=1)
prices = prices.loc[:, ~prices.columns.duplicated()]
prices = prices.dropna(axis=1, how="all")
prices = prices.sort_index()

print(f"Stocks with usable price data: {prices.shape[1]}")
print(f"Trading days: {len(prices)}")

# ------------------------------------------------------------
# RSI calculation
# ------------------------------------------------------------

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # If there are no losses, RSI is 100.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100)

    return rsi


rsi = prices.apply(calculate_rsi, period=RSI_PERIOD)

# ------------------------------------------------------------
# Historical membership lookup
# ------------------------------------------------------------

membership_by_date = {}

for date, group in membership.groupby("effective_date"):
    membership_by_date[date] = set(group["symbol"])

membership_dates = sorted(membership_by_date.keys())

def get_members_for_date(date):
    valid = [d for d in membership_dates if d <= date]

    if not valid:
        return set()

    latest = valid[-1]
    return membership_by_date[latest]

# ------------------------------------------------------------
# Month-end dates
# Only completed months are used.
# ------------------------------------------------------------

month_ends = prices.resample("ME").last().index

last_completed_month = pd.Timestamp.today().to_period("M").start_time - pd.Timedelta(days=1)
month_ends = month_ends[month_ends <= last_completed_month]

# Need enough history for 12-2 momentum.
month_ends = month_ends[month_ends >= pd.Timestamp("2020-02-29")]

# ------------------------------------------------------------
# Strategy
# ------------------------------------------------------------

portfolio_returns = []
portfolio_dates = []
previous_portfolio = set()

rebalance_count = 0
skipped_count = 0

for month_end in month_ends:

    # Historical membership available at this date.
    members = get_members_for_date(month_end)

    if not members:
        skipped_count += 1
        continue

    available = []

    for symbol in members:
        ticker = symbol + ".NS"

        if ticker not in prices.columns:
            continue

        series = prices[ticker].dropna()

        # Need sufficient price history.
        if len(series) < 270:
            continue

        available.append(symbol)

    if not available:
        skipped_count += 1
        continue

    # --------------------------------------------------------
    # 12-2 momentum
    #
    # Approximate:
    # price around 12 months ago -> price around 2 months ago
    # --------------------------------------------------------

    end_2m = month_end - pd.DateOffset(months=2)
    end_12m = month_end - pd.DateOffset(months=12)

    momentum_values = {}
    id_values = {}
    rsi_values = {}

    for symbol in available:

        ticker = symbol + ".NS"

        s = prices[ticker].dropna()

        p12_candidates = s.index[s.index <= end_12m]
        p2_candidates = s.index[s.index <= end_2m]

        if len(p12_candidates) == 0 or len(p2_candidates) == 0:
            continue

        d12 = p12_candidates[-1]
        d2 = p2_candidates[-1]

        p12 = s.loc[d12]
        p2 = s.loc[d2]

        if p12 <= 0:
            continue

        momentum = p2 / p12 - 1

        # ----------------------------------------------------
        # ID daily path over the same 12-2 window.
        # Zero-return days are excluded.
        # ----------------------------------------------------

        path = s.loc[d12:d2].pct_change().dropna()

        positive = (path > 0).sum()
        negative = (path < 0).sum()

        total_directional = positive + negative

        if total_directional == 0:
            continue

        positive_pct = positive / total_directional
        negative_pct = negative / total_directional

        id_value = np.sign(momentum) * (
            negative_pct - positive_pct
        )

        # ----------------------------------------------------
        # RSI at rebalance date
        # ----------------------------------------------------

        rsi_series = rsi[ticker].dropna()
        rsi_candidates = rsi_series.index[rsi_series.index <= month_end]

        if len(rsi_candidates) == 0:
            continue

        rsi_date = rsi_candidates[-1]
        rsi_value = rsi_series.loc[rsi_date]

        if pd.isna(rsi_value):
            continue

        momentum_values[symbol] = momentum
        id_values[symbol] = id_value
        rsi_values[symbol] = rsi_value

    if len(momentum_values) < TOP_MOMENTUM:
        skipped_count += 1
        continue

    # --------------------------------------------------------
    # Step 1: Top 100 momentum
    # --------------------------------------------------------

    momentum_series = pd.Series(momentum_values).sort_values(
        ascending=False
    )

    top100 = momentum_series.head(TOP_MOMENTUM)

    # Only positive momentum candidates.
    top100 = top100[top100 > 0]

    if len(top100) < PORTFOLIO_SIZE:
        skipped_count += 1
        continue

    # --------------------------------------------------------
    # Step 2: Lowest ID among Top 100
    # --------------------------------------------------------

    id_series = pd.Series({
        symbol: id_values[symbol]
        for symbol in top100.index
        if symbol in id_values
    })

    id_series = id_series.sort_values(ascending=True)

    top50 = id_series.head(TOP_ID)

    if len(top50) < PORTFOLIO_SIZE:
        skipped_count += 1
        continue

    # --------------------------------------------------------
    # Step 3: RSI filter
    #
    # Keep only RSI >= 50.
    # --------------------------------------------------------

    rsi_series = pd.Series({
        symbol: rsi_values[symbol]
        for symbol in top50.index
        if symbol in rsi_values
    })

    rsi_pass = rsi_series[rsi_series >= RSI_MIN]

    # Rank remaining stocks by ID.
    rsi_pass = rsi_pass.index.intersection(top50.index)

    selected = top50.loc[
        top50.index.intersection(rsi_pass)
    ].head(PORTFOLIO_SIZE)

    # If RSI leaves fewer than 15 stocks, skip rather than
    # weakening the filter.
    if len(selected) < PORTFOLIO_SIZE:
        skipped_count += 1
        continue

    portfolio = set(selected.index)

    # --------------------------------------------------------
    # Calculate next month's equal-weight return
    # --------------------------------------------------------

    next_month = month_end + pd.offsets.MonthEnd(1)

    if next_month not in month_ends:
        # No next completed month available yet.
        continue

    stock_returns = []

    for symbol in portfolio:

        ticker = symbol + ".NS"

        s = prices[ticker].dropna()

        start_candidates = s.index[s.index > month_end]
        end_candidates = s.index[s.index <= next_month]

        if len(start_candidates) == 0 or len(end_candidates) == 0:
            continue

        start_date = start_candidates[0]
        end_date = end_candidates[-1]

        if end_date <= start_date:
            continue

        ret = s.loc[end_date] / s.loc[start_date] - 1

        if pd.notna(ret):
            stock_returns.append(ret)

    if len(stock_returns) < max(5, PORTFOLIO_SIZE // 2):
        skipped_count += 1
        continue

    gross_return = np.mean(stock_returns)

    # --------------------------------------------------------
    # Transaction costs
    # --------------------------------------------------------

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
    if returns.std(ddof=1) > 0
    else np.nan
)

drawdown = equity / equity.cummax() - 1
max_drawdown = drawdown.min()

winning_months = (returns > 0).mean()

print()
print("=" * 60)
print("PLAN 2 + RSI RESULTS")
print("=" * 60)

print(f"RSI period: {RSI_PERIOD}")
print(f"RSI minimum: {RSI_MIN}")
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

# Save monthly returns for inspection.
output = pd.DataFrame({
    "date": returns.index,
    "monthly_return": returns.values,
    "equity": equity.values
})

output.to_csv("plan2_rsi_monthly_returns.csv", index=False)

print()
print("Saved: plan2_rsi_monthly_returns.csv")
