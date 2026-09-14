
import pandas as pd
import numpy as np
import yfinance as yf

# ============================================================
# PLAN 2 ROBUSTNESS TEST
#
# Tests whether the strategy remains reasonable when we make
# small changes to assumptions/parameters.
#
# Base strategy:
# 12-2 momentum -> Top Momentum -> lowest ID -> Top ID
# -> equal-weight portfolio
#
# Tests:
#   Transaction cost: 0.25%, 0.50%, 0.75%, 1.00%
#   Momentum pool:    75, 100, 150
#   ID pool:          30, 50, 75
#   Portfolio size:   10, 15, 20
#
# The script runs combinations and prints a summary.
#
# Research only. No trading.
# ============================================================

MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"
START = "2019-01-01"
END = "2026-09-30"

COSTS = [0.0025, 0.0050, 0.0075, 0.0100]
MOMENTUM_POOLS = [75, 100, 150]
ID_POOLS = [30, 50, 75]
PORTFOLIO_SIZES = [10, 15, 20]

# To keep the experiment manageable, the normal/base
# combinations are tested first, followed by the full grid.
BASE_MOMENTUM = 100
BASE_ID = 50
BASE_PORTFOLIO = 15

# ------------------------------------------------------------
# Load historical membership
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

def members_at(date):
    valid = [d for d in membership_dates if d <= date]
    return membership_by_date[valid[-1]] if valid else set()

symbols = sorted(membership["symbol"].unique())
tickers = [s + ".NS" for s in symbols]

# ------------------------------------------------------------
# Download prices
# ------------------------------------------------------------

print("Downloading price data...")

frames = []

for i in range(0, len(tickers), 25):
    batch = tickers[i:i+25]

    try:
        x = yf.download(
            batch,
            start=START,
            end=END,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column"
        )

        if x.empty:
            continue

        if isinstance(x.columns, pd.MultiIndex):
            x = x["Close"]
        else:
            x = x[["Close"]]
            x.columns = batch

        frames.append(x)

        print(f"Downloaded {min(i+25, len(tickers))}/{len(tickers)}")

    except Exception as e:
        print("Batch failed:", e)

if not frames:
    raise RuntimeError("No price data downloaded.")

close = pd.concat(frames, axis=1)
close = close.loc[:, ~close.columns.duplicated()]
close = close.dropna(axis=1, how="all")
close = close.sort_index()

print(f"Stocks with usable data: {close.shape[1]}")
print(f"Trading days: {len(close)}")

# ------------------------------------------------------------
# Completed month ends
# ------------------------------------------------------------

month_ends = close.resample("ME").last().index

last_completed = (
    pd.Timestamp.today().to_period("M").start_time
    - pd.Timedelta(days=1)
)

month_ends = month_ends[
    (month_ends <= last_completed)
    & (month_ends >= pd.Timestamp("2020-02-29"))
]

# ------------------------------------------------------------
# Pre-calculate monthly strategy ingredients
#
# This makes the parameter grid much faster.
# ------------------------------------------------------------

print("Calculating monthly candidate rankings...")

monthly_candidates = {}

for month_end in month_ends:

    members = members_at(month_end)

    if not members:
        continue

    end_2m = month_end - pd.DateOffset(months=2)
    end_12m = month_end - pd.DateOffset(months=12)

    records = []

    for symbol in members:

        ticker = symbol + ".NS"

        if ticker not in close.columns:
            continue

        s = close[ticker].dropna()

        if len(s) < 270:
            continue

        d12s = s.index[s.index <= end_12m]
        d2s = s.index[s.index <= end_2m]

        if not len(d12s) or not len(d2s):
            continue

        d12 = d12s[-1]
        d2 = d2s[-1]

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

        records.append(
            (symbol, momentum, id_value)
        )

    if records:
        df = pd.DataFrame(
            records,
            columns=["symbol", "momentum", "id"]
        )

        monthly_candidates[month_end] = df

print(f"Months prepared: {len(monthly_candidates)}")

# ------------------------------------------------------------
# Run one parameter set
# ------------------------------------------------------------

def run_strategy(momentum_pool, id_pool, portfolio_size, cost):

    returns = []
    dates = []

    previous_portfolio = set()
    skipped = 0

    for month_end in month_ends:

        if month_end not in monthly_candidates:
            skipped += 1
            continue

        df = monthly_candidates[month_end]

        # Step 1: momentum ranking.
        top_momentum = (
            df.sort_values("momentum", ascending=False)
            .head(momentum_pool)
        )

        top_momentum = top_momentum[
            top_momentum["momentum"] > 0
        ]

        if len(top_momentum) < portfolio_size:
            skipped += 1
            continue

        # Step 2: ID ranking.
        top_id = (
            top_momentum
            .sort_values("id", ascending=True)
            .head(id_pool)
        )

        if len(top_id) < portfolio_size:
            skipped += 1
            continue

        selected = top_id.head(portfolio_size)

        portfolio = set(selected["symbol"])

        # Next completed month.
        next_month = month_end + pd.offsets.MonthEnd(1)

        if next_month not in month_ends:
            continue

        stock_returns = []

        for symbol in portfolio:

            ticker = symbol + ".NS"

            if ticker not in close.columns:
                continue

            s = close[ticker].dropna()

            start_dates = s.index[s.index > month_end]
            end_dates = s.index[s.index <= next_month]

            if not len(start_dates) or not len(end_dates):
                continue

            start_date = start_dates[0]
            end_date = end_dates[-1]

            if end_date <= start_date:
                continue

            r = s.loc[end_date] / s.loc[start_date] - 1

            if pd.notna(r):
                stock_returns.append(r)

        if len(stock_returns) < max(
            5, portfolio_size // 2
        ):
            skipped += 1
            continue

        gross_return = np.mean(stock_returns)

        turnover = len(
            portfolio.symmetric_difference(previous_portfolio)
        )

        if previous_portfolio:
            turnover_fraction = turnover / (
                len(portfolio) + len(previous_portfolio)
            )
        else:
            turnover_fraction = 1.0

        net_return = (
            gross_return
            - cost * turnover_fraction
        )

        returns.append(net_return)
        dates.append(next_month)

        previous_portfolio = portfolio

    if not returns:
        return None

    r = pd.Series(
        returns,
        index=pd.DatetimeIndex(dates)
    ).sort_index()

    equity = (1 + r).cumprod()

    total_return = equity.iloc[-1] - 1

    years = len(r) / 12

    cagr = (
        equity.iloc[-1] ** (1 / years) - 1
        if years > 0 else np.nan
    )

    volatility = (
        r.std(ddof=1) * np.sqrt(12)
    )

    sharpe = (
        r.mean() / r.std(ddof=1) * np.sqrt(12)
        if r.std(ddof=1) > 0 else np.nan
    )

    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()

    winning_months = (r > 0).mean()

    return {
        "momentum_pool": momentum_pool,
        "id_pool": id_pool,
        "portfolio_size": portfolio_size,
        "cost": cost,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "winning_months": winning_months,
        "months": len(r),
        "skipped": skipped
    }

# ------------------------------------------------------------
# Test grid
# ------------------------------------------------------------

results = []

total_tests = (
    len(COSTS)
    * len(MOMENTUM_POOLS)
    * len(ID_POOLS)
    * len(PORTFOLIO_SIZES)
)

count = 0

print()
print(f"Running {total_tests} robustness combinations...")

for cost in COSTS:
    for momentum_pool in MOMENTUM_POOLS:
        for id_pool in ID_POOLS:
            for portfolio_size in PORTFOLIO_SIZES:

                # A portfolio cannot exceed the ID pool.
                if portfolio_size > id_pool:
                    continue

                count += 1

                result = run_strategy(
                    momentum_pool,
                    id_pool,
                    portfolio_size,
                    cost
                )

                if result is not None:
                    results.append(result)

                print(
                    f"Test {count}: "
                    f"cost={cost:.2%}, "
                    f"mom={momentum_pool}, "
                    f"id={id_pool}, "
                    f"portfolio={portfolio_size}"
                )

# ------------------------------------------------------------
# Save all results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    ["sharpe", "cagr"],
    ascending=False
)

results_df.to_csv(
    "plan2_robustness_results.csv",
    index=False
)

# ------------------------------------------------------------
# Base configuration
# ------------------------------------------------------------

base = results_df[
    (results_df["momentum_pool"] == BASE_MOMENTUM)
    & (results_df["id_pool"] == BASE_ID)
    & (results_df["portfolio_size"] == BASE_PORTFOLIO)
].sort_values("cost")

# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print()
print("=" * 80)
print("PLAN 2 ROBUSTNESS SUMMARY")
print("=" * 80)

print()
print("BASE STRATEGY ACROSS TRANSACTION COSTS")
print(base.to_string(index=False))

print()
print("=" * 80)
print("BEST 10 CONFIGURATIONS BY SHARPE")
print("=" * 80)

print(
    results_df.head(10).to_string(index=False)
)

print()
print("=" * 80)
print("MEDIAN RESULTS ACROSS ALL TESTED CONFIGURATIONS")
print("=" * 80)

print(
    f"Median CAGR: {results_df['cagr'].median():.2%}"
)
print(
    f"Median Sharpe: {results_df['sharpe'].median():.2f}"
)
print(
    f"Median Max DD: {results_df['max_drawdown'].median():.2%}"
)
print(
    f"Median Winning Months: "
    f"{results_df['winning_months'].median():.2%}"
)

print()
print("=" * 80)
print("ROBUSTNESS CHECK")
print("=" * 80)

# How many configurations remain profitable?
positive = (results_df["cagr"] > 0).mean()

# How many have Sharpe >= 1?
good_sharpe = (results_df["sharpe"] >= 1).mean()

# How many have CAGR >= 20%?
good_cagr = (results_df["cagr"] >= 0.20).mean()

print(
    f"Configurations with positive CAGR: {positive:.2%}"
)
print(
    f"Configurations with Sharpe >= 1: {good_sharpe:.2%}"
)
print(
    f"Configurations with CAGR >= 20%: {good_cagr:.2%}"
)

print()
print("Saved: plan2_robustness_results.csv")
