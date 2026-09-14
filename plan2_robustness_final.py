
import pandas as pd
import numpy as np
import yfinance as yf

# ============================================================
# PLAN 2 ROBUSTNESS TEST — CLEAN VERSION
#
# IMPORTANT:
# This file is standalone. It does NOT depend on backtest_id.py.
# It tests the original Plan 2 logic only.
#
# Strategy:
# 12-2 momentum -> Top 100 -> lowest ID -> Top 50 -> Top 15
# monthly rebalance, 0.50% base transaction cost.
#
# First it validates the BASE CASE.
# Only if the base case is close to our recorded Plan 2 baseline
# do we interpret the robustness grid.
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

BASE_MOMENTUM = 100
BASE_ID = 50
BASE_PORTFOLIO = 15
BASE_COST = 0.005

# Recorded Plan 2 benchmark.
EXPECTED_CAGR = 0.3342
EXPECTED_SHARPE = 1.52
EXPECTED_DD = -0.2765
EXPECTED_WIN = 0.7273

# ------------------------------------------------------------
# Membership
# ------------------------------------------------------------

membership = pd.read_csv(MEMBERSHIP_FILE)
membership["effective_date"] = pd.to_datetime(membership["effective_date"])
membership["symbol"] = (
    membership["symbol"].astype(str).str.strip().str.upper()
)
membership = membership.drop_duplicates(["effective_date", "symbol"])

membership_by_date = {
    d: set(g["symbol"])
    for d, g in membership.groupby("effective_date")
}
membership_dates = sorted(membership_by_date)

def members_at(date):
    valid = [d for d in membership_dates if d <= date]
    return membership_by_date[valid[-1]] if valid else set()

# ------------------------------------------------------------
# Download close prices
# ------------------------------------------------------------

symbols = sorted(membership["symbol"].unique())
tickers = [s + ".NS" for s in symbols]

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
# Prepare monthly Plan 2 ranking data
# ------------------------------------------------------------

print("Preparing monthly Plan 2 rankings...")

monthly_rankings = {}

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

        # ID = sign(momentum) * (% negative days - % positive days)
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

        monthly_rankings[month_end] = df

print(f"Ranking months prepared: {len(monthly_rankings)}")

# ------------------------------------------------------------
# Run a configuration
# ------------------------------------------------------------

def run_strategy(momentum_pool, id_pool, portfolio_size, cost):

    monthly_returns = []
    return_dates = []

    previous_portfolio = set()
    skipped = 0

    for month_end in month_ends:

        if month_end not in monthly_rankings:
            skipped += 1
            continue

        df = monthly_rankings[month_end]

        # 1. Momentum Top N
        top_momentum = (
            df.sort_values(
                ["momentum", "symbol"],
                ascending=[False, True]
            )
            .head(momentum_pool)
        )

        top_momentum = top_momentum[
            top_momentum["momentum"] > 0
        ]

        if len(top_momentum) < portfolio_size:
            skipped += 1
            continue

        # 2. Lowest ID
        top_id = (
            top_momentum
            .sort_values(
                ["id", "symbol"],
                ascending=[True, True]
            )
            .head(id_pool)
        )

        if len(top_id) < portfolio_size:
            skipped += 1
            continue

        # 3. Select portfolio
        selected = top_id.head(portfolio_size)
        portfolio = set(selected["symbol"])

        next_month = month_end + pd.offsets.MonthEnd(1)

        if next_month not in month_ends:
            continue

        stock_returns = []

        for symbol in portfolio:

            ticker = symbol + ".NS"

            if ticker not in close.columns:
                continue

            s = close[ticker].dropna()

            # IMPORTANT:
            # Entry = last available close ON month-end.
            # Exit  = last available close ON next month-end.
            start_dates = s.index[s.index <= month_end]
            end_dates = s.index[s.index <= next_month]

            if not len(start_dates) or not len(end_dates):
                continue

            start_date = start_dates[-1]
            end_date = end_dates[-1]

            if end_date <= start_date:
                continue

            ret = s.loc[end_date] / s.loc[start_date] - 1

            if pd.notna(ret):
                stock_returns.append(ret)

        if len(stock_returns) < max(
            5,
            portfolio_size // 2
        ):
            skipped += 1
            continue

        gross_return = np.mean(stock_returns)

        # Same turnover convention used by this experiment.
        changed = len(
            portfolio.symmetric_difference(previous_portfolio)
        )

        if previous_portfolio:
            turnover_fraction = changed / (
                len(portfolio) + len(previous_portfolio)
            )
        else:
            turnover_fraction = 1.0

        net_return = (
            gross_return
            - cost * turnover_fraction
        )

        monthly_returns.append(net_return)
        return_dates.append(next_month)

        previous_portfolio = portfolio

    if not monthly_returns:
        return None

    r = pd.Series(
        monthly_returns,
        index=pd.DatetimeIndex(return_dates)
    ).sort_index()

    equity = (1 + r).cumprod()

    total_return = equity.iloc[-1] - 1

    years = len(r) / 12

    cagr = (
        equity.iloc[-1] ** (1 / years) - 1
        if years > 0 else np.nan
    )

    volatility = r.std(ddof=1) * np.sqrt(12)

    sharpe = (
        r.mean() / r.std(ddof=1) * np.sqrt(12)
        if r.std(ddof=1) > 0 else np.nan
    )

    max_drawdown = (
        equity / equity.cummax() - 1
    ).min()

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
# BASE CASE FIRST
# ------------------------------------------------------------

print()
print("=" * 80)
print("BASE CASE VALIDATION")
print("=" * 80)

base = run_strategy(
    BASE_MOMENTUM,
    BASE_ID,
    BASE_PORTFOLIO,
    BASE_COST
)

if base is None:
    raise RuntimeError("BASE CASE FAILED: no returns produced.")

print(f"CAGR: {base['cagr']:.2%}")
print(f"Sharpe: {base['sharpe']:.2f}")
print(f"Max DD: {base['max_drawdown']:.2%}")
print(f"Winning months: {base['winning_months']:.2%}")
print(f"Months: {base['months']}")
print(f"Skipped: {base['skipped']}")

# ------------------------------------------------------------
# Decide whether base is close enough.
# ------------------------------------------------------------

base_ok = (
    abs(base["cagr"] - EXPECTED_CAGR) <= 0.05
    and abs(base["sharpe"] - EXPECTED_SHARPE) <= 0.20
    and abs(base["max_drawdown"] - EXPECTED_DD) <= 0.05
    and abs(base["winning_months"] - EXPECTED_WIN) <= 0.08
)

print()

if base_ok:
    print("BASE CASE STATUS: PASS")
    print("The base case is reasonably close to the recorded Plan 2 benchmark.")
else:
    print("BASE CASE STATUS: FAIL")
    print("STOP: robustness grid will NOT be interpreted.")
    print("The base case does not reproduce Plan 2 closely enough.")

# ------------------------------------------------------------
# Full grid ONLY after base validation.
# ------------------------------------------------------------

results = [base] if base_ok else []

if base_ok:

    print()
    print("=" * 80)
    print("RUNNING ROBUSTNESS GRID")
    print("=" * 80)

    for cost in COSTS:
        for momentum_pool in MOMENTUM_POOLS:
            for id_pool in ID_POOLS:
                for portfolio_size in PORTFOLIO_SIZES:

                    if (
                        momentum_pool == BASE_MOMENTUM
                        and id_pool == BASE_ID
                        and portfolio_size == BASE_PORTFOLIO
                        and cost == BASE_COST
                    ):
                        continue

                    if portfolio_size > id_pool:
                        continue

                    result = run_strategy(
                        momentum_pool,
                        id_pool,
                        portfolio_size,
                        cost
                    )

                    if result is not None:
                        results.append(result)

    results_df = pd.DataFrame(results)

    # Save only validated results.
    results_df.to_csv(
        "plan2_robustness_results_validated.csv",
        index=False
    )

    print()
    print("=" * 80)
    print("BEST 10 CONFIGURATIONS BY SHARPE")
    print("=" * 80)

    print(
        results_df.sort_values(
            ["sharpe", "cagr"],
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print()
    print("=" * 80)
    print("MEDIAN RESULTS")
    print("=" * 80)

    print(f"Median CAGR: {results_df['cagr'].median():.2%}")
    print(f"Median Sharpe: {results_df['sharpe'].median():.2f}")
    print(
        f"Median Max DD: "
        f"{results_df['max_drawdown'].median():.2%}"
    )
    print(
        f"Median Winning Months: "
        f"{results_df['winning_months'].median():.2%}"
    )

    print()
    print("=" * 80)
    print("ROBUSTNESS CHECK")
    print("=" * 80)

    print(
        f"Positive CAGR: "
        f"{(results_df['cagr'] > 0).mean():.2%}"
    )

    print(
        f"Sharpe >= 1: "
        f"{(results_df['sharpe'] >= 1).mean():.2%}"
    )

    print(
        f"CAGR >= 20%: "
        f"{(results_df['cagr'] >= 0.20).mean():.2%}"
    )

    print()
    print("Saved: plan2_robustness_results_validated.csv")

else:
    # Still save the failed base result for diagnosis.
    pd.DataFrame([base]).to_csv(
        "plan2_robustness_base_validation_failed.csv",
        index=False
    )

    print()
    print("Saved: plan2_robustness_base_validation_failed.csv")
