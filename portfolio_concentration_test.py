import pandas as pd
import numpy as np
import yfinance as yf

TOP_MOMENTUM = 100
TOP_ID = 50

# -----------------------------
# Load historical membership
# -----------------------------
membership = pd.read_csv("nifty500_membership_timeline.csv")

membership["date"] = pd.to_datetime(membership["effective_date"])
membership["symbol"] = membership["symbol"].astype(str).str.strip()

symbols = sorted(membership["symbol"].dropna().unique())

print("Historical symbols:", len(symbols))

# -----------------------------
# Download prices
# -----------------------------
prices = yf.download(
    [s + ".NS" for s in symbols],
    start="2019-01-01",
    end="2026-09-16",
    auto_adjust=True,
    progress=False,
    threads=True
)["Close"]

if isinstance(prices, pd.Series):
    prices = prices.to_frame()

prices.columns = [
    c.replace(".NS", "") if isinstance(c, str) else c
    for c in prices.columns
]

prices = prices.sort_index()

monthly_prices = prices.resample("ME").last()

# -----------------------------
# Build exact Plan2 portfolios
# -----------------------------
portfolio_records = []

for date in monthly_prices.index:

    available = membership[membership["date"] <= date]

    if available.empty:
        continue

    latest_snapshot = available["date"].max()

    universe = available[
        available["date"] == latest_snapshot
    ]["symbol"].tolist()

    universe = [
        s for s in universe
        if s in monthly_prices.columns
    ]

    if not universe:
        continue

    current = monthly_prices.loc[date, universe]

    # 12-month momentum
    past_date = date - pd.DateOffset(months=12)

    past_dates = monthly_prices.index[
        monthly_prices.index <= past_date
    ]

    if len(past_dates) == 0:
        continue

    past = monthly_prices.loc[
        past_dates[-1],
        universe
    ]

    momentum = current / past - 1

    momentum = momentum.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(momentum) < TOP_MOMENTUM:
        continue

    top100 = momentum.nlargest(TOP_MOMENTUM)

    # -----------------------------
    # Plan2 ID
    # -----------------------------
    start_date = date - pd.DateOffset(months=12)
    end_date = date - pd.DateOffset(months=2)

    daily = prices.loc[
        start_date:end_date,
        top100.index
    ]

    ids = {}

    for symbol in top100.index:

        series = daily[symbol].dropna()

        if len(series) < 100:
            continue

        returns = series.pct_change().dropna()
        returns = returns[returns != 0]

        if len(returns) == 0:
            continue

        negative_pct = (returns < 0).mean()
        positive_pct = (returns > 0).mean()

        sign = np.sign(top100[symbol])

        ids[symbol] = sign * (
            negative_pct - positive_pct
        )

    if not ids:
        continue

    id_series = pd.Series(ids).sort_values()

    selected = id_series.head(TOP_ID)

    for symbol in selected.index:
        portfolio_records.append({
            "date": date,
            "symbol": symbol
        })

portfolio = pd.DataFrame(portfolio_records)

if portfolio.empty:
    raise RuntimeError("No portfolio selections generated.")

print("Portfolio selection records:", len(portfolio))

# -----------------------------
# Actual monthly stock returns
# -----------------------------
stock_returns = monthly_prices.pct_change()

# -----------------------------
# Calculate portfolio statistics
# -----------------------------
monthly_results = []
correlation_results = []

dates = sorted(portfolio["date"].unique())

for date in dates:

    holdings = portfolio.loc[
        portfolio["date"] == date,
        "symbol"
    ].tolist()

    holdings = [
        s for s in holdings
        if s in stock_returns.columns
    ]

    if len(holdings) < 10:
        continue

    returns = stock_returns.loc[date, holdings].dropna()

    if len(returns) < 10:
        continue

    # Equal-weight portfolio
    weights = pd.Series(
        1 / len(returns),
        index=returns.index
    )

    contribution = weights * returns

    # Top stock return contribution
    contribution_sorted = contribution.sort_values(
        ascending=False
    )

    total_return = contribution.sum()

    if total_return != 0:

        top5_contribution = (
            contribution_sorted.head(5).sum()
            / total_return
        )

        top10_contribution = (
            contribution_sorted.head(10).sum()
            / total_return
        )

    else:
        top5_contribution = np.nan
        top10_contribution = np.nan

    # Concentration using absolute contribution
    absolute_contribution = contribution.abs()

    absolute_total = absolute_contribution.sum()

    if absolute_total > 0:
        top5_abs = (
            absolute_contribution
            .sort_values(ascending=False)
            .head(5)
            .sum()
            / absolute_total
        )

        top10_abs = (
            absolute_contribution
            .sort_values(ascending=False)
            .head(10)
            .sum()
            / absolute_total
        )
    else:
        top5_abs = np.nan
        top10_abs = np.nan

    # Herfindahl concentration
    contribution_weights = (
        absolute_contribution / absolute_total
    )

    effective_bets = 1 / (
        contribution_weights ** 2
    ).sum()

    monthly_results.append({
        "date": date,
        "holdings": len(returns),
        "portfolio_return": total_return,
        "top5_return_contribution": top5_contribution,
        "top10_return_contribution": top10_contribution,
        "top5_absolute_contribution": top5_abs,
        "top10_absolute_contribution": top10_abs,
        "effective_bets": effective_bets
    })

    # -----------------------------
    # Holding correlation
    # -----------------------------
    daily_window = prices.loc[
        date - pd.DateOffset(months=3):date,
        holdings
    ].pct_change()

    corr = daily_window.corr()

    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    )

    avg_corr = upper.stack().mean()

    correlation_results.append({
        "date": date,
        "average_pairwise_correlation": avg_corr
    })

results = pd.DataFrame(monthly_results)
correlations = pd.DataFrame(correlation_results)

# -----------------------------
# Turnover
# -----------------------------
monthly_sets = (
    portfolio.groupby("date")["symbol"]
    .apply(set)
    .sort_index()
)

turnovers = []

previous = None

for date, current in monthly_sets.items():

    if previous is None:
        previous = current
        continue

    added = current - previous
    removed = previous - current

    turnover = len(removed) / TOP_ID

    turnovers.append({
        "date": date,
        "turnover": turnover,
        "added": len(added),
        "removed": len(removed)
    })

    previous = current

turnover_df = pd.DataFrame(turnovers)

# -----------------------------
# Final report
# -----------------------------
print("\n===================================")
print("PORTFOLIO CONCENTRATION TEST")
print("===================================")

print(
    "Months tested:",
    len(results)
)

print(
    "Average Top-5 return contribution:",
    round(
        results["top5_return_contribution"].median() * 100,
        2
    ),
    "%"
)

print(
    "Average Top-10 return contribution:",
    round(
        results["top10_return_contribution"].median() * 100,
        2
    ),
    "%"
)

print(
    "Median Top-5 absolute contribution:",
    round(
        results["top5_absolute_contribution"].median() * 100,
        2
    ),
    "%"
)

print(
    "Median Top-10 absolute contribution:",
    round(
        results["top10_absolute_contribution"].median() * 100,
        2
    ),
    "%"
)

print(
    "Median effective number of bets:",
    round(
        results["effective_bets"].median(),
        2
    )
)

print(
    "Minimum effective number of bets:",
    round(
        results["effective_bets"].min(),
        2
    )
)

print(
    "Median average stock correlation:",
    round(
        correlations[
            "average_pairwise_correlation"
        ].median(),
        3
    )
)

print(
    "Maximum average stock correlation:",
    round(
        correlations[
            "average_pairwise_correlation"
        ].max(),
        3
    )
)

if not turnover_df.empty:

    print(
        "Average monthly turnover:",
        round(
            turnover_df["turnover"].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Maximum monthly turnover:",
        round(
            turnover_df["turnover"].max() * 100,
            2
        ),
        "%"
    )

print(
    "Maximum single-stock weight:",
    round(
        100 / TOP_ID,
        2
    ),
    "%"
)

print("\nTop stocks by selection frequency:")
print(
    portfolio["symbol"]
    .value_counts()
    .head(15)
    .to_string()
)

# -----------------------------
# Save detailed results
# -----------------------------
results.to_csv(
    "portfolio_concentration_monthly.csv",
    index=False
)

correlations.to_csv(
    "portfolio_correlation_monthly.csv",
    index=False
)

if not turnover_df.empty:
    turnover_df.to_csv(
        "portfolio_turnover_monthly.csv",
        index=False
    )

print("\nDetailed result files created.")
print("Plan2 parameters were NOT changed.")
