import pandas as pd
import numpy as np
import yfinance as yf

TOP_MOMENTUM = 100
TOP_ID = 50

# ---------------------------------
# Load historical Nifty 500 members
# ---------------------------------
membership = pd.read_csv("nifty500_membership_timeline.csv")

membership["date"] = pd.to_datetime(membership["effective_date"])
membership["symbol"] = membership["symbol"].astype(str).str.strip()

symbols = sorted(membership["symbol"].dropna().unique())

print("Historical symbols:", len(symbols))

# ---------------------------------
# Download daily prices
# ---------------------------------
prices = yf.download(
    [s + ".NS" for s in symbols],
    start="2019-01-01",
    end="2026-09-01",
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

# Completed months only: exclude the in-progress September 2026 month.
completed_month_end = pd.Timestamp("2026-08-31")
monthly_prices = monthly_prices[
    monthly_prices.index <= completed_month_end
]

# ---------------------------------
# Build exact Plan2 portfolios
# ---------------------------------
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

    # ---------------------------------
    # Plan2 ID
    # ---------------------------------
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

# ---------------------------------
# Calculate monthly portfolio return
# ---------------------------------
monthly_returns = []

for date in sorted(portfolio["date"].unique()):

    holdings = portfolio.loc[
        portfolio["date"] == date,
        "symbol"
    ].tolist()

    holdings = [
        s for s in holdings
        if s in monthly_prices.columns
    ]

    if len(holdings) < 10:
        continue

    # Return from previous month-end to current month-end
    dates = monthly_prices.index

    idx = dates.get_loc(date)

    if idx == 0:
        continue

    previous_date = dates[idx - 1]

    current_prices = monthly_prices.loc[
        date,
        holdings
    ]

    previous_prices = monthly_prices.loc[
        previous_date,
        holdings
    ]

    returns = (
        current_prices / previous_prices - 1
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(returns) < 10:
        continue

    # Equal-weight Plan2
    portfolio_return = returns.mean()

    # Absolute contribution
    contributions = returns / len(returns)

    absolute_contribution = contributions.abs()

    total_absolute = absolute_contribution.sum()

    if total_absolute > 0:

        top5_abs = (
            absolute_contribution
            .sort_values(ascending=False)
            .head(5)
            .sum()
            / total_absolute
        )

        top10_abs = (
            absolute_contribution
            .sort_values(ascending=False)
            .head(10)
            .sum()
            / total_absolute
        )

    else:

        top5_abs = np.nan
        top10_abs = np.nan

    # Effective bets
    if total_absolute > 0:

        contribution_weights = (
            absolute_contribution / total_absolute
        )

        effective_bets = 1 / (
            contribution_weights ** 2
        ).sum()

    else:

        effective_bets = np.nan

    monthly_returns.append({
        "date": date,
        "portfolio_return": portfolio_return,
        "top5_absolute_contribution": top5_abs,
        "top10_absolute_contribution": top10_abs,
        "effective_bets": effective_bets,
        "holdings": len(returns)
    })

results = pd.DataFrame(monthly_returns)

if results.empty:
    raise RuntimeError("No monthly portfolio results generated.")

# ---------------------------------
# Calculate drawdown
# ---------------------------------
results["equity"] = (
    1 + results["portfolio_return"]
).cumprod()

results["peak"] = results["equity"].cummax()

results["drawdown"] = (
    results["equity"] / results["peak"] - 1
)

# ---------------------------------
# Identify worst drawdown months
# ---------------------------------
worst = results.sort_values(
    "portfolio_return"
).head(15).copy()

print("\n===================================")
print("CONCENTRATION vs DRAWDOWN TEST")
print("===================================")

print("Months tested:", len(results))

print(
    "Worst monthly return:",
    round(
        worst.iloc[0]["portfolio_return"] * 100,
        2
    ),
    "%"
)

print(
    "Maximum portfolio drawdown:",
    round(
        results["drawdown"].min() * 100,
        2
    ),
    "%"
)

print("\nWorst 15 portfolio months:")

display_columns = [
    "date",
    "portfolio_return",
    "drawdown",
    "top5_absolute_contribution",
    "top10_absolute_contribution",
    "effective_bets",
    "holdings"
]

print(
    worst[display_columns].to_string(
        index=False
    )
)

# ---------------------------------
# Compare worst months vs all months
# ---------------------------------
bottom_10 = results.nsmallest(
    max(1, int(len(results) * 0.10)),
    "portfolio_return"
)

normal = results[
    ~results.index.isin(bottom_10.index)
]

print("\n===================================")
print("WORST 10% MONTHS vs NORMAL MONTHS")
print("===================================")

print(
    "Worst 10% average return:",
    round(
        bottom_10["portfolio_return"].mean() * 100,
        2
    ),
    "%"
)

print(
    "Worst 10% median effective bets:",
    round(
        bottom_10["effective_bets"].median(),
        2
    )
)

print(
    "Normal median effective bets:",
    round(
        normal["effective_bets"].median(),
        2
    )
)

print(
    "Worst 10% median Top-5 absolute contribution:",
    round(
        bottom_10[
            "top5_absolute_contribution"
        ].median() * 100,
        2
    ),
    "%"
)

print(
    "Normal median Top-5 absolute contribution:",
    round(
        normal[
            "top5_absolute_contribution"
        ].median() * 100,
        2
    ),
    "%"
)

print(
    "Worst 10% median Top-10 absolute contribution:",
    round(
        bottom_10[
            "top10_absolute_contribution"
        ].median() * 100,
        2
    ),
    "%"
)

print(
    "Normal median Top-10 absolute contribution:",
    round(
        normal[
            "top10_absolute_contribution"
        ].median() * 100,
        2
    ),
    "%"
)

# ---------------------------------
# Concentrated drawdown months
# ---------------------------------
concentration_threshold = (
    results["effective_bets"].quantile(0.10)
)

concentrated = results[
    results["effective_bets"]
    <= concentration_threshold
]

concentrated_drawdown_months = (
    concentrated[
        concentrated["portfolio_return"] < 0
    ]
)

print("\n===================================")
print("CONCENTRATED NEGATIVE MONTHS")
print("===================================")

print(
    "10th percentile effective bets:",
    round(
        concentration_threshold,
        2
    )
)

print(
    "Concentrated negative months:",
    len(concentrated_drawdown_months)
)

print(
    "Total negative months:",
    int(
        (results["portfolio_return"] < 0).sum()
    )
)

if len(concentrated_drawdown_months) > 0:

    print(
        "Average return during concentrated "
        "negative months:",
        round(
            concentrated_drawdown_months[
                "portfolio_return"
            ].mean() * 100,
            2
        ),
        "%"
    )

# ---------------------------------
# Correlation test
# ---------------------------------
corr_values = []

for date in results["date"]:

    holdings = portfolio.loc[
        portfolio["date"] == date,
        "symbol"
    ].tolist()

    holdings = [
        s for s in holdings
        if s in prices.columns
    ]

    if len(holdings) < 10:
        continue

    window_start = (
        date - pd.DateOffset(months=3)
    )

    daily_returns = prices.loc[
        window_start:date,
        holdings
    ].pct_change()

    corr = daily_returns.corr()

    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    )

    avg_corr = upper.stack().mean()

    corr_values.append({
        "date": date,
        "average_correlation": avg_corr
    })

correlation_df = pd.DataFrame(corr_values)

print("\n===================================")
print("CORRELATION")
print("===================================")

if not correlation_df.empty:

    print(
        "Median correlation:",
        round(
            correlation_df[
                "average_correlation"
            ].median(),
            3
        )
    )

    print(
        "Maximum correlation:",
        round(
            correlation_df[
                "average_correlation"
            ].max(),
            3
        )
    )

# ---------------------------------
# Save detailed results
# ---------------------------------
results.to_csv(
    "concentration_drawdown_monthly.csv",
    index=False
)

correlation_df.to_csv(
    "concentration_drawdown_correlation.csv",
    index=False
)

print("\nDetailed files created.")

print("\nPlan2 parameters were NOT changed.")
