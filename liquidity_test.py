import pandas as pd
import numpy as np
import yfinance as yf

TOP_MOMENTUM = 100
TOP_ID = 50

# Minimum liquidity thresholds for research
MIN_AVG_DAILY_VALUE = 5_000_000       # ₹50 lakh/day
MIN_MEDIAN_DAILY_VALUE = 2_500_000    # ₹25 lakh/day

# -------------------------------------------------
# Load historical membership
# -------------------------------------------------
membership = pd.read_csv("nifty500_membership_timeline.csv")

membership["date"] = pd.to_datetime(membership["effective_date"])
membership["symbol"] = membership["symbol"].astype(str).str.strip()

symbols = sorted(membership["symbol"].dropna().unique())

print("Historical symbols:", len(symbols))

# -------------------------------------------------
# Download OHLCV data
# -------------------------------------------------
data = yf.download(
    [s + ".NS" for s in symbols],
    start="2019-01-01",
    end="2026-09-16",
    auto_adjust=True,
    progress=False,
    threads=True
)

close = data["Close"]
volume = data["Volume"]

if isinstance(close, pd.Series):
    close = close.to_frame()

if isinstance(volume, pd.Series):
    volume = volume.to_frame()

close.columns = [
    c.replace(".NS", "") if isinstance(c, str) else c
    for c in close.columns
]

volume.columns = [
    c.replace(".NS", "") if isinstance(c, str) else c
    for c in volume.columns
]

close = close.sort_index()
volume = volume.sort_index()

monthly_prices = close.resample("ME").last()

# -------------------------------------------------
# Build exact Plan2 portfolio
# -------------------------------------------------
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

    # -------------------------------------------------
    # Plan2 ID
    # -------------------------------------------------
    start_date = date - pd.DateOffset(months=12)
    end_date = date - pd.DateOffset(months=2)

    daily = close.loc[
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

# -------------------------------------------------
# Liquidity analysis
# -------------------------------------------------
results = []

for date in sorted(portfolio["date"].unique()):

    holdings = portfolio.loc[
        portfolio["date"] == date,
        "symbol"
    ].tolist()

    holdings = [
        s for s in holdings
        if s in close.columns and s in volume.columns
    ]

    if len(holdings) < 10:
        continue

    # Use previous 60 trading days to estimate liquidity
    end_date = date
    start_date = date - pd.DateOffset(days=100)

    prices_window = close.loc[
        start_date:end_date,
        holdings
    ]

    volume_window = volume.loc[
        start_date:end_date,
        holdings
    ]

    # Daily traded value = price × volume
    traded_value = prices_window * volume_window

    avg_daily_value = traded_value.mean()
    median_daily_value = traded_value.median()

    # Portfolio-level statistics
    avg_value = avg_daily_value.mean()
    median_value = median_daily_value.median()

    # Count stocks below thresholds
    below_avg = (
        avg_daily_value < MIN_AVG_DAILY_VALUE
    ).sum()

    below_median = (
        median_daily_value < MIN_MEDIAN_DAILY_VALUE
    ).sum()

    # Percentage of portfolio with weak liquidity
    weak_avg_pct = below_avg / len(holdings)
    weak_median_pct = below_median / len(holdings)

    # Worst stock in this portfolio
    worst_symbol = avg_daily_value.idxmin()
    worst_value = avg_daily_value.min()

    results.append({
        "date": date,
        "holdings": len(holdings),
        "portfolio_avg_daily_value": avg_value,
        "portfolio_median_daily_value": median_value,
        "stocks_below_avg_threshold": below_avg,
        "stocks_below_median_threshold": below_median,
        "weak_avg_pct": weak_avg_pct,
        "weak_median_pct": weak_median_pct,
        "worst_symbol": worst_symbol,
        "worst_avg_daily_value": worst_value
    })

results = pd.DataFrame(results)

if results.empty:
    raise RuntimeError("No liquidity results generated.")

# -------------------------------------------------
# Summary
# -------------------------------------------------
print("\n===================================")
print("LIQUIDITY TEST")
print("===================================")

print("Months tested:", len(results))

print(
    "Median portfolio average daily traded value: ₹",
    f"{results['portfolio_avg_daily_value'].median():,.0f}"
)

print(
    "Minimum portfolio average daily traded value: ₹",
    f"{results['portfolio_avg_daily_value'].min():,.0f}"
)

print(
    "Median portfolio median daily traded value: ₹",
    f"{results['portfolio_median_daily_value'].median():,.0f}"
)

print(
    "Median % stocks below ₹50 lakh/day:",
    round(results["weak_avg_pct"].median() * 100, 2),
    "%"
)

print(
    "Maximum % stocks below ₹50 lakh/day:",
    round(results["weak_avg_pct"].max() * 100, 2),
    "%"
)

print(
    "Median % stocks below ₹25 lakh/day:",
    round(results["weak_median_pct"].median() * 100, 2),
    "%"
)

print(
    "Maximum % stocks below ₹25 lakh/day:",
    round(results["weak_median_pct"].max() * 100, 2),
    "%"
)

# -------------------------------------------------
# Worst liquidity months
# -------------------------------------------------
print("\nWorst liquidity months:")

worst_months = results.sort_values(
    "portfolio_avg_daily_value"
).head(10)

print(
    worst_months[
        [
            "date",
            "portfolio_avg_daily_value",
            "weak_avg_pct",
            "worst_symbol",
            "worst_avg_daily_value"
        ]
    ].to_string(index=False)
)

# -------------------------------------------------
# Worst individual stocks
# -------------------------------------------------
print("\nWorst individual liquidity observations:")

individual_records = []

for date in sorted(portfolio["date"].unique()):

    holdings = portfolio.loc[
        portfolio["date"] == date,
        "symbol"
    ].tolist()

    holdings = [
        s for s in holdings
        if s in close.columns and s in volume.columns
    ]

    if not holdings:
        continue

    start_date = date - pd.DateOffset(days=100)

    traded_value = (
        close.loc[start_date:date, holdings]
        * volume.loc[start_date:date, holdings]
    )

    median_value = traded_value.median()

    for symbol in median_value.dropna().index:

        individual_records.append({
            "date": date,
            "symbol": symbol,
            "median_daily_traded_value":
                median_value[symbol]
        })

individual = pd.DataFrame(individual_records)

if not individual.empty:

    worst_individual = individual.sort_values(
        "median_daily_traded_value"
    ).head(20)

    print(
        worst_individual.to_string(index=False)
    )

# -------------------------------------------------
# Save results
# -------------------------------------------------
results.to_csv(
    "liquidity_monthly.csv",
    index=False
)

if not individual.empty:
    individual.to_csv(
        "liquidity_individual.csv",
        index=False
    )

print("\nDetailed liquidity files created.")
print("Plan2 parameters were NOT changed.")
