import pandas as pd
import numpy as np
import yfinance as yf

# Plan2 settings
TOP_MOMENTUM = 100
TOP_ID = 50

# Read historical membership
membership = pd.read_csv("nifty500_membership_timeline.csv")

membership["date"] = pd.to_datetime(membership["date"])
membership["symbol"] = membership["symbol"].astype(str).str.strip()

# Use the same historical universe as Plan2
symbols = sorted(membership["symbol"].dropna().unique())

print("Historical symbols:", len(symbols))

# Download daily prices
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

# Calculate monthly portfolio selections
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

    past = monthly_prices.loc[past_dates[-1], universe]

    momentum = current / past - 1
    momentum = momentum.replace([np.inf, -np.inf], np.nan).dropna()

    if len(momentum) < TOP_MOMENTUM:
        continue

    top100 = momentum.nlargest(TOP_MOMENTUM)

    # Plan2 ID:
    # 12-2 month daily path
    start_date = date - pd.DateOffset(months=12)
    end_date = date - pd.DateOffset(months=2)

    daily = prices.loc[start_date:end_date, top100.index]

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
# 1. Top 5 / Top 10 selection frequency
# -------------------------------------------------

counts = portfolio["symbol"].value_counts()

total_selections = len(portfolio)

top5_share = counts.head(5).sum() / total_selections
top10_share = counts.head(10).sum() / total_selections

print("\n=== SELECTION CONCENTRATION ===")
print("Top 5 selection share:", round(top5_share * 100, 2), "%")
print("Top 10 selection share:", round(top10_share * 100, 2), "%")

print("\nMost frequently selected stocks:")
print(counts.head(20).to_string())

# -------------------------------------------------
# 2. Single-stock maximum portfolio weight
# -------------------------------------------------

max_weight = 1 / TOP_ID

print("\n=== SINGLE STOCK RISK ===")
print("Maximum equal-weight stock allocation:",
      round(max_weight * 100, 2), "%")

# -------------------------------------------------
# 3. Monthly turnover
# -------------------------------------------------

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

    removed = previous - current
    added = current - previous

    turnover = len(removed) / TOP_ID

    turnovers.append({
        "date": date,
        "turnover": turnover,
        "removed": len(removed),
        "added": len(added)
    })

    previous = current

turnover_df = pd.DataFrame(turnovers)

print("\n=== TURNOVER ===")

if not turnover_df.empty:
    print(
        "Average monthly turnover:",
        round(turnover_df["turnover"].mean() * 100, 2),
        "%"
    )

    print(
        "Maximum monthly turnover:",
        round(turnover_df["turnover"].max() * 100, 2),
        "%"
    )

# -------------------------------------------------
# 4. Effective number of bets
# -------------------------------------------------

weights = counts / counts.sum()

effective_bets = 1 / (weights ** 2).sum()

print("\n=== EFFECTIVE NUMBER OF BETS ===")
print("Effective number of bets:",
      round(effective_bets, 2))

# -------------------------------------------------
# Final summary
# -------------------------------------------------

print("\n=== PORTFOLIO CONCENTRATION TEST ===")

print("Top 5 selection share:",
      round(top5_share * 100, 2), "%")

print("Top 10 selection share:",
      round(top10_share * 100, 2), "%")

print("Maximum single-stock weight:",
      round(max_weight * 100, 2), "%")

print("Effective number of bets:",
      round(effective_bets, 2))

print("\nThis test does NOT change Plan2.")
