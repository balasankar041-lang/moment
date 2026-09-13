import yfinance as yf
import pandas as pd
import numpy as np

TOP_N = 15
YEARS = 10
TRADING_COST = 0.0015
MIN_HISTORY_MONTHS = 36

print("NIFTY 500 TOP-15 CONCENTRATION BACKTEST")
print("=======================================")

# -----------------------------
# NIFTY 500 UNIVERSE
# -----------------------------
url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

universe = pd.read_csv(
    url,
    storage_options={"User-Agent": "Mozilla/5.0"}
)

symbols = (
    universe["Symbol"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.upper()
)

symbols = [
    s + ".NS"
    for s in symbols
    if s.isalpha() and len(s) <= 15
]

symbols = list(dict.fromkeys(symbols))

print(f"Universe candidates: {len(symbols)}")
print(f"History: {YEARS} years")
print(f"Portfolio: Top {TOP_N}")
print("Downloading data...")

# -----------------------------
# DOWNLOAD DATA
# -----------------------------
all_data = []
BATCH_SIZE = 50

for start in range(0, len(symbols), BATCH_SIZE):

    batch = symbols[start:start + BATCH_SIZE]

    print(
        f"Batch {start + 1}-"
        f"{min(start + BATCH_SIZE, len(symbols))}"
    )

    try:
        data = yf.download(
            batch,
            period=f"{YEARS}y",
            auto_adjust=True,
            progress=False,
            threads=True
        )

        if data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):

            if "Close" not in data.columns.levels[0]:
                continue

            data = data["Close"]

        else:

            if "Close" not in data.columns:
                continue

            data = data[["Close"]]
            data.columns = [batch[0]]

        all_data.append(data)

    except Exception as e:
        print("Batch error:", e)

if not all_data:
    raise RuntimeError("No price data downloaded.")

prices = pd.concat(all_data, axis=1)

prices = prices.loc[
    :,
    ~prices.columns.duplicated()
]

prices = prices.dropna(
    axis=1,
    how="all"
)

prices = prices.ffill()

print(
    f"Stocks with usable data: "
    f"{len(prices.columns)}"
)

if len(prices.columns) < 450:
    raise RuntimeError(
        "Too few stocks downloaded."
    )

# -----------------------------
# MONTHLY DATA
# -----------------------------
monthly = prices.resample("ME").last()

portfolio_returns = []
previous_stocks = set()

# Stock statistics
stock_stats = {}

total_turnover = 0
rebalance_count = 0
replacement_count = 0

# -----------------------------
# BACKTEST
# -----------------------------
for i in range(MIN_HISTORY_MONTHS, len(monthly) - 1):

    current = monthly.iloc[i]

    p3 = monthly.iloc[i - 3]
    p6 = monthly.iloc[i - 6]
    p12 = monthly.iloc[i - 12]

    ret3 = current / p3 - 1
    ret6 = current / p6 - 1
    ret12 = current / p12 - 1

    start_date = monthly.index[i - 12]
    end_date = monthly.index[i]

    daily = prices.loc[
        start_date:end_date
    ]

    valid_days = daily.count()

    eligible = valid_days[
        valid_days >= 200
    ].index

    if len(eligible) < TOP_N:
        continue

    # -----------------------------
    # VOLATILITY
    # -----------------------------
    daily_returns = daily[eligible].pct_change()

    vol = (
        daily_returns.std()
        * np.sqrt(252)
    )

    # -----------------------------
    # RISK-ADJUSTED MOMENTUM
    # -----------------------------
    score = (
        0.20 * ret3 +
        0.30 * ret6 +
        0.50 * ret12
    )

    score = score / vol

    score = score.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    score = score[score > 0]

    if len(score) < TOP_N:
        continue

    selected = score.nlargest(TOP_N).index

    next_month = monthly.iloc[i + 1]

    stock_returns = (
        next_month[selected]
        / current[selected]
        - 1
    ).dropna()

    if len(stock_returns) < 10:
        continue

    portfolio_return = stock_returns.mean()

    # -----------------------------
    # STOCK STATISTICS
    # -----------------------------
    for stock in stock_returns.index:

        if stock not in stock_stats:
            stock_stats[stock] = {
                "months_held": 0,
                "gross_contribution": 0.0,
                "positive_months": 0,
                "negative_months": 0,
                "best_month": -999.0,
                "worst_month": 999.0
            }

        r = stock_returns[stock]

        stock_stats[stock]["months_held"] += 1

        # Equal-weight contribution
        contribution = r / TOP_N

        stock_stats[stock][
            "gross_contribution"
        ] += contribution

        if r > 0:
            stock_stats[stock]["positive_months"] += 1
        else:
            stock_stats[stock]["negative_months"] += 1

        stock_stats[stock]["best_month"] = max(
            stock_stats[stock]["best_month"],
            r
        )

        stock_stats[stock]["worst_month"] = min(
            stock_stats[stock]["worst_month"],
            r
        )

    # -----------------------------
    # TURNOVER
    # -----------------------------
    current_stocks = set(
        stock_returns.index
    )

    if previous_stocks:

        changed = len(
            current_stocks.symmetric_difference(
                previous_stocks
            )
        )

        turnover = (
            changed / (2 * TOP_N)
        )

        total_turnover += turnover
        replacement_count += changed / 2

        portfolio_return -= (
            turnover * TRADING_COST
        )

        rebalance_count += 1

    previous_stocks = current_stocks

    portfolio_returns.append({
        "Date": monthly.index[i + 1],
        "Return": portfolio_return
    })

# -----------------------------
# RESULTS
# -----------------------------
result = pd.DataFrame(
    portfolio_returns
)

if result.empty:
    raise RuntimeError(
        "Backtest produced no results."
    )

result = result.set_index("Date")

equity = (
    1 + result["Return"]
).cumprod()

years_tested = (
    result.index[-1]
    - result.index[0]
).days / 365.25

cagr = (
    equity.iloc[-1]
    ** (1 / years_tested)
) - 1

annual_volatility = (
    result["Return"].std()
    * np.sqrt(12)
)

sharpe = (
    result["Return"].mean()
    / result["Return"].std()
) * np.sqrt(12)

drawdown = (
    equity / equity.cummax()
) - 1

max_drawdown = drawdown.min()

win_rate = (
    result["Return"] > 0
).mean()

total_return = (
    equity.iloc[-1] - 1
)

# -----------------------------
# STOCK ANALYSIS
# -----------------------------
stats = pd.DataFrame.from_dict(
    stock_stats,
    orient="index"
)

stats.index.name = "Stock"

stats["holding_percentage"] = (
    stats["months_held"]
    / len(result)
)

stats["positive_month_percentage"] = (
    stats["positive_months"]
    / stats["months_held"]
)

stats = stats.sort_values(
    "gross_contribution",
    ascending=False
)

stats.to_csv(
    "stock_concentration_analysis.csv"
)

# -----------------------------
# YEARLY RETURNS
# -----------------------------
yearly_returns = (
    (1 + result["Return"])
    .groupby(result.index.year)
    .prod()
    - 1
)

yearly_returns.to_csv(
    "yearly_returns.csv"
)

# -----------------------------
# SAVE MAIN RESULT
# -----------------------------
result["Equity"] = equity
result["Drawdown"] = drawdown

result.to_csv(
    "nifty500_top15_concentration_backtest.csv"
)

# -----------------------------
# PRINT MAIN RESULT
# -----------------------------
print()
print("================================")
print("BACKTEST RESULT")
print("================================")

print(
    f"Period: "
    f"{result.index[0].date()} "
    f"to "
    f"{result.index[-1].date()}"
)

print(f"Total return: {total_return:.2%}")
print(f"CAGR: {cagr:.2%}")
print(f"Annual volatility: {annual_volatility:.2%}")
print(f"Sharpe ratio: {sharpe:.2f}")
print(f"Maximum drawdown: {max_drawdown:.2%}")
print(f"Winning months: {win_rate:.2%}")
print(f"Months tested: {len(result)}")

# -----------------------------
# TURNOVER
# -----------------------------
if rebalance_count > 0:

    average_turnover = (
        total_turnover /
        rebalance_count
    )

    print()
    print("TURNOVER ANALYSIS")
    print("=================")
    print(
        f"Rebalances: {rebalance_count}"
    )
    print(
        f"Average monthly turnover: "
        f"{average_turnover:.2%}"
    )
    print(
        f"Estimated replacements: "
        f"{replacement_count:.0f}"
    )

# -----------------------------
# TOP STOCKS
# -----------------------------
print()
print("TOP 20 STOCK CONTRIBUTIONS")
print("===========================")

for stock, row in stats.head(20).iterrows():

    print(
        f"{stock}: "
        f"held {int(row['months_held'])} months | "
        f"contribution "
        f"{row['gross_contribution']:.2%} | "
        f"positive months "
        f"{row['positive_month_percentage']:.1%}"
    )

# -----------------------------
# YEARLY RETURNS
# -----------------------------
print()
print("YEAR-BY-YEAR RETURNS")
print("=====================")

for year, value in yearly_returns.items():
    print(
        f"{year}: {value:.2%}"
    )

print()
print("Saved:")
print("nifty500_top15_concentration_backtest.csv")
print("stock_concentration_analysis.csv")
print("yearly_returns.csv")
print("BACKTEST COMPLETE")
