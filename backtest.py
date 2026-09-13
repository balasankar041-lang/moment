import yfinance as yf
import pandas as pd
import numpy as np

TOP_N = 15
YEARS = 10
TRADING_COST = 0.0015
MIN_HISTORY_MONTHS = 36

print("NIFTY 500 ROBUST TOP-15 BACKTEST")
print("================================")

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
# DOWNLOAD IN BATCHES
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
# MONTHLY PRICES
# -----------------------------
monthly = prices.resample("ME").last()

portfolio_returns = []
previous_stocks = set()

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

    daily_returns = daily[eligible].pct_change()

    vol = (
        daily_returns.std()
        * np.sqrt(252)
    )

    # Risk-adjusted momentum
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

    # Transaction costs
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

        portfolio_return -= (
            turnover * TRADING_COST
        )

    portfolio_returns.append({
        "Date": monthly.index[i + 1],
        "Return": portfolio_return
    })

    previous_stocks = current_stocks

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
# YEARLY RETURNS
# -----------------------------
yearly_returns = (
    (1 + result["Return"])
    .groupby(result.index.year)
    .prod()
    - 1
)

best_year = yearly_returns.idxmax()
worst_year = yearly_returns.idxmin()

# -----------------------------
# BENCHMARK: NIFTY 500
# -----------------------------
print()
print("Downloading NIFTY 500 benchmark...")

benchmark = yf.download(
    "^CRSLDX",
    start=result.index[0],
    end=result.index[-1] + pd.Timedelta(days=31),
    auto_adjust=True,
    progress=False
)

if not benchmark.empty:

    if isinstance(benchmark.columns, pd.MultiIndex):
        benchmark_close = benchmark["Close"].squeeze()
    else:
        benchmark_close = benchmark["Close"]

    benchmark_close = benchmark_close.dropna()

    benchmark_return = (
        benchmark_close.iloc[-1]
        / benchmark_close.iloc[0]
        - 1
    )

    benchmark_years = (
        benchmark_close.index[-1]
        - benchmark_close.index[0]
    ).days / 365.25

    benchmark_cagr = (
        (benchmark_close.iloc[-1]
         / benchmark_close.iloc[0])
        ** (1 / benchmark_years)
        - 1
    )

else:

    benchmark_return = np.nan
    benchmark_cagr = np.nan

# -----------------------------
# SAVE
# -----------------------------
result["Equity"] = equity
result["Drawdown"] = drawdown

result.to_csv(
    "nifty500_robust_top15_backtest.csv"
)

yearly_returns.to_csv(
    "yearly_returns.csv"
)

# -----------------------------
# PRINT MAIN RESULT
# -----------------------------
print()
print("================================")
print("ROBUST TOP-15 BACKTEST RESULT")
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
# YEARLY PERFORMANCE
# -----------------------------
print()
print("YEAR-BY-YEAR RETURNS")
print("=====================")

for year, value in yearly_returns.items():
    print(f"{year}: {value:.2%}")

print()
print(f"Best year: {best_year} ({yearly_returns[best_year]:.2%})")
print(f"Worst year: {worst_year} ({yearly_returns[worst_year]:.2%})")

# -----------------------------
# BENCHMARK
# -----------------------------
print()
print("BENCHMARK COMPARISON")
print("====================")

if not np.isnan(benchmark_cagr):

    print(
        f"Nifty 500 CAGR: "
        f"{benchmark_cagr:.2%}"
    )

    print(
        f"Strategy CAGR: "
        f"{cagr:.2%}"
    )

    print(
        f"CAGR advantage: "
        f"{cagr - benchmark_cagr:.2%}"
    )

else:

    print("Nifty 500 benchmark data unavailable.")

print()
print("Saved:")
print("nifty500_robust_top15_backtest.csv")
print("yearly_returns.csv")
print("BACKTEST COMPLETE")
