import yfinance as yf
import pandas as pd
import numpy as np

TOP_N = 15
YEARS = 10
TRADING_COST = 0.0015

print("NIFTY 500 RISK-ADJUSTED MOMENTUM BACKTEST")
print("=========================================")

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
    .unique()
)

symbols = [s + ".NS" for s in symbols]

print(f"Universe: {len(symbols)} stocks")
print(f"History: {YEARS} years")
print(f"Portfolio: Top {TOP_N}")
print("Downloading data...")

# -----------------------------
# DOWNLOAD PRICE DATA
# -----------------------------
prices = yf.download(
    symbols,
    period=f"{YEARS}y",
    auto_adjust=True,
    progress=False,
    threads=True
)["Close"]

prices = prices.dropna(axis=1, how="all")
prices = prices.ffill()

print(f"Stocks with data: {len(prices.columns)}")

# -----------------------------
# MONTHLY DATA
# -----------------------------
monthly = prices.resample("ME").last()

portfolio_returns = []
previous_stocks = set()

# -----------------------------
# MOMENTUM CALCULATION
# -----------------------------
for i in range(12, len(monthly) - 1):

    current = monthly.iloc[i]

    price_3m = monthly.iloc[i - 3]
    price_6m = monthly.iloc[i - 6]
    price_12m = monthly.iloc[i - 12]

    ret_3m = current / price_3m - 1
    ret_6m = current / price_6m - 1
    ret_12m = current / price_12m - 1

    # 12-month volatility
    daily_start = prices.index[
        prices.index <= monthly.index[i - 12]
    ]

    daily_end = prices.index[
        prices.index <= monthly.index[i]
    ]

    if len(daily_start) == 0 or len(daily_end) == 0:
        continue

    start_date = daily_start[-1]
    end_date = daily_end[-1]

    daily_prices = prices.loc[start_date:end_date]

    daily_returns = daily_prices.pct_change()

    volatility = daily_returns.std() * np.sqrt(252)

    # -----------------------------
    # RISK-ADJUSTED MOMENTUM SCORE
    # -----------------------------
    score = (
        0.20 * ret_3m +
        0.30 * ret_6m +
        0.50 * ret_12m
    )

    # Penalize high volatility
    risk_adjusted_score = score / volatility

    # Remove invalid values
    risk_adjusted_score = risk_adjusted_score.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    # Only positive momentum
    risk_adjusted_score = risk_adjusted_score[
        risk_adjusted_score > 0
    ]

    if len(risk_adjusted_score) < TOP_N:
        continue

    # -----------------------------
    # SELECT TOP 15
    # -----------------------------
    selected = risk_adjusted_score.nlargest(TOP_N).index

    next_month = monthly.iloc[i + 1]

    stock_returns = (
        next_month[selected] /
        current[selected] - 1
    ).dropna()

    if stock_returns.empty:
        continue

    portfolio_return = stock_returns.mean()

    # -----------------------------
    # TURNOVER COST
    # -----------------------------
    current_stocks = set(selected)

    if previous_stocks:

        changed = len(
            current_stocks.symmetric_difference(
                previous_stocks
            )
        )

        turnover = changed / (2 * TOP_N)

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
result = pd.DataFrame(portfolio_returns)

if result.empty:
    raise RuntimeError(
        "Backtest produced no results."
    )

result = result.set_index("Date")

# Equity curve
equity = (
    1 + result["Return"]
).cumprod()

# CAGR
years_tested = (
    result.index[-1] -
    result.index[0]
).days / 365.25

cagr = (
    equity.iloc[-1] **
    (1 / years_tested)
) - 1

# Annual volatility
volatility = (
    result["Return"].std()
    * np.sqrt(12)
)

# Sharpe ratio
sharpe = (
    result["Return"].mean()
    / result["Return"].std()
) * np.sqrt(12)

# Maximum drawdown
drawdown = (
    equity / equity.cummax()
) - 1

max_drawdown = drawdown.min()

# Winning months
win_rate = (
    result["Return"] > 0
).mean()

# Total return
total_return = (
    equity.iloc[-1] - 1
)

result["Equity"] = equity
result["Drawdown"] = drawdown

# Save detailed results
result.to_csv(
    "nifty500_risk_adjusted_backtest.csv"
)

# -----------------------------
# PRINT RESULTS
# -----------------------------
print()
print("==========================================")
print("RISK-ADJUSTED TOP-15 BACKTEST RESULT")
print("==========================================")

print(
    f"Period: "
    f"{result.index[0].date()} "
    f"to "
    f"{result.index[-1].date()}"
)

print(
    f"Total return: "
    f"{total_return:.2%}"
)

print(
    f"CAGR: "
    f"{cagr:.2%}"
)

print(
    f"Annual volatility: "
    f"{volatility:.2%}"
)

print(
    f"Sharpe ratio: "
    f"{sharpe:.2f}"
)

print(
    f"Maximum drawdown: "
    f"{max_drawdown:.2%}"
)

print(
    f"Winning months: "
    f"{win_rate:.2%}"
)

print(
    f"Months tested: "
    f"{len(result)}"
)

print()
print(
    "Saved: "
    "nifty500_risk_adjusted_backtest.csv"
)

print("BACKTEST COMPLETE")
