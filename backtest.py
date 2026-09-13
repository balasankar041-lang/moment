import yfinance as yf
import pandas as pd
import numpy as np

TOP_N = 15
YEARS = 10
TRADING_COST = 0.0015

print("NIFTY 500 TOP-15 BACKTEST")
print("=========================")

# Current Nifty 500 universe
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

# Monthly prices
monthly = prices.resample("ME").last()

portfolio_returns = []
previous_stocks = set()

for i in range(12, len(monthly) - 1):

    current = monthly.iloc[i]
    one_year_ago = monthly.iloc[i - 12]

    momentum = (current / one_year_ago - 1).dropna()

    # Only positive momentum stocks
    momentum = momentum[momentum > 0]

    if len(momentum) < TOP_N:
        continue

    selected = momentum.nlargest(TOP_N).index

    next_month = monthly.iloc[i + 1]

    stock_returns = (
        next_month[selected] / current[selected] - 1
    ).dropna()

    if stock_returns.empty:
        continue

    portfolio_return = stock_returns.mean()

    # Approximate turnover cost
    current_stocks = set(selected)

    if previous_stocks:
        changed = len(
            current_stocks.symmetric_difference(previous_stocks)
        )

        turnover = changed / (2 * TOP_N)
        portfolio_return -= turnover * TRADING_COST

    portfolio_returns.append({
        "Date": monthly.index[i + 1],
        "Return": portfolio_return
    })

    previous_stocks = current_stocks

result = pd.DataFrame(portfolio_returns)

if result.empty:
    raise RuntimeError("Backtest produced no results.")

result = result.set_index("Date")

# Equity curve
equity = (1 + result["Return"]).cumprod()

# CAGR
years_tested = (
    result.index[-1] - result.index[0]
).days / 365.25

cagr = equity.iloc[-1] ** (1 / years_tested) - 1

# Volatility
volatility = result["Return"].std() * np.sqrt(12)

# Sharpe
sharpe = (
    result["Return"].mean() /
    result["Return"].std()
) * np.sqrt(12)

# Drawdown
drawdown = equity / equity.cummax() - 1

max_drawdown = drawdown.min()

# Win rate
win_rate = (result["Return"] > 0).mean()

# Total return
total_return = equity.iloc[-1] - 1

print()
print("================================")
print("NIFTY 500 TOP-15 BACKTEST RESULT")
print("================================")

print(f"Period: {result.index[0].date()} to {result.index[-1].date()}")
print(f"Total return: {total_return:.2%}")
print(f"CAGR: {cagr:.2%}")
print(f"Annual volatility: {volatility:.2%}")
print(f"Sharpe ratio: {sharpe:.2f}")
print(f"Maximum drawdown: {max_drawdown:.2%}")
print(f"Winning months: {win_rate:.2%}")
print(f"Months tested: {len(result)}")

result["Equity"] = equity
result["Drawdown"] = drawdown

result.to_csv("nifty500_top15_backtest.csv")

print()
print("Saved: nifty500_top15_backtest.csv")
print("BACKTEST COMPLETE")
