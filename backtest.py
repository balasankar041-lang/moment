import yfinance as yf
import pandas as pd
import numpy as np

SYMBOLS = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "TCS.NS",
    "SBIN.NS",
    "ITC.NS",
    "BHARTIARTL.NS",
    "LT.NS",
    "AXISBANK.NS",
]

print("10-YEAR MOMENTUM BACKTEST")
print("=========================")

prices = yf.download(
    SYMBOLS,
    period="10y",
    auto_adjust=True,
    progress=False,
    threads=True
)["Close"]

prices = prices.dropna(axis=1, how="all").ffill()

monthly = prices.resample("ME").last()

returns = []

for i in range(12, len(monthly) - 1):

    current = monthly.iloc[i]
    previous = monthly.iloc[i - 12]

    momentum = (current / previous - 1).dropna()

    if len(momentum) < 5:
        continue

    # Select strongest 5 stocks
    selected = momentum.nlargest(5).index

    next_month = monthly.iloc[i + 1]

    portfolio_return = (
        next_month[selected] / current[selected] - 1
    ).dropna().mean()

    returns.append(portfolio_return)

returns = pd.Series(returns)

equity = (1 + returns).cumprod()

total_return = equity.iloc[-1] - 1

years = len(returns) / 12

cagr = equity.iloc[-1] ** (1 / years) - 1

volatility = returns.std() * np.sqrt(12)

sharpe = (
    returns.mean() / returns.std()
) * np.sqrt(12)

drawdown = equity / equity.cummax() - 1

max_drawdown = drawdown.min()

win_rate = (returns > 0).mean()

print()
print("BACKTEST RESULT")
print("================")
print(f"Total return: {total_return:.2%}")
print(f"CAGR: {cagr:.2%}")
print(f"Annual volatility: {volatility:.2%}")
print(f"Sharpe ratio: {sharpe:.2f}")
print(f"Maximum drawdown: {max_drawdown:.2%}")
print(f"Winning months: {win_rate:.2%}")
print(f"Months tested: {len(returns)}")

print()
print("BACKTEST COMPLETE")
