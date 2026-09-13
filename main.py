import yfinance as yf
import numpy as np

SYMBOL = "RELIANCE.NS"

print("QUANT MOMENTUM SCREENER")
print("=======================")

data = yf.download(
    SYMBOL,
    period="2y",
    auto_adjust=True,
    progress=False
)

if data.empty:
    print("No market data received.")
    raise SystemExit

close = data["Close"].squeeze().dropna()

# Returns
return_3m = close.iloc[-1] / close.iloc[-64] - 1
return_6m = close.iloc[-1] / close.iloc[-127] - 1
return_12m = close.iloc[-1] / close.iloc[-253] - 1

# Daily volatility
daily_returns = close.pct_change().dropna()
volatility = daily_returns.std() * np.sqrt(252)

# 200-day trend
sma_200 = close.rolling(200).mean().iloc[-1]
trend = close.iloc[-1] / sma_200 - 1

# Preliminary score
score = (
    return_3m * 0.20
    + return_6m * 0.30
    + return_12m * 0.50
    + trend * 0.20
    - volatility * 0.10
)

print(f"Stock: {SYMBOL}")
print(f"Latest price: ₹{close.iloc[-1]:.2f}")
print(f"3-month return: {return_3m:.2%}")
print(f"6-month return: {return_6m:.2%}")
print(f"12-month return: {return_12m:.2%}")
print(f"Annual volatility: {volatility:.2%}")
print(f"Trend vs 200 DMA: {trend:.2%}")
print(f"Quantitative score: {score:.4f}")
print("=======================")
print("Calculation: WORKING")
