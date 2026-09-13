import yfinance as yf
import pandas as pd

print("QUANT MOMENTUM SCREENER")
print("=======================")

symbol = "RELIANCE.NS"

data = yf.download(
    symbol,
    period="2y",
    auto_adjust=True,
    progress=False
)

if data.empty:
    print("No market data received.")
else:
    price = float(data["Close"].iloc[-1])
    return_6m = (price / float(data["Close"].iloc[-127])) - 1
    return_12m = (price / float(data["Close"].iloc[-253])) - 1

    print(f"Stock: {symbol}")
    print(f"Latest price: ₹{price:.2f}")
    print(f"6-month return: {return_6m:.2%}")
    print(f"12-month return: {return_12m:.2%}")
    print("Market data connection: WORKING")
