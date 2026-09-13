import pandas as pd
import yfinance as yf

print("NIFTY 500 QUANT SCREENER")
print("========================")

# Temporary test universe.
# We will replace this with the full current Nifty 500 list
# after confirming the data pipeline.
stocks = [
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

results = []

for symbol in stocks:
    try:
        data = yf.download(
            symbol,
            period="2y",
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            continue

        close = data["Close"].squeeze().dropna()

        if len(close) < 253:
            continue

        r3 = close.iloc[-1] / close.iloc[-64] - 1
        r6 = close.iloc[-1] / close.iloc[-127] - 1
        r12 = close.iloc[-1] / close.iloc[-253] - 1

        daily = close.pct_change().dropna()
        volatility = daily.std() * (252 ** 0.5)

        sma200 = close.rolling(200).mean().iloc[-1]
        trend = close.iloc[-1] / sma200 - 1

        score = (
            r3 * 0.20
            + r6 * 0.30
            + r12 * 0.50
            + trend * 0.20
            - volatility * 0.10
        )

        results.append({
            "Stock": symbol,
            "3M": r3,
            "6M": r6,
            "12M": r12,
            "Volatility": volatility,
            "Trend": trend,
            "Score": score
        })

    except Exception as e:
        print(f"Skipped {symbol}: {e}")

df = pd.DataFrame(results)

if df.empty:
    print("No results.")
else:
    df = df.sort_values("Score", ascending=False)
    df["Rank"] = range(1, len(df) + 1)

    print("\nTOP STOCKS")
    print(df.head(10).to_string(index=False))

    df.to_csv("ranking.csv", index=False)

    print("\nRanking saved to ranking.csv")
    print("CALCULATION: WORKING")
