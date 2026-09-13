import yfinance as yf
import pandas as pd
import numpy as np

print("QUANT MOMENTUM SCREENER")
print("=======================")

# Read stock universe
universe = pd.read_csv("universe.csv")

results = []

for symbol in universe["symbol"].dropna():
    print(f"Processing {symbol}...")

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

        # Momentum
        r3 = close.iloc[-1] / close.iloc[-64] - 1
        r6 = close.iloc[-1] / close.iloc[-127] - 1
        r12 = close.iloc[-1] / close.iloc[-253] - 1

        # Volatility
        daily_returns = close.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)

        # Trend
        sma200 = close.rolling(200).mean().iloc[-1]
        trend = close.iloc[-1] / sma200 - 1

        # Quantitative score
        score = (
            r3 * 0.20
            + r6 * 0.30
            + r12 * 0.50
            + trend * 0.20
            - volatility * 0.10
        )

        results.append({
            "Stock": symbol,
            "3M Return": r3,
            "6M Return": r6,
            "12M Return": r12,
            "Volatility": volatility,
            "Trend": trend,
            "Score": score
        })

    except Exception as e:
        print(f"Skipped {symbol}: {e}")

df = pd.DataFrame(results)

if df.empty:
    print("No valid results.")
else:
    df = df.sort_values("Score", ascending=False)
    df["Rank"] = range(1, len(df) + 1)

    print("\nTOP 15 STOCKS")
    print("=============")
    print(df.head(15).to_string(index=False))

    df.to_csv("ranking.csv", index=False)

    print("\nRanking saved to ranking.csv")
    print("CALCULATION: WORKING")
