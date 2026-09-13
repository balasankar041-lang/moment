import yfinance as yf
import pandas as pd
import numpy as np

print("QUANT MOMENTUM SCREENER")
print("=======================")

universe = pd.read_csv("universe.csv")

results = []

for symbol in universe["symbol"].dropna():

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

        # Returns
        r3 = close.iloc[-1] / close.iloc[-64] - 1
        r6 = close.iloc[-1] / close.iloc[-127] - 1
        r12 = close.iloc[-1] / close.iloc[-253] - 1

        # Daily returns
        daily = close.pct_change().dropna()

        # Volatility
        volatility = daily.std() * np.sqrt(252)

        # Sharpe-like risk-adjusted return
        risk_adjusted = r12 / volatility if volatility > 0 else 0

        # 200-day trend
        sma200 = close.rolling(200).mean().iloc[-1]
        trend = close.iloc[-1] / sma200 - 1

        # Maximum drawdown
        wealth = (1 + daily).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        max_drawdown = drawdown.min()

        # Preliminary composite score
        score = (
            r3 * 0.15
            + r6 * 0.25
            + r12 * 0.35
            + risk_adjusted * 0.15
            + trend * 0.10
        )

        # Risk penalty
        score = score + max_drawdown * 0.10

        results.append({
            "Stock": symbol,
            "3M Return": r3,
            "6M Return": r6,
            "12M Return": r12,
            "Volatility": volatility,
            "Risk Adjusted": risk_adjusted,
            "Trend": trend,
            "Max Drawdown": max_drawdown,
            "Score": score
        })

    except Exception as e:
        print(f"Skipped {symbol}: {e}")

df = pd.DataFrame(results)

if df.empty:
    print("No valid results.")
else:

    df = df.sort_values(
        "Score",
        ascending=False
    ).reset_index(drop=True)

    df["Rank"] = df.index + 1

    top15 = df.head(15)

    print("\nTOP 15")
    print("=======")

    print(
        top15[
            [
                "Rank",
                "Stock",
                "3M Return",
                "6M Return",
                "12M Return",
                "Volatility",
                "Risk Adjusted",
                "Trend",
                "Max Drawdown",
                "Score"
            ]
        ].to_string(index=False)
    )

    df.to_csv("ranking.csv", index=False)

    print("\nFull ranking saved to ranking.csv")
    print("TOP 15 CALCULATION: WORKING")
