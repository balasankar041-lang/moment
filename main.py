import pandas as pd
import numpy as np
import yfinance as yf
from io import BytesIO
import requests

print("NIFTY 500 QUANT MOMENTUM SCREENER")
print("=================================")

# Official Nifty 500 constituent file
url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers, timeout=30)
response.raise_for_status()

universe = pd.read_csv(BytesIO(response.content))

if "Symbol" not in universe.columns:
    raise ValueError(
        f"Unexpected Nifty file columns: {list(universe.columns)}"
    )

stocks = (
    universe["Symbol"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.upper()
    .unique()
)

stocks = [symbol + ".NS" for symbol in stocks]

print(f"Nifty 500 universe loaded: {len(stocks)} stocks")

results = []

for number, symbol in enumerate(stocks, start=1):

    print(f"[{number}/{len(stocks)}] {symbol}")

    try:
        data = yf.download(
            symbol,
            period="2y",
            auto_adjust=True,
            progress=False,
            threads=False
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

        # Daily risk
        daily = close.pct_change().dropna()

        volatility = daily.std() * np.sqrt(252)

        if volatility <= 0:
            continue

        # Risk-adjusted momentum
        risk_adjusted = r12 / volatility

        # 200-day trend
        sma200 = close.rolling(200).mean().iloc[-1]

        if pd.isna(sma200):
            continue

        trend = close.iloc[-1] / sma200 - 1

        # Maximum drawdown
        wealth = (1 + daily).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        max_drawdown = drawdown.min()

        # Composite score
        score = (
            r3 * 0.15
            + r6 * 0.25
            + r12 * 0.35
            + risk_adjusted * 0.15
            + trend * 0.10
            + max_drawdown * 0.10
        )

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
    raise RuntimeError("No stocks produced valid results.")

df = df.sort_values(
    "Score",
    ascending=False
).reset_index(drop=True)

df["Rank"] = df.index + 1

top15 = df.head(15)

print("\n============================")
print("TOP 15 QUANT MOMENTUM STOCKS")
print("============================")

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
print(f"Valid stocks scored: {len(df)}")
print("STATUS: SUCCESS")
