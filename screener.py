import numpy as np
import pandas as pd

def annualized_volatility(returns, periods=252):
    return returns.std() * np.sqrt(periods)

def max_drawdown(series):
    wealth = (1 + series.pct_change().fillna(0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return drawdown.min()

def score_universe(prices, universe=None):
    if universe:
        cols = [c for c in universe if c in prices.columns]
        prices = prices[cols]

    rows = []
    for symbol in prices.columns:
        s = prices[symbol].dropna()
        if len(s) < 252:
            continue

        r63 = s.iloc[-1] / s.iloc[-64] - 1
        r126 = s.iloc[-1] / s.iloc[-127] - 1
        r252 = s.iloc[-1] / s.iloc[-253] - 1

        daily = s.pct_change().dropna()
        vol = annualized_volatility(daily)
        sma200 = s.rolling(200).mean().iloc[-1]
        trend = s.iloc[-1] / sma200 - 1 if sma200 else np.nan

        # Research score: multi-period momentum with trend and volatility penalty.
        raw = 0.20*r63 + 0.30*r126 + 0.50*r252
        score = raw + 0.20*trend - 0.10*vol

        rows.append({
            "symbol": symbol,
            "return_3m": r63,
            "return_6m": r126,
            "return_12m": r252,
            "volatility": vol,
            "trend_vs_200dma": trend,
            "score": score,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank"] = out["score"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("rank")
