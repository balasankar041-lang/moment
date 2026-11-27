import re
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from price_cache import load_cache
except Exception:
    load_cache = None

warnings.filterwarnings("ignore")

TOP_MOMENTUM = 100
TOP_ID = 50


def clean_symbol(s):
    s = str(s).strip().upper()
    if re.fullmatch(r"[A-Z0-9&\-]{2,20}", s) and not s.endswith("&") and not s.startswith("&"):
        return s
    return None


def load_prices(symbols, start_date):
    if load_cache is not None:
        cached_close, _ = load_cache(with_ns_suffix=False)
        if cached_close is not None and not cached_close.empty:
            cols = [s for s in symbols if s in cached_close.columns]
            if len(cols) / max(1, len(symbols)) >= 0.80:
                print(f"Loaded prices from data_cache: {len(cols)}/{len(symbols)} symbols")
                return cached_close[cols].sort_index()

    today = pd.Timestamp.today().normalize()
    end_date = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tickers = [s + ".NS" for s in symbols]
    frames = []
    for i in range(0, len(tickers), 20):
        batch = tickers[i:i + 20]
        try:
            data = yf.download(batch, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=False, group_by="column")
            if data is not None and not data.empty:
                close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].rename(columns={"Close": batch[0]})
                if isinstance(close, pd.Series):
                    close = close.to_frame(name=batch[0])
                close.columns = [str(c).replace(".NS", "") for c in close.columns]
                frames.append(close)
        except Exception:
            pass
        time.sleep(1)

    if not frames:
        raise RuntimeError("No usable price data downloaded.")
    prices = pd.concat(frames, axis=1)
    return prices.loc[:, ~prices.columns.duplicated()].sort_index()


def plan2_signal(prices, symbol, date):
    if symbol not in prices.columns:
        return None
    s = prices[symbol].dropna()
    a = s.loc[s.index <= date - pd.DateOffset(months=12)]
    b = s.loc[s.index <= date - pd.DateOffset(months=2)]
    if a.empty or b.empty:
        return None
    p0, p1 = float(a.iloc[-1]), float(b.iloc[-1])
    if p0 <= 0 or p1 <= 0:
        return None
    momentum = p1 / p0 - 1.0
    path = s.loc[(s.index >= a.index[-1]) & (s.index <= b.index[-1])]
    if len(path) < 100:
        return None
    dr = path.pct_change().dropna()
    dr = dr[dr != 0]
    if dr.empty:
        return None
    id_score = (1.0 if momentum > 0 else -1.0) * ((dr < 0).mean() - (dr > 0).mean())
    return momentum, float(id_score)


def main():
    membership = pd.read_csv("nifty500_membership_timeline.csv")
    membership["date"] = pd.to_datetime(membership["effective_date"])
    membership["symbol"] = membership["symbol"].map(clean_symbol)
    membership = membership.dropna(subset=["date", "symbol"])

    symbols = sorted(membership["symbol"].unique())
    start_date = (membership["date"].min() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    prices = load_prices(symbols, start_date).dropna(axis=1, how="all")

    today = pd.Timestamp.today().normalize()
    last_trading_day = prices.index.max()
    if last_trading_day.to_period("M") == today.to_period("M"):
        completed_month = (today.to_period("M") - 1).end_time.normalize()
    else:
        completed_month = last_trading_day.to_period("M").end_time.normalize()

    monthly_prices = prices.resample("ME").last()
    monthly_prices = monthly_prices[monthly_prices.index <= completed_month]

    rebalance_dates = [
        d for d in monthly_prices.index
        if d >= membership["date"].min() + pd.DateOffset(months=13)
    ]

    portfolio_records = []
    monthly_results = []
    correlation_results = []

    for i, date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]
        available = membership[membership["date"] <= date]
        if available.empty:
            continue
        latest_snapshot = available["date"].max()
        universe = [
            s for s in available.loc[available["date"] == latest_snapshot, "symbol"].unique()
            if s in prices.columns
        ]

        signals = {}
        for sym in universe:
            z = plan2_signal(prices, sym, date)
            if z is not None:
                signals[sym] = z

        if len(signals) < TOP_MOMENTUM:
            continue

        top100 = sorted(signals.items(), key=lambda x: (x[1][0], x[0]), reverse=True)[:TOP_MOMENTUM]
        selected = [s for s, _ in sorted(top100, key=lambda x: (x[1][1], x[0]))[:TOP_ID]]

        for sym in selected:
            portfolio_records.append({"date": date, "return_date": next_date, "symbol": sym})

        # Forward return (signal month-end -> next month-end)
        p0 = monthly_prices.loc[date, selected]
        p1 = monthly_prices.loc[next_date, selected]
        returns = (p1 / p0 - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) < 10:
            continue

        weights = pd.Series(1.0 / len(returns), index=returns.index)
        contribution = weights * returns
        contribution_sorted = contribution.sort_values(ascending=False)
        total_return = float(contribution.sum())

        top5_contrib = float(contribution_sorted.head(5).sum() / total_return) if total_return != 0 else np.nan
        top10_contrib = float(contribution_sorted.head(10).sum() / total_return) if total_return != 0 else np.nan

        abs_c = contribution.abs()
        abs_total = float(abs_c.sum())
        if abs_total > 0:
            top5_abs = float(abs_c.sort_values(ascending=False).head(5).sum() / abs_total)
            top10_abs = float(abs_c.sort_values(ascending=False).head(10).sum() / abs_total)
            cw = abs_c / abs_total
            effective_bets = float(1.0 / (cw ** 2).sum())
        else:
            top5_abs = top10_abs = effective_bets = np.nan

        monthly_results.append({
            "signal_date": date,
            "date": next_date,
            "holdings": len(returns),
            "portfolio_return": total_return,
            "top5_return_contribution": top5_contrib,
            "top10_return_contribution": top10_contrib,
            "top5_absolute_contribution": top5_abs,
            "top10_absolute_contribution": top10_abs,
            "effective_bets": effective_bets,
        })

        daily_window = prices.loc[date - pd.DateOffset(months=3):date, selected].pct_change()
        corr = daily_window.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        correlation_results.append({
            "signal_date": date,
            "date": next_date,
            "average_pairwise_correlation": upper.stack().mean(),
        })

    portfolio = pd.DataFrame(portfolio_records)
    results = pd.DataFrame(monthly_results)
    correlations = pd.DataFrame(correlation_results)

    if results.empty:
        raise RuntimeError("No portfolio concentration results generated.")

    monthly_sets = portfolio.groupby("date")["symbol"].apply(set).sort_index()
    turnovers = []
    previous = None
    for d, current in monthly_sets.items():
        if previous is None:
            previous = current
            continue
        added = current - previous
        removed = previous - current
        turnovers.append({"date": d, "turnover": len(removed) / TOP_ID, "added": len(added), "removed": len(removed)})
        previous = current
    turnover_df = pd.DataFrame(turnovers)

    print("\n===================================")
    print("PORTFOLIO CONCENTRATION TEST (FORWARD RETURNS, 12-2 PLAN 2)")
    print("===================================")
    print("Months tested:", len(results))
    print("Median Top-5 absolute contribution:", round(results["top5_absolute_contribution"].median() * 100, 2), "%")
    print("Median Top-10 absolute contribution:", round(results["top10_absolute_contribution"].median() * 100, 2), "%")
    print("Median effective number of bets:", round(results["effective_bets"].median(), 2))
    print("Median average stock correlation:", round(correlations["average_pairwise_correlation"].median(), 3))
    if not turnover_df.empty:
        print("Average monthly turnover:", round(turnover_df["turnover"].mean() * 100, 2), "%")

    results.to_csv("portfolio_concentration_monthly.csv", index=False)
    correlations.to_csv("portfolio_correlation_monthly.csv", index=False)
    if not turnover_df.empty:
        turnover_df.to_csv("portfolio_turnover_monthly.csv", index=False)
    print("Detailed result files created.")


if __name__ == "__main__":
    main()
