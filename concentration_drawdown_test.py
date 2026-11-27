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
    batch_size = 15

    for start_idx in range(0, len(tickers), batch_size):
        batch = tickers[start_idx:start_idx + batch_size]
        for attempt in range(4):
            try:
                data = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    group_by="column",
                )
                if data is not None and not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        close = data["Close"]
                    else:
                        close = data[["Close"]].copy()
                        close.columns = [batch[0]]
                    if isinstance(close, pd.Series):
                        close = close.to_frame(name=batch[0])
                    close.columns = [str(c).replace(".NS", "") for c in close.columns]
                    frames.append(close)
                    break
            except Exception as exc:
                print(f"Batch {start_idx + 1} attempt {attempt + 1} failed: {exc}")
                time.sleep(min(30, 3 * (2 ** attempt)))
        time.sleep(1.5)

    if not frames:
        raise RuntimeError("No usable price data downloaded.")

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    return prices


def plan2_signal(prices, symbol, date):
    if symbol not in prices.columns:
        return None
    s = prices[symbol].dropna()
    start_cut = date - pd.DateOffset(months=12)
    end_cut = date - pd.DateOffset(months=2)
    a = s.loc[s.index <= start_cut]
    b = s.loc[s.index <= end_cut]
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
    pos = (dr > 0).mean()
    neg = (dr < 0).mean()
    id_score = (1.0 if momentum > 0 else -1.0) * (neg - pos)
    return momentum, float(id_score)


def main():
    membership = pd.read_csv("nifty500_membership_timeline.csv")
    membership["date"] = pd.to_datetime(membership["effective_date"])
    membership["symbol"] = membership["symbol"].map(clean_symbol)
    membership = membership.dropna(subset=["date", "symbol"])

    symbols = sorted(membership["symbol"].unique())
    print("Historical symbols:", len(symbols))

    start_date = (membership["date"].min() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    prices = load_prices(symbols, start_date).dropna(axis=1, how="all")

    today = pd.Timestamp.today().normalize()
    last_trading_day = prices.index.max()
    if last_trading_day.to_period("M") == today.to_period("M"):
        completed_month_end = (today.to_period("M") - 1).end_time.normalize()
    else:
        completed_month_end = last_trading_day.to_period("M").end_time.normalize()

    monthly_prices = prices.resample("ME").last()
    monthly_prices = monthly_prices[monthly_prices.index <= completed_month_end]

    rebalance_dates = [
        d for d in monthly_prices.index
        if d >= membership["date"].min() + pd.DateOffset(months=13)
    ]

    # Build exact Plan 2 (12-2 momentum + lowest ID) portfolios and evaluate FORWARD returns
    portfolio_records = []
    monthly_returns = []
    corr_values = []

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
            portfolio_records.append({"signal_date": date, "return_date": next_date, "symbol": sym})

        # Forward return from signal month-end (date) to next month-end (next_date) — NO look-ahead bias
        p0 = monthly_prices.loc[date, selected]
        p1 = monthly_prices.loc[next_date, selected]
        returns = (p1 / p0 - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) < 10:
            continue

        portfolio_return = float(returns.mean())
        contributions = returns / len(returns)
        abs_contrib = contributions.abs()
        total_abs = float(abs_contrib.sum())

        if total_abs > 0:
            top5_abs = float(abs_contrib.sort_values(ascending=False).head(5).sum() / total_abs)
            top10_abs = float(abs_contrib.sort_values(ascending=False).head(10).sum() / total_abs)
            cw = abs_contrib / total_abs
            effective_bets = float(1.0 / (cw ** 2).sum())
        else:
            top5_abs = top10_abs = effective_bets = np.nan

        monthly_returns.append({
            "signal_date": date,
            "date": next_date,
            "portfolio_return": portfolio_return,
            "top5_absolute_contribution": top5_abs,
            "top10_absolute_contribution": top10_abs,
            "effective_bets": effective_bets,
            "holdings": len(returns),
        })

        # Trailing 3-month correlation known at signal_date
        window_start = date - pd.DateOffset(months=3)
        daily_ret = prices.loc[window_start:date, selected].pct_change()
        corr = daily_ret.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        avg_corr = upper.stack().mean()
        corr_values.append({"signal_date": date, "date": next_date, "average_correlation": avg_corr})

    results = pd.DataFrame(monthly_returns)
    correlation_df = pd.DataFrame(corr_values)
    if results.empty:
        raise RuntimeError("No monthly portfolio results generated.")

    results["equity"] = (1.0 + results["portfolio_return"]).cumprod()
    results["peak"] = results["equity"].cummax()
    results["drawdown"] = results["equity"] / results["peak"] - 1.0

    worst = results.sort_values("portfolio_return").head(15).copy()

    print("\n===================================")
    print("CONCENTRATION vs DRAWDOWN TEST (FORWARD RETURNS, 12-2 PLAN 2)")
    print("===================================")
    print("Months tested:", len(results))
    print("Worst monthly return:", round(worst.iloc[0]["portfolio_return"] * 100, 2), "%")
    print("Maximum portfolio drawdown:", round(results["drawdown"].min() * 100, 2), "%")

    display_columns = [
        "date", "portfolio_return", "drawdown",
        "top5_absolute_contribution", "top10_absolute_contribution",
        "effective_bets", "holdings",
    ]
    print("\nWorst 15 portfolio months:")
    print(worst[display_columns].to_string(index=False))

    bottom_10 = results.nsmallest(max(1, int(len(results) * 0.10)), "portfolio_return")
    normal = results[~results.index.isin(bottom_10.index)]
    print("\nWorst 10% avg return:", round(bottom_10["portfolio_return"].mean() * 100, 2), "%")
    print("Worst 10% median effective bets:", round(bottom_10["effective_bets"].median(), 2))
    print("Normal median effective bets:", round(normal["effective_bets"].median(), 2))

    if not correlation_df.empty:
        print("Median correlation:", round(correlation_df["average_correlation"].median(), 3))
        print("Maximum correlation:", round(correlation_df["average_correlation"].max(), 3))

    results.to_csv("concentration_drawdown_monthly.csv", index=False)
    correlation_df.to_csv("concentration_drawdown_correlation.csv", index=False)
    print("\nSaved: concentration_drawdown_monthly.csv, concentration_drawdown_correlation.csv")
    print("Plan2 parameters were NOT changed.")


if __name__ == "__main__":
    main()
