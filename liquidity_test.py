"""
Plan 2 Portfolio Liquidity Test

Evaluates the actual historical Plan 2 Top-50 holdings' 100-day Average Daily Traded Value (ADV)
and flags how many orders exceed 2% of ADV across realistic portfolio capital levels.
"""

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
ADV_WINDOW = 100
ADV_LIMIT = 0.02
CAPITALS = [20_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]


def clean_symbol(s):
    s = str(s).strip().upper()
    if re.fullmatch(r"[A-Z0-9&\-]{2,20}", s) and not s.endswith("&") and not s.startswith("&"):
        return s
    return None


def load_close_volume(symbols, start_date):
    if load_cache is not None:
        close, volume = load_cache(with_ns_suffix=False)
        if close is not None and volume is not None and not close.empty:
            cols = [s for s in symbols if s in close.columns and s in volume.columns]
            if len(cols) / max(1, len(symbols)) >= 0.80:
                print(f"Loaded close & volume from data_cache: {len(cols)}/{len(symbols)} symbols")
                return close[cols].sort_index(), volume[cols].sort_index()

    today = pd.Timestamp.today().normalize()
    end_date = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tickers = [s + ".NS" for s in symbols]
    close_parts, vol_parts = [], []
    for i in range(0, len(tickers), 15):
        batch = tickers[i:i + 15]
        for attempt in range(4):
            try:
                data = yf.download(batch, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=False, group_by="column")
                if data is not None and not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        c, v = data["Close"], data["Volume"]
                    else:
                        c = data[["Close"]].rename(columns={"Close": batch[0]})
                        v = data[["Volume"]].rename(columns={"Volume": batch[0]})
                    if isinstance(c, pd.Series):
                        c = c.to_frame(name=batch[0])
                    if isinstance(v, pd.Series):
                        v = v.to_frame(name=batch[0])
                    c.columns = [str(x).replace(".NS", "") for x in c.columns]
                    v.columns = [str(x).replace(".NS", "") for x in v.columns]
                    close_parts.append(c)
                    vol_parts.append(v)
                    break
            except Exception:
                time.sleep(min(30, 3 * (2 ** attempt)))
        time.sleep(1.5)

    if not close_parts:
        raise RuntimeError("No Yahoo price/volume data available.")
    close = pd.concat(close_parts, axis=1)
    volume = pd.concat(vol_parts, axis=1)
    close = close.loc[:, ~close.columns.duplicated()].sort_index()
    volume = volume.loc[:, ~volume.columns.duplicated()].sort_index()
    common = sorted(set(close.columns) & set(volume.columns))
    return close[common], volume[common]


def plan2_signal(close, symbol, date):
    if symbol not in close.columns:
        return None
    s = close[symbol].dropna()
    a = s.loc[s.index <= date - pd.DateOffset(months=12)]
    b = s.loc[s.index <= date - pd.DateOffset(months=2)]
    if a.empty or b.empty:
        return None
    p0, p1 = float(a.iloc[-1]), float(b.iloc[-1])
    if p0 <= 0 or p1 <= 0:
        return None
    mom = p1 / p0 - 1.0
    path = s.loc[(s.index >= a.index[-1]) & (s.index <= b.index[-1])]
    if len(path) < 100:
        return None
    dr = path.pct_change().dropna()
    dr = dr[dr != 0]
    if dr.empty:
        return None
    id_score = (1.0 if mom > 0 else -1.0) * ((dr < 0).mean() - (dr > 0).mean())
    return mom, float(id_score)


def main():
    membership = pd.read_csv("nifty500_membership_timeline.csv")
    membership["date"] = pd.to_datetime(membership["effective_date"])
    membership["symbol"] = membership["symbol"].map(clean_symbol)
    membership = membership.dropna(subset=["date", "symbol"])

    symbols = sorted(membership["symbol"].unique())
    start_date = (membership["date"].min() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    close, volume = load_close_volume(symbols, start_date)

    traded_value = (close * volume).replace([np.inf, -np.inf], np.nan)
    adv100 = traded_value.rolling(ADV_WINDOW, min_periods=50).mean()

    today = pd.Timestamp.today().normalize()
    last_trading_day = close.index.max()
    if last_trading_day.to_period("M") == today.to_period("M"):
        completed_month = (today.to_period("M") - 1).end_time.normalize()
    else:
        completed_month = last_trading_day.to_period("M").end_time.normalize()

    monthly_close = close.resample("ME").last()
    monthly_close = monthly_close[monthly_close.index <= completed_month]
    rebalance_dates = [
        d for d in monthly_close.index
        if d >= membership["date"].min() + pd.DateOffset(months=13)
    ]

    monthly_rows = []
    for date in rebalance_dates:
        available = membership[membership["date"] <= date]
        if available.empty:
            continue
        latest_snap = available["date"].max()
        universe = [
            s for s in available.loc[available["date"] == latest_snap, "symbol"].unique()
            if s in close.columns
        ]

        signals = {}
        for sym in universe:
            z = plan2_signal(close, sym, date)
            if z is not None:
                signals[sym] = z
        if len(signals) < TOP_MOMENTUM:
            continue

        top100 = sorted(signals.items(), key=lambda x: (x[1][0], x[0]), reverse=True)[:TOP_MOMENTUM]
        selected = [s for s, _ in sorted(top100, key=lambda x: (x[1][1], x[0]))[:TOP_ID]]

        adv_slice = adv100.loc[adv100.index <= date, selected]
        if adv_slice.empty:
            continue
        latest_adv = adv_slice.iloc[-1].dropna()
        if latest_adv.empty:
            continue

        row = {
            "date": date,
            "holdings_with_adv": len(latest_adv),
            "min_adv_cr": round(float(latest_adv.min()) / 1e7, 4),
            "median_adv_cr": round(float(latest_adv.median()) / 1e7, 4),
            "below_25L_count": int((latest_adv < 25_00_000).sum()),
            "below_1Cr_count": int((latest_adv < 1_00_00_000).sum()),
        }
        for cap in CAPITALS:
            order_size = cap / TOP_ID
            row[f"flagged_pct_{cap}"] = round(float(((order_size / latest_adv) > ADV_LIMIT).mean() * 100.0), 2)
        monthly_rows.append(row)

    monthly_df = pd.DataFrame(monthly_rows)
    if monthly_df.empty:
        raise RuntimeError("No liquidity test records generated.")

    summary_rows = []
    for cap in CAPITALS:
        summary_rows.append({
            "capital_inr": cap,
            "order_per_stock_inr": cap / TOP_ID,
            "avg_flagged_stocks_pct": round(float(monthly_df[f"flagged_pct_{cap}"].mean()), 2),
            "max_flagged_stocks_pct": round(float(monthly_df[f"flagged_pct_{cap}"].max()), 2),
        })
    summary_df = pd.DataFrame(summary_rows)

    print("\n==========================================")
    print("PLAN 2 PORTFOLIO LIQUIDITY TEST (2% ADV RULE)")
    print("==========================================")
    print("Months evaluated:", len(monthly_df))
    print("Median portfolio holding ADV (₹ Cr):", round(monthly_df["median_adv_cr"].median(), 2))
    print("Worst single holding ADV (₹ Cr):", round(monthly_df["min_adv_cr"].min(), 4))
    print("\nCapacity Summary:")
    print(summary_df.to_string(index=False))

    monthly_df.to_csv("liquidity_test_monthly.csv", index=False)
    summary_df.to_csv("liquidity_test_summary.csv", index=False)
    print("\nSaved: liquidity_test_monthly.csv, liquidity_test_summary.csv")


if __name__ == "__main__":
    main()
