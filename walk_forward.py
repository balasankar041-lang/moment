import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"
TOP_100 = 100
TOP_50 = 50
TRADING_COST = 0.005

WINDOWS = [
    ("2020-03-31", "2022-12-31", "2023-01-31", "2023-12-31"),
    ("2021-01-31", "2023-12-31", "2024-01-31", "2024-12-31"),
    ("2022-01-31", "2024-12-31", "2025-01-31", "2025-12-31"),
    ("2023-01-31", "2025-12-31", "2026-01-31", None),
]

membership = pd.read_csv(MEMBERSHIP_FILE)
membership["effective_date"] = pd.to_datetime(membership["effective_date"])
membership["symbol"] = membership["symbol"].astype(str).str.strip().str.upper()
membership = membership.dropna(subset=["effective_date","symbol"]).sort_values(["effective_date","symbol"])

def yahoo_symbol(s):
    s = str(s).strip().upper()
    return s + ".NS" if s.isalpha() and len(s) <= 20 else None

def members_at_date(date):
    x = membership[membership["effective_date"] <= date]
    if x.empty: return []
    latest = x["effective_date"].max()
    return sorted(set(yahoo_symbol(s) for s in x.loc[x["effective_date"] == latest, "symbol"] if yahoo_symbol(s)))

symbols = sorted(set(yahoo_symbol(s) for s in membership["symbol"] if yahoo_symbol(s)))

download_start = membership["effective_date"].min() - pd.DateOffset(years=2)
today = pd.Timestamp.today().normalize()
download_end = today + pd.Timedelta(days=1)

def download_prices(symbols, batch_size=25):
    frames = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"Downloading {i+1}-{min(i+batch_size,len(symbols))}")
        try:
            data = yf.download(batch, start=download_start.strftime("%Y-%m-%d"), end=download_end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False, threads=True)
            if data.empty: continue
            if isinstance(data.columns, pd.MultiIndex):
                if "Close" not in data.columns.get_level_values(0): continue
                close = data["Close"]
            else:
                if "Close" not in data.columns: continue
                close = data[["Close"]]; close.columns = [batch[0]]
            frames.append(close)
        except Exception:
            continue
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1, sort=True).loc[:, lambda x: ~x.columns.duplicated()].sort_index()

prices = download_prices(symbols).dropna(axis=1, how="all")
if prices.empty: raise SystemExit("No price data downloaded.")

monthly = prices.resample("ME").last()
last_trading_day = prices.index.max()
if last_trading_day.to_period("M") == today.to_period("M"):
    completed_month = (today.to_period("M") - 1).end_time.normalize()
else:
    completed_month = last_trading_day.to_period("M").end_time.normalize()
monthly = monthly[monthly.index <= completed_month]

def signal(symbol, date):
    if symbol not in prices.columns: return None
    s = prices[symbol].dropna()
    start_cut = date - pd.DateOffset(months=12)
    end_cut = date - pd.DateOffset(months=2)
    a = s.loc[s.index <= start_cut]
    b = s.loc[s.index <= end_cut]
    if a.empty or b.empty: return None
    p0, p1 = a.iloc[-1], b.iloc[-1]
    if p0 <= 0 or p1 <= 0: return None
    ret = p1/p0 - 1
    path = s.loc[(s.index >= a.index[-1]) & (s.index <= b.index[-1])]
    if len(path) < 100: return None
    dr = path.pct_change().dropna()
    dr = dr[dr != 0]
    if dr.empty: return None
    pos = (dr > 0).mean()
    neg = (dr < 0).mean()
    id_score = (1 if ret > 0 else -1) * (neg - pos)
    return ret, id_score

def run_test(test_start, test_end):
    test_start = pd.Timestamp(test_start)
    test_end = completed_month if test_end is None else min(pd.Timestamp(test_end), completed_month)
    dates = [d for d in monthly.index if test_start <= d <= test_end]
    if len(dates) < 2: return None

    equity = 1.0
    previous = {}
    values = []
    rebalances = 0
    skipped = 0

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i+1]
        candidates = [s for s in members_at_date(date) if s in prices.columns]
        signals = {}
        for s in candidates:
            z = signal(s, date)
            if z is not None: signals[s] = z
        if len(signals) < TOP_100:
            skipped += 1
            continue

        top100 = sorted(signals.items(), key=lambda x: x[1][0], reverse=True)[:TOP_100]
        selected = [s for s, z in sorted(top100, key=lambda x: x[1][1])[:TOP_50]]
        if len(selected) < TOP_50:
            skipped += 1
            continue

        weight = 1.0 / len(selected)
        new = {s: weight for s in selected}
        all_symbols = set(previous) | set(new)
        turnover = sum(abs(new.get(s, 0) - previous.get(s, 0)) for s in all_symbols)
        equity *= 1 - turnover * TRADING_COST

        period_return = 0.0
        for s, w in new.items():
            try:
                p0 = monthly.loc[date, s]
                p1 = monthly.loc[next_date, s]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    period_return += w * (p1/p0 - 1)
            except Exception:
                pass

        equity *= 1 + period_return
        values.append((next_date, equity))
        previous = new
        rebalances += 1

    if len(values) < 2: return None
    eq = pd.Series(dict(values)).sort_index()
    returns = eq.pct_change().dropna()
    total_return = eq.iloc[-1]/eq.iloc[0] - 1
    years = max((eq.index[-1]-eq.index[0]).days/365.25, 1/12)
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/years) - 1
    volatility = returns.std()*np.sqrt(12)
    sharpe = returns.mean()*12/volatility if volatility > 0 else np.nan
    max_drawdown = (eq/eq.cummax()-1).min()
    winning_months = (returns > 0).mean()

    return {
        "training_start": None,
        "training_end": None,
        "test_start": eq.index[0].date(),
        "test_end": eq.index[-1].date(),
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "winning_months": winning_months,
        "months": len(returns),
        "rebalances": rebalances,
        "skipped": skipped
    }

results = []
for train_start, train_end, test_start, test_end in WINDOWS:
    print("\n--------------------------------")
    print("Training:", train_start, "to", train_end)
    print("Unseen test:", test_start, "to", test_end if test_end else completed_month.date())
    result = run_test(test_start, test_end)
    if result is None:
        print("Insufficient data.")
        continue
    result["training_start"] = train_start
    result["training_end"] = train_end
    results.append(result)
    print("CAGR:", f"{result['cagr']*100:.2f}%")
    print("Sharpe:", f"{result['sharpe']:.2f}")
    print("Max DD:", f"{result['max_drawdown']*100:.2f}%")
    print("Winning months:", f"{result['winning_months']*100:.2f}%")

if not results: raise SystemExit("No walk-forward results.")

summary = pd.DataFrame(results)
summary["total_return"] *= 100
summary["cagr"] *= 100
summary["volatility"] *= 100
summary["max_drawdown"] *= 100
summary["winning_months"] *= 100

print("\n\nWALK-FORWARD SUMMARY")
print("====================")
print(summary[["training_start","training_end","test_start","test_end","cagr","volatility","sharpe","max_drawdown","winning_months","months"]].to_string(index=False))
print("\nAVERAGE UNSEEN CAGR:", f"{summary['cagr'].mean():.2f}%")
print("MEDIAN UNSEEN CAGR:", f"{summary['cagr'].median():.2f}%")
print("AVERAGE SHARPE:", f"{summary['sharpe'].mean():.2f}")
print("AVERAGE MAX DRAWDOWN:", f"{summary['max_drawdown'].mean():.2f}%")
summary.to_csv("walk_forward_results.csv", index=False)
print("\nSaved: walk_forward_results.csv")
print("WALK-FORWARD TEST COMPLETE")
