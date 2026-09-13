
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# SETTINGS
# ============================================================
MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"
TOP_N = 15

W3, W6, W12 = 0.20, 0.30, 0.50
TRADING_COST = 0.005          # 0.50% of turnover
MIN_HISTORY_MONTHS = 36
MIN_VOL_DAYS = 200
BATCH_SIZE = 25
RETRIES = 3

# Keep Yahoo's noisy warnings out of the Actions log.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# ============================================================
# LOAD MEMBERSHIP
# ============================================================
membership = pd.read_csv(MEMBERSHIP_FILE)
membership["effective_date"] = pd.to_datetime(membership["effective_date"], errors="coerce")
membership["symbol"] = membership["symbol"].astype(str).str.strip().str.upper()
membership = membership.dropna(subset=["effective_date"])
membership = membership[membership["symbol"].str.match(r"^[A-Z0-9]+$")]
membership = membership.drop_duplicates(["effective_date", "symbol"])
membership = membership.sort_values(["effective_date", "symbol"])

print("\nHISTORICAL MEMBERSHIP")
print("=====================")
print("Records:", len(membership))
print("Snapshots:", membership["effective_date"].nunique())
print("Date range:", membership["effective_date"].min().date(), "to", membership["effective_date"].max().date())

def yahoo_symbol(symbol):
    # Historical symbols are used as recorded. We deliberately do NOT
    # guess merger/name changes because that can create false history.
    return f"{symbol}.NS"

def members_at_date(date):
    available = membership[membership["effective_date"] <= date]
    if available.empty:
        return []
    latest = available["effective_date"].max()
    return sorted(set(yahoo_symbol(s) for s in available.loc[
        available["effective_date"] == latest, "symbol"
    ]))

# ============================================================
# DOWNLOAD PRICES
# ============================================================
download_start = membership["effective_date"].min() - pd.DateOffset(years=2)
download_end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)

symbols = sorted(set(yahoo_symbol(s) for s in membership["symbol"]))

print("\nDownloading price data...")
print("Price period:", download_start.date(), "to", download_end.date())
print("Historical symbols:", len(symbols))

def download_batch(batch):
    for attempt in range(1, RETRIES + 1):
        try:
            data = yf.download(
                batch,
                start=download_start.strftime("%Y-%m-%d"),
                end=download_end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
            )
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    if "Close" not in data.columns.get_level_values(0):
                        return pd.DataFrame()
                    close = data["Close"].copy()
                else:
                    if "Close" not in data.columns:
                        return pd.DataFrame()
                    close = data[["Close"]].copy()
                    close.columns = [batch[0]]
                return close
        except Exception:
            pass
    return pd.DataFrame()

frames = []
for i in range(0, len(symbols), BATCH_SIZE):
    batch = symbols[i:i+BATCH_SIZE]
    print(f"Downloading {i+1}-{min(i+BATCH_SIZE, len(symbols))}")
    frame = download_batch(batch)
    if not frame.empty:
        frames.append(frame)

if not frames:
    raise RuntimeError("No Yahoo Finance price data was downloaded.")

prices = pd.concat(frames, axis=1, sort=True)
prices = prices.loc[:, ~prices.columns.duplicated()]
prices = prices.sort_index()

# IMPORTANT: remove columns that contain no actual price history.
prices = prices.dropna(axis=1, how="all")

print("\nPRICE DATA")
print("==========")
print("Stocks with actual data:", prices.shape[1])
print("Trading days:", len(prices))
print("Last available trading day:", prices.index.max().date())

# ============================================================
# USE ONLY COMPLETED MONTHS
# ============================================================
last_trading_day = prices.index.max().normalize()

# Do not label an incomplete current month as a future month-end.
last_complete_month = (
    last_trading_day.to_period("M") - 1
).to_timestamp("M")

monthly_prices = prices.resample("ME").last()
monthly_prices = monthly_prices.loc[monthly_prices.index <= last_complete_month]
monthly_prices = monthly_prices.dropna(axis=0, how="all")

print("Last completed month:", monthly_prices.index.max().date())

# ============================================================
# MOMENTUM
# ============================================================
def momentum_score(symbol, date):
    if symbol not in prices.columns or date not in monthly_prices.index:
        return None

    daily = prices[symbol].dropna()
    if daily.empty:
        return None

    past_12m = daily.loc[date - pd.DateOffset(months=12):date]
    if len(past_12m) < MIN_VOL_DAYS:
        return None

    current = monthly_prices.loc[date, symbol]
    if pd.isna(current) or current <= 0:
        return None

    def prior_price(months):
        target = date - pd.DateOffset(months=months)
        x = monthly_prices.loc[monthly_prices.index <= target, symbol].dropna()
        return None if x.empty else x.iloc[-1]

    p3, p6, p12 = prior_price(3), prior_price(6), prior_price(12)
    if p3 is None or p6 is None or p12 is None:
        return None
    if min(p3, p6, p12) <= 0:
        return None

    r3, r6, r12 = current/p3 - 1, current/p6 - 1, current/p12 - 1

    daily_returns = past_12m.pct_change().dropna()
    if len(daily_returns) < MIN_VOL_DAYS - 1:
        return None

    vol = daily_returns.std() * np.sqrt(252)
    if pd.isna(vol) or vol <= 0:
        return None

    momentum = W3*r3 + W6*r6 + W12*r12
    if momentum <= 0:
        return None

    return momentum / vol

# ============================================================
# BACKTEST
# ============================================================
rebalance_dates = list(monthly_prices.index)
if not rebalance_dates:
    raise RuntimeError("No completed monthly data available.")

warmup_cutoff = rebalance_dates[0] + pd.DateOffset(months=MIN_HISTORY_MONTHS)
rebalance_dates = [d for d in rebalance_dates if d >= warmup_cutoff]

portfolio_value = 1.0
equity = []
previous_weights = {}
rebalance_count = 0
skipped = 0

for i, date in enumerate(rebalance_dates[:-1]):
    historical_members = members_at_date(date)
    candidates = [s for s in historical_members if s in prices.columns]

    scores = {}
    for symbol in candidates:
        score = momentum_score(symbol, date)
        if score is not None and np.isfinite(score):
            scores[symbol] = score

    if len(scores) < TOP_N:
        skipped += 1
        continue

    selected = [
        s for s, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
    ]
    weight = 1.0 / len(selected)
    new_weights = {s: weight for s in selected}

    # Sum of absolute weight changes = turnover.
    turnover = sum(
        abs(new_weights.get(s, 0.0) - previous_weights.get(s, 0.0))
        for s in set(previous_weights) | set(new_weights)
    )
    portfolio_value *= (1.0 - turnover * TRADING_COST)

    next_date = rebalance_dates[i + 1]
    period_return = 0.0

    for symbol, w in new_weights.items():
        if symbol not in monthly_prices.columns:
            continue
        p0 = monthly_prices.loc[date, symbol]
        p1 = monthly_prices.loc[next_date, symbol]
        if pd.notna(p0) and pd.notna(p1) and p0 > 0:
            period_return += w * (p1/p0 - 1.0)

    portfolio_value *= (1.0 + period_return)
    equity.append((next_date, portfolio_value))
    previous_weights = new_weights
    rebalance_count += 1

if not equity:
    raise RuntimeError("No backtest results were generated.")

equity = pd.Series(dict(equity)).sort_index()
monthly_returns = equity.pct_change().dropna()

# Fixed split chosen before looking at results: training through 2023, unseen test from 2024.
TRAIN_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")

train = monthly_returns.loc[monthly_returns.index <= TRAIN_END]
test = monthly_returns.loc[monthly_returns.index >= TEST_START]

def metrics(r):
    if r.empty:
        return None
    curve = (1.0 + r).cumprod()
    total = curve.iloc[-1] - 1.0
    years = max(len(r) / 12.0, 1/12)
    cagr = curve.iloc[-1] ** (1.0 / years) - 1.0
    vol = r.std() * np.sqrt(12)
    sharpe = (r.mean() * 12 / vol) if vol > 0 else np.nan
    dd = curve / curve.cummax() - 1.0
    return {
        "start": r.index[0], "end": r.index[-1], "total": total,
        "cagr": cagr, "vol": vol, "sharpe": sharpe, "dd": dd.min(),
        "win": (r > 0).mean(), "months": len(r)
    }

train_m = metrics(train)
test_m = metrics(test)
all_m = metrics(monthly_returns)

print("\n")
print("NIFTY 500 HISTORICAL-MEMBERSHIP")
print("RISK-ADJUSTED TOP-15 OUT-OF-SAMPLE TEST")
print("==========================================")
print("Top stocks:", TOP_N)
print("Momentum: 20% 3M + 30% 6M + 50% 12M")
print("Risk adjustment: Momentum / annualized volatility")
print(f"Trading cost: {TRADING_COST*100:.2f}%")
print("Rebalances:", rebalance_count)
print("Skipped months:", skipped)

for label, m in [("TRAINING (2020-2023)", train_m), ("UNSEEN TEST (2024-2026)", test_m), ("FULL PERIOD", all_m)]:
    print("\n" + label)
    print("-" * len(label))
    if m is None:
        print("No data")
        continue
    print("Period:", m["start"].date(), "to", m["end"].date())
    print(f"Total return: {m['total']*100:.2f}%")
    print(f"CAGR: {m['cagr']*100:.2f}%")
    print(f"Annual volatility: {m['vol']*100:.2f}%")
    print(f"Sharpe ratio: {m['sharpe']:.2f}")
    print(f"Maximum drawdown: {m['dd']*100:.2f}%")
    print(f"Winning months: {m['win']*100:.2f}%")
    print(f"Months tested: {m['months']}")

print("\nBACKTEST COMPLETE")
