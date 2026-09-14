import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# LIVE MOMENTUM SCREENER
# Same project as the backtests.
#
# Strategy:
#   12-2 momentum
#   -> top 100 momentum stocks
#   -> lowest Information Discretion (ID)
#   -> top 50
#
# IMPORTANT:
# This is a research/screening signal, NOT an automatic trading system.
# ============================================================

MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"
TOP_100 = 100
TOP_50 = 50

def yahoo_symbol(s):
    s = str(s).strip().upper()
    return s + ".NS" if s.isalpha() and len(s) <= 20 else None

# ------------------------------------------------------------
# Load latest available Nifty 500 membership
# ------------------------------------------------------------
membership = pd.read_csv(MEMBERSHIP_FILE)
membership["effective_date"] = pd.to_datetime(membership["effective_date"])
membership["symbol"] = membership["symbol"].astype(str).str.strip().str.upper()

latest_date = membership["effective_date"].max()

symbols = sorted(set(
    yahoo_symbol(s)
    for s in membership.loc[
        membership["effective_date"] == latest_date, "symbol"
    ]
    if yahoo_symbol(s)
))

print("\nLIVE NIFTY 500 MOMENTUM SCREENER")
print("================================")
print("Membership snapshot:", latest_date.date())
print("Symbols:", len(symbols))

# ------------------------------------------------------------
# Download enough daily history for 12-2 momentum + ID
# ------------------------------------------------------------
end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
start_date = pd.Timestamp.today().normalize() - pd.DateOffset(months=14)

def download_prices(symbols, batch_size=25):
    frames = []

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"Downloading {i+1}-{min(i+batch_size, len(symbols))}")

        try:
            data = yf.download(
                batch,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True
            )

            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                if "Close" not in data.columns.get_level_values(0):
                    continue
                close = data["Close"]
            else:
                if "Close" not in data.columns:
                    continue
                close = data[["Close"]]
                close.columns = [batch[0]]

            frames.append(close)

        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1, sort=True).loc[
        :, lambda x: ~x.columns.duplicated()
    ].sort_index()

prices = download_prices(symbols)

if prices.empty:
    raise SystemExit("No price data available.")

prices = prices.dropna(axis=1, how="all")

# ------------------------------------------------------------
# Calculate 12-2 momentum and ID
# ------------------------------------------------------------
today = pd.Timestamp.today().normalize()

signal_rows = []

for symbol in prices.columns:

    s = prices[symbol].dropna()

    start_cut = today - pd.DateOffset(months=12)
    end_cut = today - pd.DateOffset(months=2)

    start_data = s.loc[s.index <= start_cut]
    end_data = s.loc[s.index <= end_cut]

    if start_data.empty or end_data.empty:
        continue

    p0 = start_data.iloc[-1]
    p1 = end_data.iloc[-1]

    if p0 <= 0 or p1 <= 0:
        continue

    momentum = p1 / p0 - 1

    path = s.loc[
        (s.index >= start_data.index[-1]) &
        (s.index <= end_data.index[-1])
    ]

    if len(path) < 100:
        continue

    daily_returns = path.pct_change().dropna()
    nonzero = daily_returns[daily_returns != 0]

    if nonzero.empty:
        continue

    positive_pct = (nonzero > 0).mean()
    negative_pct = (nonzero < 0).mean()

    sign = 1 if momentum > 0 else -1

    id_score = sign * (
        negative_pct - positive_pct
    )

    last_price = s.iloc[-1]

    signal_rows.append({
        "symbol": symbol.replace(".NS", ""),
        "yahoo_symbol": symbol,
        "live_price": round(float(last_price), 2),
        "momentum_12_2": momentum,
        "ID": id_score,
        "positive_days_pct": positive_pct,
        "negative_days_pct": negative_pct
    })

signals = pd.DataFrame(signal_rows)

if signals.empty:
    raise SystemExit("No valid signals calculated.")

# ------------------------------------------------------------
# FIRST SORT: Top 100 by 12-2 momentum
# ------------------------------------------------------------
top100 = signals.sort_values(
    "momentum_12_2",
    ascending=False
).head(TOP_100).copy()

# ------------------------------------------------------------
# SECOND SORT: Lowest ID
# ------------------------------------------------------------
top50 = top100.sort_values(
    "ID",
    ascending=True
).head(TOP_50).copy()

# ------------------------------------------------------------
# Signal classification
# ------------------------------------------------------------
top50["rank"] = range(1, len(top50) + 1)

top50["signal"] = np.where(
    top50["momentum_12_2"] > 0,
    "BUY_CANDIDATE",
    "WAIT"
)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
display = top50[
    [
        "rank",
        "symbol",
        "live_price",
        "momentum_12_2",
        "ID",
        "positive_days_pct",
        "negative_days_pct",
        "signal"
    ]
].copy()

display["momentum_12_2"] = (
    display["momentum_12_2"] * 100
).round(2)

display["ID"] = display["ID"].round(4)

display["positive_days_pct"] = (
    display["positive_days_pct"] * 100
).round(1)

display["negative_days_pct"] = (
    display["negative_days_pct"] * 100
).round(1)

print("\nTOP 50 CURRENT SIGNALS")
print("======================")
print(display.to_string(index=False))

# Save full output for later dashboard use.
display.to_csv(
    "live_top50_signals.csv",
    index=False
)

print("\nSaved: live_top50_signals.csv")
print("LIVE SCREEN COMPLETE")
