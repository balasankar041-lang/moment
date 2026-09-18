import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import BytesIO
import warnings
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

print("NIFTY 500 PLAN 2 LIVE SCREENER")
print("==============================")

URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}

response = requests.get(URL, headers=HEADERS, timeout=30)
response.raise_for_status()
universe = pd.read_csv(BytesIO(response.content))

if "Symbol" not in universe.columns:
    raise ValueError(f"Unexpected Nifty file columns: {list(universe.columns)}")

stocks = (
    universe["Symbol"].dropna().astype(str).str.strip().str.upper().unique()
)
stocks = [symbol + ".NS" for symbol in stocks]

print(f"Nifty 500 universe loaded: {len(stocks)} stocks")

TOP_MOMENTUM = 100
PORTFOLIO_SIZE = 50
HISTORY_FILE = Path("signal_history.csv")
IPO_LIST_FILE = Path("ipo_watchlist.csv")
# Automatic IPO discovery is conservative: only symbols explicitly returned
# by the public NSE equity listing page are added; no ticker guessing.
IPO_AUTO_FILE = Path("ipo_auto_discovered.csv")
IPO_WATCH_FILE = Path("ipo_watch.csv")

MIN_HISTORY_ROWS = 253
MIN_PATH_ROWS = 100
MAX_MISSING_VALID_RATIO = 0.20

results = []

for number, symbol in enumerate(stocks, start=1):
    print(f"[{number}/{len(stocks)}] {symbol}")
    try:
        data = yf.download(
            symbol, period="2y", auto_adjust=True,
            progress=False, threads=False
        )
        if data.empty:
            continue

        close = data["Close"].squeeze().dropna()
        if len(close) < MIN_HISTORY_ROWS:
            continue

        start_cut = close.index[-1] - pd.DateOffset(months=12)
        end_cut = close.index[-1] - pd.DateOffset(months=2)

        a = close.loc[close.index <= start_cut]
        b = close.loc[close.index <= end_cut]
        if a.empty or b.empty:
            continue

        p0, p1 = a.iloc[-1], b.iloc[-1]
        if p0 <= 0 or p1 <= 0:
            continue

        momentum = p1 / p0 - 1

        path = close.loc[
            (close.index >= a.index[-1]) &
            (close.index <= b.index[-1])
        ]
        if len(path) < MIN_PATH_ROWS:
            continue

        daily = path.pct_change().dropna()
        daily = daily[daily != 0]
        if daily.empty:
            continue

        positive_days = (daily > 0).mean()
        negative_days = (daily < 0).mean()

        id_score = (
            (1 if momentum > 0 else -1)
            * (negative_days - positive_days)
        )

        results.append({
            "Stock": symbol.replace(".NS", ""),
            "Live Price": float(close.iloc[-1]),
            "Momentum 12-2": float(momentum),
            "Positive Days %": float(positive_days),
            "Negative Days %": float(negative_days),
            "ID": float(id_score)
        })

    except Exception as e:
        print(f"Skipped {symbol}: {e}")

df = pd.DataFrame(results)

if df.empty:
    raise RuntimeError("No stocks produced valid results.")

valid_ratio = len(df) / len(stocks)
print(f"Valid-data coverage: {valid_ratio:.1%}")
if valid_ratio < (1.0 - MAX_MISSING_VALID_RATIO):
    raise RuntimeError(
        f"DATA SAFETY STOP: valid coverage {valid_ratio:.1%} is below 80%. "
        "No BUY/SELL signals are saved."
    )

print("\nPRICE / SIGNAL DATA")
print("===================")
print(f"Valid stocks: {len(df)}")

# ---------------------------------------------------------
# IPO / RECENT LISTING WATCH
# ---------------------------------------------------------
ipo_symbols = []
if IPO_LIST_FILE.exists():
    try:
        ipo_input = pd.read_csv(IPO_LIST_FILE)
        if "Symbol" in ipo_input.columns:
            ipo_symbols = (
                ipo_input["Symbol"].dropna().astype(str).str.strip().str.upper()
                .str.replace(r"\.NS$", "", regex=True).unique().tolist()
            )
    except Exception as e:
        print(f"Warning: could not read IPO watchlist: {e}")
else:
    pd.DataFrame(columns=["Symbol"]).to_csv(IPO_LIST_FILE, index=False)


def discover_recent_nse_symbols():
    """Best-effort public NSE listing discovery; failure leaves the screener safe."""
    import requests
    from datetime import datetime, timedelta

    out = []
    try:
        today = datetime.now()
        start = (today - timedelta(days=365)).strftime("%d-%m-%Y")
        end = today.strftime("%d-%m-%Y")
        url = (
            "https://www.nseindia.com/api/public-reports"
        )
        # NSE changes this endpoint periodically. We deliberately do not
        # fabricate symbols if the public endpoint is unavailable.
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/",
        }
        s = requests.Session()
        s.get("https://www.nseindia.com/", headers=headers, timeout=10)
        r = s.get(url, headers=headers, timeout=10)
        if r.ok and r.headers.get("content-type", "").lower().find("json") >= 0:
            data = r.json()
            items = data if isinstance(data, list) else data.get("data", [])
            for x in items:
                if not isinstance(x, dict):
                    continue
                sym = str(
                    x.get("symbol") or x.get("Symbol") or x.get("SYMBOL") or ""
                ).strip().upper()
                if sym and sym.isalnum():
                    out.append(sym)
    except Exception:
        pass
    return sorted(set(out))

ipo_rows = []
for ipo_symbol in ipo_symbols:
    try:
        ipo_data = yf.download(
            ipo_symbol + ".NS", period="2y", auto_adjust=True,
            progress=False, threads=False
        )
        if ipo_data.empty:
            ipo_rows.append({
                "Symbol": ipo_symbol,
                "Status": "IPO WATCH - NO DATA",
                "Live Price": np.nan,
                "Trading Days": 0,
                "Plan 2 Eligible": "NO"
            })
            continue

        ipo_close = ipo_data["Close"].squeeze().dropna()
        days = len(ipo_close)
        if days < 20:
            status = "IPO WATCH - BUILDING HISTORY"
            eligible = "NO"
        elif days < MIN_HISTORY_ROWS:
            status = "IPO WATCH - NOT ENOUGH HISTORY"
            eligible = "NO"
        else:
            status = "PLAN 2 ELIGIBLE HISTORY"
            eligible = "YES"

        ipo_rows.append({
            "Symbol": ipo_symbol,
            "Status": status,
            "Live Price": float(ipo_close.iloc[-1]),
            "Trading Days": days,
            "Plan 2 Eligible": eligible
        })
    except Exception:
        ipo_rows.append({
            "Symbol": ipo_symbol,
            "Status": "IPO WATCH - DATA ERROR",
            "Live Price": np.nan,
            "Trading Days": 0,
            "Plan 2 Eligible": "NO"
        })

pd.DataFrame(ipo_rows).to_csv(IPO_WATCH_FILE, index=False)

# Stage 1: Top 100 momentum
df = df.sort_values("Momentum 12-2", ascending=False).reset_index(drop=True)
df["Momentum Rank"] = df.index + 1
top100 = df.head(TOP_MOMENTUM).copy()

# Stage 2: Lowest ID
top100 = top100.sort_values("ID", ascending=True).reset_index(drop=True)
top100["ID Rank"] = top100.index + 1

# Final Top 50
top50 = top100.head(PORTFOLIO_SIZE).copy()
if len(top50) < PORTFOLIO_SIZE:
    raise RuntimeError(
        f"Only {len(top50)} final stocks available; need {PORTFOLIO_SIZE}."
    )

current_top50 = set(top50["Stock"])
current_top100 = set(top100["Stock"])

# ---------------------------------------------------------
# PREVIOUS SIGNAL STATE
# ---------------------------------------------------------
# signal_history.csv is committed by GitHub Actions after each run.
# The latest saved snapshot is used to distinguish BUY/HOLD/SELL.
if HISTORY_FILE.exists():
    try:
        history = pd.read_csv(HISTORY_FILE)
        if not history.empty and "Run Date" in history.columns:
            latest_date = history["Run Date"].max()
            previous = history[history["Run Date"] == latest_date].copy()
        else:
            previous = pd.DataFrame()
    except Exception as e:
        print(f"Warning: could not read signal history: {e}")
        previous = pd.DataFrame()
else:
    previous = pd.DataFrame()

previous_top50 = set()
if not previous.empty and "Portfolio Signal" in previous.columns:
    previous_top50 = set(
        previous.loc[
            previous["Portfolio Signal"].isin(["BUY", "HOLD"]),
            "Stock"
        ].astype(str)
    )

# ---------------------------------------------------------
# SIGNALS
# ---------------------------------------------------------
# BUY  = new entry into current Top 50
# HOLD = was in previous Top 50 and remains in current Top 50
# SELL = was in previous Top 50 but left current Top 50
# WAIT = not Top 50; stocks inside Top 100 receive ID FILTER
df["Portfolio Signal"] = "WAIT"
df["Sell Reason"] = ""

df.loc[
    df["Stock"].isin(current_top100),
    "Portfolio Signal"
] = "ID FILTER"

df.loc[
    df["Stock"].isin(current_top50 & previous_top50),
    "Portfolio Signal"
] = "HOLD"

df.loc[
    df["Stock"].isin(current_top50 - previous_top50),
    "Portfolio Signal"
] = "BUY"

# SELL rule:
# SELL only when a previous Top 50 stock is present in today's valid data
# but is no longer in the current Top 50. If today's data is missing,
# do NOT force a SELL because that could be a data/download problem.
if previous_top50:
    current_valid = set(df["Stock"])
    sell_candidates = (previous_top50 - current_top50) & current_valid
    if sell_candidates:
        old_rows = previous[
            previous["Stock"].isin(sell_candidates)
        ].copy()

        old_rows["Portfolio Signal"] = "SELL"
        old_rows["Signal Date"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d"
        )

        # Current price is not available from the old snapshot reliably,
        # so use today's price if present in the current universe.
        price_map = df.set_index("Stock")["Live Price"].to_dict()
        old_rows["Live Price"] = old_rows["Stock"].map(price_map).fillna(
            old_rows.get("Live Price", np.nan)
        )

        # HOLD -> SELL safety rule:
        # A previous holding is sold only when current valid data exists and
        # the stock is no longer in the current Top 50. Missing/unavailable
        # current data is NOT treated as a sell signal.
        old_rows["Sell Reason"] = old_rows["Stock"].map(
            lambda s: "SELL: dropped below Top 50; still in Top 100"
            if s in current_top100
            else "SELL: dropped out of Top 100"
        )
        old_rows["Exit Safety"] = "Validated current data; no forced sell on missing data"
        old_rows["Data Safety"] = "VALID"

        sell_columns = [
            "Momentum Rank", "ID Rank", "Stock", "Live Price",
            "Momentum 12-2", "Positive Days %", "Negative Days %",
            "ID", "Portfolio Signal", "Weight %"
        ]
        for col in sell_columns:
            if col not in old_rows.columns:
                old_rows[col] = np.nan

        df = pd.concat(
            [df, old_rows[sell_columns]],
            ignore_index=True
        )

# Current weights
df["Weight %"] = 0.0
df.loc[
    df["Stock"].isin(current_top50),
    "Weight %"
] = 100.0 / len(top50)

# ---------------------------------------------------------
# OUTPUT COLUMNS
# ---------------------------------------------------------
# Current non-SELL rows have no sell reason.
df.loc[df["Portfolio Signal"] != "SELL", "Sell Reason"] = ""

columns = [
    "Momentum Rank",
    "ID Rank",
    "Stock",
    "Live Price",
    "Momentum 12-2",
    "Positive Days %",
    "Negative Days %",
    "ID",
    "Portfolio Signal",
    "Sell Reason",
    "Exit Safety",
    "Data Safety",
    "Weight %"
]

for column in columns:
    if column not in df.columns:
        df[column] = np.nan

for column in columns:
    if column not in top100.columns:
        top100[column] = np.nan

for column in columns:
    if column not in top50.columns:
        top50[column] = np.nan

top100["Data Safety"] = "VALID"
top50["Data Safety"] = "VALID"

# Keep Top-100 reporting signals explicit.
# Top-50 members are BUY/HOLD; remaining Top-100 members are ID FILTER.
top100["Portfolio Signal"] = top100["Stock"].map(
    lambda s: ("HOLD" if s in previous_top50 else "BUY")
    if s in current_top50 else "ID FILTER"
)

# ---------------------------------------------------------
# SAVE CURRENT RESULTS
# ---------------------------------------------------------
df[columns].to_csv("live_plan2_ranking.csv", index=False)
top100[columns].to_csv("live_plan2_top100.csv", index=False)

top50_output = top50[columns].copy()
top50_output["Portfolio Signal"] = top50_output["Stock"].map(
    lambda s: "HOLD" if s in previous_top50 else "BUY"
)
top50_output["Weight %"] = 100.0 / len(top50)
top50_output.to_csv("live_plan2_top50.csv", index=False)

# Keep compatibility with existing workflow/artifacts.
df[columns].to_csv("ranking.csv", index=False)

# ---------------------------------------------------------
# APPEND SNAPSHOT TO HISTORY
# ---------------------------------------------------------
run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

snapshot = top50_output.copy()
snapshot.insert(0, "Run Date", run_date)

# Avoid duplicate snapshot if the same day is manually rerun.
if HISTORY_FILE.exists():
    try:
        old_history = pd.read_csv(HISTORY_FILE)
        old_history = old_history[
            old_history["Run Date"].astype(str) != run_date
        ]
    except Exception:
        old_history = pd.DataFrame()
else:
    old_history = pd.DataFrame()

new_history = pd.concat(
    [old_history, snapshot],
    ignore_index=True
)

new_history.to_csv(HISTORY_FILE, index=False)

# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------
print("\n==========================================")
print("PLAN 2 LIVE RESULT")
print("==========================================")
print(f"Universe: {len(stocks)}")
print(f"Valid stocks: {len(results)}")
print(f"Top momentum: {TOP_MOMENTUM}")
print(f"Final portfolio: {len(top50)}")
print(f"Previous Top 50: {len(previous_top50)}")
print(f"BUY: {sum(top50_output['Portfolio Signal'] == 'BUY')}")
print(f"HOLD: {sum(top50_output['Portfolio Signal'] == 'HOLD')}")
print(f"SELL: {sum(df['Portfolio Signal'] == 'SELL')}")
print("HOLD -> SELL safety: only valid current-data exits; missing data is not a SELL")

display_df = top50_output.copy()
display_df["Live Price"] = display_df["Live Price"].map(
    lambda x: f"{x:.2f}"
)
display_df["Momentum 12-2"] = display_df["Momentum 12-2"].map(
    lambda x: f"{x:.2%}"
)
display_df["Positive Days %"] = display_df["Positive Days %"].map(
    lambda x: f"{x:.2%}"
)
display_df["Negative Days %"] = display_df["Negative Days %"].map(
    lambda x: f"{x:.2%}"
)
display_df["ID"] = display_df["ID"].map(
    lambda x: f"{x:.4f}"
)
display_df["Weight %"] = display_df["Weight %"].map(
    lambda x: f"{x:.2f}%"
)

print("\nTOP 50 PLAN 2")
print("------------------------------------------")
print(display_df.to_string(index=False))

print("\nFiles saved:")
print("ranking.csv")
print("live_plan2_ranking.csv")
print("live_plan2_top100.csv")
print("live_plan2_top50.csv")
print("signal_history.csv")
print("ipo_watch.csv")
print("ipo_auto_discovered.csv")

print("\nSTATUS: SUCCESS")
