import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

print("NIFTY 500 PLAN 2 LIVE SCREENER")
print("==============================")

# =========================================================
# NIFTY 500 UNIVERSE
# =========================================================

URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}

response = requests.get(URL, headers=HEADERS, timeout=30)
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

# =========================================================
# VALIDATED PLAN 2
# =========================================================
# 1. 12-2 month momentum
# 2. Top 100 momentum stocks
# 3. Lowest ID from Top 100
# 4. Final Top 50
# 5. Equal weight
# =========================================================

TOP_MOMENTUM = 100
PORTFOLIO_SIZE = 50

results = []

# =========================================================
# DOWNLOAD + CALCULATE
# =========================================================

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

        # -------------------------------------------------
        # 12-2 MONTH MOMENTUM
        # Same definition as validated backtest
        # -------------------------------------------------

        start_cut = close.index[-1] - pd.DateOffset(months=12)
        end_cut = close.index[-1] - pd.DateOffset(months=2)

        a = close.loc[close.index <= start_cut]
        b = close.loc[close.index <= end_cut]

        if a.empty or b.empty:
            continue

        p0 = a.iloc[-1]
        p1 = b.iloc[-1]

        if p0 <= 0 or p1 <= 0:
            continue

        momentum = p1 / p0 - 1

        # -------------------------------------------------
        # ID
        # Exact formula from validated backtest
        # -------------------------------------------------

        path = close.loc[
            (close.index >= a.index[-1])
            & (close.index <= b.index[-1])
        ]

        if len(path) < 100:
            continue

        daily = path.pct_change().dropna()

        # Zero-return days excluded
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

# =========================================================
# VALID RESULTS
# =========================================================

df = pd.DataFrame(results)

if df.empty:
    raise RuntimeError("No stocks produced valid results.")

print("\nPRICE / SIGNAL DATA")
print("===================")
print(f"Valid stocks: {len(df)}")

# =========================================================
# STAGE 1
# TOP 100 MOMENTUM
# =========================================================

df = (
    df.sort_values(
        "Momentum 12-2",
        ascending=False
    )
    .reset_index(drop=True)
)

df["Momentum Rank"] = df.index + 1

top100 = df.head(TOP_MOMENTUM).copy()

# =========================================================
# STAGE 2
# LOWEST ID
# =========================================================

top100 = (
    top100.sort_values(
        "ID",
        ascending=True
    )
    .reset_index(drop=True)
)

top100["ID Rank"] = top100.index + 1

# =========================================================
# FINAL TOP 50
# =========================================================

top50 = top100.head(PORTFOLIO_SIZE).copy()

if len(top50) < PORTFOLIO_SIZE:
    raise RuntimeError(
        f"Only {len(top50)} final stocks available; "
        f"need {PORTFOLIO_SIZE}."
    )

top50["Signal"] = "BUY"
top50["Weight %"] = 100.0 / len(top50)

# =========================================================
# FULL UNIVERSE SIGNALS
# =========================================================

top100_symbols = set(top100["Stock"])
selected_symbols = set(top50["Stock"])

df["Signal"] = "WAIT"
df["Weight %"] = 0.0

# ID Rank exists for all rows.
# Only Top 100 receive an actual rank.
id_rank_map = top100.set_index("Stock")["ID Rank"]
df["ID Rank"] = df["Stock"].map(id_rank_map)

# Top 100 but not final Top 50
df.loc[
    df["Stock"].isin(top100_symbols),
    "Signal"
] = "ID FILTER"

# Final Top 50
df.loc[
    df["Stock"].isin(selected_symbols),
    "Signal"
] = "BUY"

df.loc[
    df["Stock"].isin(selected_symbols),
    "Weight %"
] = 100.0 / len(top50)

# =========================================================
# OUTPUT COLUMNS
# =========================================================

columns = [
    "Momentum Rank",
    "ID Rank",
    "Stock",
    "Live Price",
    "Momentum 12-2",
    "Positive Days %",
    "Negative Days %",
    "ID",
    "Signal",
    "Weight %"
]

# Ensure all output columns exist
for column in columns:
    if column not in df.columns:
        df[column] = np.nan

for column in columns:
    if column not in top100.columns:
        top100[column] = np.nan

for column in columns:
    if column not in top50.columns:
        top50[column] = np.nan

# =========================================================
# SAVE CSV FILES
# =========================================================

df[columns].to_csv(
    "live_plan2_ranking.csv",
    index=False
)

top100[columns].to_csv(
    "live_plan2_top100.csv",
    index=False
)

top50[columns].to_csv(
    "live_plan2_top50.csv",
    index=False
)

# Keep the original ranking.csv output too,
# so existing workflow/artifacts do not break.
df[columns].to_csv(
    "ranking.csv",
    index=False
)

# =========================================================
# DISPLAY FINAL RESULT
# =========================================================

print("\n==========================================")
print("PLAN 2 LIVE RESULT")
print("==========================================")

print(f"Universe: {len(stocks)}")
print(f"Valid stocks: {len(df)}")
print(f"Top momentum: {TOP_MOMENTUM}")
print(f"Final portfolio: {len(top50)}")

print("\nTOP 50 PLAN 2")
print("------------------------------------------")

display_df = top50[columns].copy()

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

print(display_df.to_string(index=False))

print("\nFiles saved:")
print("ranking.csv")
print("live_plan2_ranking.csv")
print("live_plan2_top100.csv")
print("live_plan2_top50.csv")

print("\nSTATUS: SUCCESS")
