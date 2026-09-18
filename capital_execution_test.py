
from pathlib import Path
import pandas as pd
import math

CAPITAL = 20_000.0
SIZES = [10, 15, 20, 25, 50]
TOP50_FILE = Path("live_plan2_top50.csv")

if not TOP50_FILE.exists():
    raise SystemExit("live_plan2_top50.csv not found")

df = pd.read_csv(TOP50_FILE)

# Current Plan 2 output uses "Stock" and "Live Price".
symbol_col = next(
    (c for c in ["Stock", "Symbol", "symbol", "Ticker", "ticker"] if c in df.columns),
    None
)
price_col = next(
    (c for c in ["Live Price", "Price", "price", "Close", "close"] if c in df.columns),
    None
)

if symbol_col is None or price_col is None:
    raise SystemExit(
        "Required columns not found. "
        f"Found columns: {list(df.columns)}"
    )

df = df[[symbol_col, price_col]].copy()
df.columns = ["Symbol", "Price"]
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Price"])
df = df[df["Price"] > 0].reset_index(drop=True)

if df.empty:
    raise SystemExit("No valid stock prices found in live_plan2_top50.csv")

rows = []

for n in SIZES:
    selected = df.head(n).copy()

    target = CAPITAL / n

    # Whole shares only. No fractional shares.
    selected["Shares"] = (target / selected["Price"]).apply(math.floor)
    selected["Invested"] = selected["Shares"] * selected["Price"]

    executable = int((selected["Shares"] >= 1).sum())
    invested = float(selected["Invested"].sum())
    leftover = CAPITAL - invested

    rows.append({
        "Portfolio Stocks": n,
        "Capital": CAPITAL,
        "Target per Stock": round(target, 2),
        "Executable Stocks": executable,
        "Unexecutable Stocks": n - executable,
        "Invested": round(invested, 2),
        "Leftover Cash": round(leftover, 2),
        "Capital Used %": round(invested / CAPITAL * 100, 2),
    })

out = pd.DataFrame(rows)
out.to_csv("capital_execution_test.csv", index=False)

print("\n₹20,000 CAPITAL EXECUTION TEST")
print("=" * 80)
print(out.to_string(index=False))
print("=" * 80)
print("Created: capital_execution_test.csv")
