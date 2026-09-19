
from pathlib import Path
import pandas as pd
import math

CAPITAL = 20_000.0
SIZES = [10, 15, 20, 25, 50]
TOP50_FILE = Path("live_plan2_top50.csv")

if not TOP50_FILE.exists():
    raise SystemExit("live_plan2_top50.csv not found")

df = pd.read_csv(TOP50_FILE)
price_col = next((c for c in ["Price", "price", "Close", "close"] if c in df.columns), None)
symbol_col = next((c for c in ["Symbol", "symbol", "Ticker", "ticker"] if c in df.columns), None)

if price_col is None or symbol_col is None:
    raise SystemExit(f"Required columns not found. Columns: {list(df.columns)}")

df = df[[symbol_col, price_col]].copy()
df.columns = ["Symbol", "Price"]
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Price"])
df = df[df["Price"] > 0].reset_index(drop=True)

rows = []
for n in SIZES:
    selected = df.head(n).copy()
    target = CAPITAL / n
    selected["Shares"] = (target / selected["Price"]).apply(math.floor)
    selected["Invested"] = selected["Shares"] * selected["Price"]
    executable = int((selected["Shares"] >= 1).sum())
    invested = float(selected["Invested"].sum())
    leftover = CAPITAL - invested
    rows.append({
        "Portfolio Stocks": n,
        "Capital": CAPITAL,
        "Target per Stock": target,
        "Executable Stocks": executable,
        "Unexecutable Stocks": n - executable,
        "Invested": invested,
        "Leftover Cash": leftover,
        "Capital Used %": invested / CAPITAL * 100,
    })

out = pd.DataFrame(rows)
out.to_csv("capital_execution_test.csv", index=False)

print(out.to_string(index=False))
print("\nCreated capital_execution_test.csv")
