
from pathlib import Path
import pandas as pd
import numpy as np

CAPITAL = 20_000.0
SIZES = [10, 15, 20, 25, 50]

# This test is intentionally separate from the live Plan 2 strategy.
# It does not modify main.py or the live signal logic.
SELECTION_FILE = Path("plan2_monthly_selections.csv")

if not SELECTION_FILE.exists():
    raise SystemExit(
        "plan2_monthly_selections.csv not found. "
        "This backtest needs historical monthly Plan 2 selections; "
        "it will not invent them."
    )

df = pd.read_csv(SELECTION_FILE)

required = {"Date", "Stock", "Price", "Next Month Return"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(
        f"Missing columns: {sorted(missing)}. "
        f"Found: {list(df.columns)}"
    )

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Next Month Return"] = pd.to_numeric(
    df["Next Month Return"], errors="coerce"
)
df = df.dropna(subset=["Date", "Price", "Next Month Return"])
df = df[df["Price"] > 0].copy()

results = []

for n in SIZES:
    monthly = []

    for date, g in df.sort_values(["Date", "Stock"]).groupby("Date"):
        selected = g.head(n).copy()
        if selected.empty:
            continue

        # Equal target capital per selected stock, whole shares only.
        target = CAPITAL / n
        selected["Shares"] = np.floor(target / selected["Price"])
        selected["Invested"] = selected["Shares"] * selected["Price"]

        invested = selected["Invested"].sum()
        cash = CAPITAL - invested

        # One-month portfolio return on the actual invested capital.
        # Cash earns 0% in this conservative test.
        pnl = (selected["Invested"] * selected["Next Month Return"]).sum()
        portfolio_return = pnl / CAPITAL

        monthly.append({
            "Date": date,
            "Return": portfolio_return,
            "Invested": invested,
            "Cash": cash,
            "Executable": int((selected["Shares"] >= 1).sum()),
        })

    if not monthly:
        continue

    m = pd.DataFrame(monthly).sort_values("Date")
    wealth = (1 + m["Return"]).cumprod()
    total = wealth.iloc[-1] - 1
    years = len(m) / 12
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = m["Return"].std(ddof=1) * np.sqrt(12)
    sharpe = (
        m["Return"].mean() / m["Return"].std(ddof=1) * np.sqrt(12)
        if m["Return"].std(ddof=1) > 0 else np.nan
    )
    dd = wealth / wealth.cummax() - 1

    results.append({
        "Stocks": n,
        "Months": len(m),
        "Total Return %": total * 100,
        "CAGR %": cagr * 100,
        "Volatility %": vol * 100,
        "Sharpe": sharpe,
        "Max Drawdown %": dd.min() * 100,
        "Win Rate %": (m["Return"] > 0).mean() * 100,
        "Avg Invested ₹": m["Invested"].mean(),
        "Avg Cash ₹": m["Cash"].mean(),
        "Avg Executable Stocks": m["Executable"].mean(),
    })

out = pd.DataFrame(results)
out.to_csv("capital_sizing_backtest.csv", index=False)

print("\n₹20,000 CAPITAL SIZING BACKTEST")
print("=" * 110)
print(out.to_string(index=False))
print("=" * 110)
print("Created: capital_sizing_backtest.csv")
