import pandas as pd
from pathlib import Path
import math

CAPITAL = 20_000.0
MAX_PER_STOCK = 1_500.0
MIN_STOCKS = 10

INPUT_FILE = Path("live_plan2_top50.csv")
OUTPUT_FILE = Path("capital_execution_test.csv")
ALLOCATION_FILE = Path("capital_auto_allocation.csv")


def find_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found")

    df = pd.read_csv(INPUT_FILE)

    stock_col = find_column(df, ["Stock", "Symbol", "Ticker"])
    price_col = find_column(df, ["Live Price", "Price", "Close"])

    if stock_col is None or price_col is None:
        raise ValueError(
            f"Required columns not found. Available columns: {list(df.columns)}"
        )

    work = df[[stock_col, price_col]].copy()
    work.columns = ["Stock", "Live Price"]

    work["Live Price"] = pd.to_numeric(work["Live Price"], errors="coerce")
    work = work.dropna(subset=["Stock", "Live Price"])
    work = work[work["Live Price"] > 0].copy()

    # Keep the existing Plan 2 Top-50 order.
    # Only stocks priced at or below ₹1,500 are eligible.
    eligible = work[work["Live Price"] <= MAX_PER_STOCK].copy()

    # ₹20,000 / ₹1,500 = maximum 13 stocks.
    max_possible = min(
        len(eligible),
        math.floor(CAPITAL / MAX_PER_STOCK)
    )

    target_count = max_possible

    if target_count < MIN_STOCKS:
        target_count = min(len(eligible), MIN_STOCKS)

    selected = eligible.head(target_count).copy()

    # Whole shares only.
    selected["Target Amount"] = MAX_PER_STOCK
    selected["Shares"] = (
        selected["Target Amount"] / selected["Live Price"]
    ).apply(math.floor)

    selected = selected[selected["Shares"] >= 1].copy()

    selected["Invested Amount"] = (
        selected["Shares"] * selected["Live Price"]
    )

    # Hard capital safety check.
    while (
        selected["Invested Amount"].sum() > CAPITAL
        and len(selected) > MIN_STOCKS
    ):
        selected = selected.iloc[:-1].copy()

    invested = float(selected["Invested Amount"].sum())
    cash = CAPITAL - invested

    selected["Weight %"] = (
        selected["Invested Amount"] / CAPITAL * 100
    )

    status = "PASS"

    if invested > CAPITAL + 1e-9:
        status = "FAIL: capital exceeded"
    elif len(selected) < MIN_STOCKS:
        status = f"CAUTION: only {len(selected)} executable stocks"

    allocation = selected[
        [
            "Stock",
            "Live Price",
            "Target Amount",
            "Shares",
            "Invested Amount",
            "Weight %",
        ]
    ].copy()

    allocation.to_csv(ALLOCATION_FILE, index=False)

    summary = pd.DataFrame([{
        "Capital": CAPITAL,
        "Max Per Stock": MAX_PER_STOCK,
        "Minimum Stocks": MIN_STOCKS,
        "Target Stock Count": target_count,
        "Executable Stock Count": len(selected),
        "Invested": round(invested, 2),
        "Cash Remaining": round(cash, 2),
        "Status": status,
    }])

    summary.to_csv(OUTPUT_FILE, index=False)

    print("=== ₹20,000 CAPITAL EXECUTION TEST ===")
    print(f"Max per stock: ₹{MAX_PER_STOCK:,.0f}")
    print(f"Minimum stocks: {MIN_STOCKS}")
    print(f"Target stock count: {target_count}")
    print(f"Executable stock count: {len(selected)}")
    print(f"Invested: ₹{invested:,.2f}")
    print(f"Cash remaining: ₹{cash:,.2f}")
    print(f"Status: {status}")

    print()
    print("=== AUTOMATIC ALLOCATION ===")
    print(allocation.to_string(index=False))


if __name__ == "__main__":
    main()
