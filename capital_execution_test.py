import math
from pathlib import Path
import pandas as pd

CAPITAL = 20_000.0
MIN_STOCKS = 10
MAX_STOCKS = 50

INPUT_FILE = Path("live_plan2_top50.csv")
OUTPUT_FILE = Path("capital_execution_test.csv")
ALLOCATION_FILE = Path("capital_auto_allocation.csv")


def find_column(df, names):
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    stock_col = find_column(
        df,
        ["Stock", "Symbol", "Ticker"]
    )

    price_col = find_column(
        df,
        ["Live Price", "Price", "Current Price"]
    )

    if stock_col is None or price_col is None:
        raise ValueError(
            "live_plan2_top50.csv must contain "
            "Stock and Live Price columns. "
            f"Found columns: {list(df.columns)}"
        )

    work = df[[stock_col, price_col]].copy()

    work.columns = [
        "Stock",
        "Live Price"
    ]

    work["Live Price"] = pd.to_numeric(
        work["Live Price"],
        errors="coerce"
    )

    work = work.dropna(
        subset=["Stock", "Live Price"]
    )

    work = work[
        work["Live Price"] > 0
    ].copy()

    work["Stock"] = (
        work["Stock"]
        .astype(str)
        .str.strip()
    )

    work = work.drop_duplicates(
        "Stock"
    ).reset_index(drop=True)

    if len(work) < MIN_STOCKS:
        raise ValueError(
            f"Only {len(work)} valid stocks available. "
            f"At least {MIN_STOCKS} are required."
        )

    # Find the largest feasible stock count
    # between 50 and 10.
    chosen_n = None
    selected = None

    max_possible = min(
        MAX_STOCKS,
        len(work)
    )

    for n in range(
        max_possible,
        MIN_STOCKS - 1,
        -1
    ):

        target_per_stock = CAPITAL / n

        candidates = work[
            work["Live Price"]
            <= target_per_stock
        ].head(n).copy()

        if len(candidates) == n:
            chosen_n = n
            selected = candidates
            break

    # Safety fallback
    if selected is None:

        chosen_n = MIN_STOCKS

        selected = (
            work
            .sort_values("Live Price")
            .head(MIN_STOCKS)
            .copy()
        )

    target_per_stock = CAPITAL / chosen_n

    selected["Target Amount"] = (
        target_per_stock
    )

    selected["Shares"] = (
        selected["Target Amount"]
        / selected["Live Price"]
    ).apply(math.floor).astype(int)

    selected["Invested"] = (
        selected["Shares"]
        * selected["Live Price"]
    )

    selected = selected[
        selected["Shares"] > 0
    ].copy()

    actual_count = len(selected)

    invested = float(
        selected["Invested"].sum()
    )

    cash = CAPITAL - invested

    selected["Weight % of Capital"] = (
        selected["Invested"]
        / CAPITAL
        * 100
    )

    summary = pd.DataFrame([{

        "Capital": CAPITAL,

        "Target Stock Count":
            chosen_n,

        "Executable Stock Count":
            actual_count,

        "Invested":
            round(invested, 2),

        "Cash Remaining":
            round(cash, 2),

        "Min Stocks":
            MIN_STOCKS,

        "Max Stocks":
            MAX_STOCKS,

        "Status":
            "PASS"
            if actual_count >= MIN_STOCKS
            else "REVIEW"

    }])

    summary.to_csv(
        OUTPUT_FILE,
        index=False
    )

    selected.to_csv(
        ALLOCATION_FILE,
        index=False
    )

    print(
        "Capital execution test completed."
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Saved: {ALLOCATION_FILE}"
    )


if __name__ == "__main__":
    main()
