"""
Plan 2 practical capital allocation

Rule:
- Capital = ₹20,000
- Select 10 affordable stocks from Plan 2 Top 50
- Maximum ₹1,500 per stock
- Whole shares only
- Total investment <= ₹20,000
- Plan 2 ranking is unchanged
"""

from pathlib import Path
import math
import pandas as pd

CAPITAL = 20000.0
MAX_PER_STOCK = 1500.0
TARGET_STOCKS = 12

TOP50_FILE = Path("live_plan2_top50.csv")
OUTPUT_FILE = Path("capital_auto_allocation.csv")


def pick(row, names):
    for name in names:
        if name in row.index:
            value = row[name]
            if pd.notna(value) and str(value).strip() != "":
                return value
    return ""


def to_num(value):
    try:
        return float(
            str(value)
            .replace(",", "")
            .replace("%", "")
            .strip()
        )
    except Exception:
        return 0.0


def main():

    if not TOP50_FILE.exists():
        raise FileNotFoundError(
            f"{TOP50_FILE} not found"
        )

    df = pd.read_csv(TOP50_FILE)

    if df.empty:
        raise RuntimeError(
            "live_plan2_top50.csv is empty"
        )

    rows = []
    total_invested = 0.0

    # Keep Plan 2 ranking order unchanged
    for _, r in df.iterrows():

        if len(rows) >= TARGET_STOCKS:
            break

        stock = str(
            pick(r, ["Stock", "Symbol", "Ticker"])
        ).strip()

        price = to_num(
            pick(
                r,
                [
                    "Live Price",
                    "Price",
                    "Current Price"
                ]
            )
        )

        if not stock or price <= 0:
            continue

        # Only stocks whose share price is <= ₹1,500
        if price > MAX_PER_STOCK:
            continue

        remaining_capital = (
            CAPITAL - total_invested
        )

        if remaining_capital <= 0:
            break

        # Whole shares only
        shares = math.floor(
            min(
                MAX_PER_STOCK,
                remaining_capital
            ) / price
        )

        if shares < 1:
            continue

        invested = round(
            shares * price,
            2
        )

        if invested <= 0:
            continue

        if invested > MAX_PER_STOCK:
            continue

        if (
            total_invested + invested
            > CAPITAL + 1e-9
        ):
            continue

        total_invested += invested

        rows.append(
            {
                "Rank": pick(
                    r,
                    [
                        "Rank",
                        "ID Rank",
                        "Portfolio Rank"
                    ]
                ),

                "Stock": stock,

                "Live Price": round(
                    price,
                    2
                ),

                "Momentum %": (
                    lambda m: round(m * 100, 2) if abs(m) <= 2 else round(m, 2)
                )(
                    to_num(
                        pick(
                            r,
                            [
                                "Momentum 12-2",
                                "Momentum %",
                                "Momentum",
                                "Momentum Return"
                            ]
                        )
                    )
                ),

                "ID": to_num(
                    pick(
                        r,
                        [
                            "ID",
                            "Id",
                            "ID Score"
                        ]
                    )
                ),

                "Signal": pick(
                    r,
                    [
                        "Signal",
                        "Portfolio Signal"
                    ]
                ),

                "Target Amount": invested,

                "Shares": int(shares),

                "Invested Amount": invested
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:

        out["Weight %"] = (
            out["Invested Amount"]
            / CAPITAL
            * 100
        ).round(2)

        cash = round(
            CAPITAL - total_invested,
            2
        )

        out["Cash After Allocation"] = cash

    else:

        out = pd.DataFrame(
            columns=[
                "Rank",
                "Stock",
                "Live Price",
                "Momentum %",
                "ID",
                "Signal",
                "Target Amount",
                "Shares",
                "Invested Amount",
                "Weight %",
                "Cash After Allocation"
            ]
        )

    # Safety checks

    if len(out) > TARGET_STOCKS:
        raise RuntimeError(
            "More than 10 stocks selected"
        )

    if not out.empty:

        if (
            out["Invested Amount"]
            > MAX_PER_STOCK + 1e-9
        ).any():

            raise RuntimeError(
                "A stock exceeded ₹1,500"
            )

        if (
            out["Invested Amount"].sum()
            > CAPITAL + 1e-9
        ):

            raise RuntimeError(
                "Allocation exceeded ₹20,000"
            )

    out.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "Plan 2 practical allocation"
    )

    print(
        f"Capital: ₹{CAPITAL:,.2f}"
    )

    print(
        f"Max per stock: ₹{MAX_PER_STOCK:,.2f}"
    )

    print(
        f"Target stocks: {TARGET_STOCKS}"
    )

    print(
        f"Selected stocks: {len(out)}"
    )

    print(
        f"Invested: ₹{total_invested:,.2f}"
    )

    print(
        f"Cash: ₹{CAPITAL-total_invested:,.2f}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    if not out.empty:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
