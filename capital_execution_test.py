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

TOP50_FILE = Path("live_plan2_top50.csv")
OUTPUT_FILE = Path("capital_auto_allocation.csv")


def allocation_rules(capital):
    if 1000 <= capital <= 2999:
        return 2, 1000.0
    if 3000 <= capital <= 4999:
        return 3, 1250.0
    if 5000 <= capital <= 9999:
        return 5, 1500.0
    if 10000 <= capital <= 19999:
        return 8, 2000.0
    if 20000 <= capital <= 29999:
        return 10, 2500.0
    if capital >= 30000:
        return 12, 3000.0
    return 0, 0.0


TARGET_STOCKS, MAX_PER_STOCK = allocation_rules(CAPITAL)

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

                "Momentum %": to_num(
                    pick(
                        r,
                        [
                            "Momentum %",
                            "Momentum",
                            "Momentum Return"
                        ]
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
            "More than TARGET_STOCKS stocks selected"
        )

    if not out.empty:

        if (
            out["Invested Amount"]
            > MAX_PER_STOCK + 1e-9
        ).any():

            raise RuntimeError(
                "A stock exceeded MAX_PER_STOCK"
            )

        if (
            out["Invested Amount"].sum()
            > CAPITAL + 1e-9
        ):

            raise RuntimeError(
                "Allocation exceeded CAPITAL"
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
