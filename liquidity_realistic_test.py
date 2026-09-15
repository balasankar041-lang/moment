import pandas as pd
import numpy as np
import yfinance as yf

# ============================================================
# MULTI-CAPITAL REALISTIC LIQUIDITY TEST
# ============================================================

CAPITALS = [
    20_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
    2_500_000,
    5_000_000,
    10_000_000,
]

PORTFOLIO_SIZE = 50

# Order as % of average daily traded value
LIQUIDITY_LIMITS = [0.01, 0.02, 0.05, 0.10]

START_DATE = "2020-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")

UNIVERSE_FILE = "nifty500_membership_timeline.csv"


def load_symbols():
    df = pd.read_csv(UNIVERSE_FILE)

    possible = ["symbol", "Symbol", "ticker", "Ticker", "SYMBOL"]
    column = next((c for c in possible if c in df.columns), None)

    if column is None:
        raise ValueError(
            f"No symbol column found. Columns: {list(df.columns)}"
        )

    return (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
        .tolist()
    )


def yahoo_symbol(symbol):
    return symbol if symbol.endswith(".NS") else symbol + ".NS"


def main():

    print("=" * 75)
    print("MULTI-CAPITAL REALISTIC EXECUTION TEST")
    print("=" * 75)

    print(f"Portfolio stocks: {PORTFOLIO_SIZE}")
    print("Execution: whole shares only")
    print()

    symbols = load_symbols()
    print(f"Universe symbols: {len(symbols)}")
    print()

    stock_rows = []

    for i, symbol in enumerate(symbols, 1):

        try:
            data = yf.download(
                yahoo_symbol(symbol),
                start=START_DATE,
                end=END_DATE,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if not {"Close", "Volume"}.issubset(data.columns):
                continue

            data = data.dropna(subset=["Close", "Volume"])

            if data.empty:
                continue

            price = float(data["Close"].iloc[-1])

            recent = data.tail(100)

            traded_value = (
                recent["Close"] * recent["Volume"]
            ).replace([np.inf, -np.inf], np.nan).dropna()

            if traded_value.empty:
                continue

            avg_daily_value = float(traded_value.mean())
            median_daily_value = float(traded_value.median())

            stock_rows.append({
                "symbol": symbol,
                "price": price,
                "avg_daily_traded_value": avg_daily_value,
                "median_daily_traded_value": median_daily_value,
            })

        except Exception:
            continue

        if i % 50 == 0:
            print(f"Processed {i}/{len(symbols)}")

    stocks = pd.DataFrame(stock_rows)

    if stocks.empty:
        raise RuntimeError("No usable stock data found.")

    results = []

    for capital in CAPITALS:

        target = capital / PORTFOLIO_SIZE

        temp = stocks.copy()

        temp["target_value"] = target

        # Whole shares only
        temp["shares"] = (
            temp["target_value"] // temp["price"]
        ).astype(int)

        temp["actual_value"] = (
            temp["shares"] * temp["price"]
        )

        temp["cash_shortfall"] = (
            temp["target_value"] - temp["actual_value"]
        )

        temp["order_vs_avg_daily_value"] = np.where(
            temp["avg_daily_traded_value"] > 0,
            temp["actual_value"] /
            temp["avg_daily_traded_value"],
            np.nan
        )

        executable = temp[temp["shares"] > 0].copy()

        if executable.empty:
            continue

        for limit in LIQUIDITY_LIMITS:

            affected = (
                executable["order_vs_avg_daily_value"] > limit
            ).mean() * 100

            results.append({
                "capital": capital,
                "target_per_stock": target,
                "executable_stocks": len(executable),
                "stocks_unable_to_buy": len(temp) - len(executable),
                "avg_actual_allocation": executable["actual_value"].mean(),
                "avg_unused_target": executable["cash_shortfall"].mean(),
                "liquidity_limit": limit,
                "stocks_above_liquidity_limit_pct": affected,
            })

    result_df = pd.DataFrame(results)

    output = "liquidity_realistic_results.csv"
    result_df.to_csv(output, index=False)

    print()
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)

    summary = (
        result_df[
            result_df["liquidity_limit"] == 0.02
        ][
            [
                "capital",
                "target_per_stock",
                "executable_stocks",
                "stocks_unable_to_buy",
                "avg_actual_allocation",
                "avg_unused_target",
                "stocks_above_liquidity_limit_pct",
            ]
        ]
    )

    print(summary.to_string(index=False))

    print()
    print("=" * 75)
    print("2% LIQUIDITY RULE")
    print("=" * 75)
    print(
        "A stock is flagged when the order is >2% "
        "of its average daily traded value."
    )

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
