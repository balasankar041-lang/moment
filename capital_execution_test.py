"""
Plan 2 practical capital allocation

Rules (aligned with auto_allocation.js and index.html):
- Default Capital = ₹20,000
- Target stock count and max per stock come from allocation_rules(capital)
  (for ₹20,000: 10 target stocks, max ₹1,500 per stock)
- Maximum 3 stocks per sector (when sector_wise_top50.csv is available)
- Whole shares only
- Total investment <= CAPITAL
- Plan 2 ranking (ID Rank) is unchanged
"""

from pathlib import Path
import math
import pandas as pd

CAPITAL = 20000.0
MAX_PER_SECTOR = 3

TOP50_FILE = Path("live_plan2_top50.csv")
SECTOR_FILE = Path("sector_wise_top50.csv")
OUTPUT_FILE = Path("capital_auto_allocation.csv")
SUMMARY_FILE = Path("capital_execution_test.csv")


def allocation_rules(capital):
    capital = float(capital)
    if capital <= 0:
        return 0, 0.0
    if capital < 3000:
        target_stocks = 2
    elif capital < 5000:
        target_stocks = 3
    elif capital < 10000:
        target_stocks = 5
    elif capital < 20000:
        target_stocks = 8
    elif capital < 30000:
        target_stocks = 10
    elif capital < 50000:
        target_stocks = 12
    elif capital < 100000:
        target_stocks = 15
    else:
        target_stocks = 20

    # ₹1,500 max per stock for ₹20,000 capital (7.5% rule, bounded [₹1,000, ₹5,000])
    max_per_stock = min(5000.0, max(1000.0, round(capital * 0.075, 2)))
    return target_stocks, max_per_stock


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
        return float(str(value).replace("₹", "").replace(",", "").replace("%", "").strip())
    except Exception:
        return 0.0


def momentum_pct(row):
    if "Momentum 12-2" in row.index and pd.notna(row["Momentum 12-2"]):
        raw = str(row["Momentum 12-2"]).strip()
        val = to_num(raw)
        return round(val if "%" in raw else val * 100.0, 2)
    raw = pick(row, ["Momentum %", "Momentum Return %", "Momentum", "Momentum Return"])
    if raw == "":
        return 0.0
    text = str(raw).strip()
    val = to_num(text)
    return round(val, 2)


def load_sector_map():
    if not SECTOR_FILE.exists():
        return {}
    try:
        sec = pd.read_csv(SECTOR_FILE)
        out = {}
        for _, r in sec.iterrows():
            sym = str(pick(r, ["Stock", "Symbol", "Ticker"])).strip().upper()
            sector = str(pick(r, ["Sector"])).strip()
            if sym and sector:
                out[sym] = sector
        return out
    except Exception:
        return {}


def main():
    if not TOP50_FILE.exists():
        raise FileNotFoundError(f"{TOP50_FILE} not found")

    df = pd.read_csv(TOP50_FILE)
    if df.empty:
        raise RuntimeError("live_plan2_top50.csv is empty")

    # Sort strictly by Plan 2 final rank (ID Rank)
    if "ID Rank" in df.columns:
        df = df.sort_values("ID Rank", ascending=True).reset_index(drop=True)

    sector_map = load_sector_map()
    sector_count = {}
    rows = []
    total_invested = 0.0

    for idx, r in df.iterrows():
        if len(rows) >= TARGET_STOCKS:
            break

        stock = str(pick(r, ["Stock", "Symbol", "Ticker"])).strip()
        price = to_num(pick(r, ["Live Price", "Price", "Current Price"]))
        if not stock or price <= 0:
            continue

        # Only stocks whose share price fits within MAX_PER_STOCK (₹1,500 for ₹20k capital)
        if price > MAX_PER_STOCK:
            continue

        sector = sector_map.get(stock.upper(), str(pick(r, ["Sector"]) or "Other").strip())
        if sector_count.get(sector, 0) >= MAX_PER_SECTOR:
            continue

        remaining_capital = CAPITAL - total_invested
        if remaining_capital < price:
            continue

        shares = math.floor(min(MAX_PER_STOCK, remaining_capital) / price)
        if shares < 1:
            continue

        invested = round(shares * price, 2)
        if invested <= 0 or invested > MAX_PER_STOCK + 1e-9:
            continue
        if total_invested + invested > CAPITAL + 1e-9:
            continue

        total_invested = round(total_invested + invested, 2)
        sector_count[sector] = sector_count.get(sector, 0) + 1

        rank_val = pick(r, ["ID Rank", "Rank", "Portfolio Rank", "Momentum Rank"])
        rank_int = int(to_num(rank_val)) if rank_val != "" else (idx + 1)

        rows.append({
            "Rank": rank_int,
            "Stock": stock,
            "Sector": sector,
            "Live Price": round(price, 2),
            "Momentum %": momentum_pct(r),
            "ID": round(to_num(pick(r, ["ID", "Id", "ID Score"])), 6),
            "Signal": pick(r, ["Portfolio Signal", "Signal"]),
            "Target Amount": round(MAX_PER_STOCK, 2),
            "Shares": int(shares),
            "Invested Amount": invested,
        })

    out = pd.DataFrame(rows)
    cash = round(CAPITAL - total_invested, 2)

    if not out.empty:
        out["Weight %"] = (out["Invested Amount"] / CAPITAL * 100.0).round(2)
        out["Cash After Allocation"] = cash
    else:
        out = pd.DataFrame(columns=[
            "Rank", "Stock", "Sector", "Live Price", "Momentum %",
            "ID", "Signal", "Target Amount", "Shares",
            "Invested Amount", "Weight %", "Cash After Allocation",
        ])

    # Safety checks
    if len(out) > TARGET_STOCKS:
        raise RuntimeError("More than TARGET_STOCKS stocks selected")
    if not out.empty:
        if (out["Invested Amount"] > MAX_PER_STOCK + 1e-9).any():
            raise RuntimeError("A stock exceeded MAX_PER_STOCK")
        if out["Invested Amount"].sum() > CAPITAL + 1e-9:
            raise RuntimeError("Allocation exceeded CAPITAL")

    out.to_csv(OUTPUT_FILE, index=False)

    summary = pd.DataFrame([{
        "Capital": CAPITAL,
        "Max Per Stock": MAX_PER_STOCK,
        "Max Per Sector": MAX_PER_SECTOR,
        "Target Stock Count": TARGET_STOCKS,
        "Executable Stock Count": len(out),
        "Invested": total_invested,
        "Cash Remaining": cash,
        "Status": "PASS" if len(out) >= min(TARGET_STOCKS, 5) else "LOW_COUNT",
    }])
    summary.to_csv(SUMMARY_FILE, index=False)

    print("Plan 2 practical allocation")
    print(f"Capital: ₹{CAPITAL:,.2f}")
    print(f"Max per stock: ₹{MAX_PER_STOCK:,.2f}")
    print(f"Max per sector: {MAX_PER_SECTOR}")
    print(f"Target stocks: {TARGET_STOCKS}")
    print(f"Selected stocks: {len(out)}")
    print(f"Invested: ₹{total_invested:,.2f}")
    print(f"Cash: ₹{cash:,.2f}")
    print(f"Output: {OUTPUT_FILE}, {SUMMARY_FILE}")
    if not out.empty:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
