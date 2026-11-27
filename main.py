import time
import warnings
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    from price_cache import load_cache
except Exception:
    load_cache = None

warnings.filterwarnings("ignore")

print("NIFTY 500 PLAN 2 LIVE SCREENER")
print("==============================")

URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}

TOP_MOMENTUM = 100
PORTFOLIO_SIZE = 50
HISTORY_FILE = Path("signal_history.csv")
IPO_LIST_FILE = Path("ipo_watchlist.csv")
IPO_AUTO_FILE = Path("ipo_auto_discovered.csv")
IPO_WATCH_FILE = Path("ipo_watch.csv")

MIN_HISTORY_ROWS = 253
MIN_PATH_ROWS = 100
MAX_MISSING_VALID_RATIO = 0.20


def load_nifty500_universe():
    response = requests.get(URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    universe = pd.read_csv(BytesIO(response.content))

    if "Symbol" not in universe.columns:
        raise ValueError(f"Unexpected Nifty file columns: {list(universe.columns)}")

    symbols = (
        universe["Symbol"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    )
    return symbols


def fetch_close_series(symbols):
    """Use fresh local price cache when available, downloading missing symbols from Yahoo."""
    close_map = {}

    if load_cache is not None:
        cached_close, _ = load_cache(with_ns_suffix=False)
        if cached_close is not None and not cached_close.empty:
            last_dt = cached_close.index.max().date()
            age_days = (datetime.now(timezone.utc).date() - last_dt).days
            if age_days <= 4:
                print(f"Using fresh price cache through {last_dt} (age: {age_days}d)")
                for sym in symbols:
                    if sym in cached_close.columns:
                        s = cached_close[sym].dropna()
                        if len(s) >= MIN_HISTORY_ROWS:
                            close_map[sym] = s

    missing = [s for s in symbols if s not in close_map]
    print(f"Cached valid symbols: {len(close_map)} | Downloading from Yahoo: {len(missing)}")

    batch_size = 25
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]
        tickers = [s + ".NS" for s in batch]
        print(f"[{i + 1}-{min(i + batch_size, len(missing))}/{len(missing)}] Downloading batch...")

        frame = None
        for attempt in range(3):
            try:
                frame = yf.download(
                    tickers,
                    period="2y",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    group_by="column",
                )
                if frame is not None and not frame.empty:
                    break
            except Exception as e:
                print(f"  Batch attempt {attempt + 1} failed: {e}")
                time.sleep(2 * (attempt + 1))

        if frame is None or frame.empty:
            continue

        try:
            if isinstance(frame.columns, pd.MultiIndex):
                close_df = frame["Close"]
            else:
                close_df = frame[["Close"]].copy()
                close_df.columns = [batch[0]]

            if isinstance(close_df, pd.Series):
                close_df = close_df.to_frame(name=batch[0])

            for col in close_df.columns:
                sym = str(col).replace(".NS", "").strip().upper()
                s = close_df[col].dropna()
                if not s.empty:
                    close_map[sym] = s
        except Exception as e:
            print(f"  Batch parse error: {e}")

        time.sleep(1)

    return close_map


def discover_recent_nse_symbols():
    """Best-effort public NSE listing discovery; failure leaves the screener safe."""
    out = []
    try:
        url = "https://www.nseindia.com/api/public-reports"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/",
        }
        s = requests.Session()
        s.get("https://www.nseindia.com/", headers=headers, timeout=10)
        r = s.get(url, headers=headers, timeout=10)
        if r.ok and "json" in r.headers.get("content-type", "").lower():
            data = r.json()
            items = data if isinstance(data, list) else data.get("data", [])
            for x in items:
                if not isinstance(x, dict):
                    continue
                sym = str(
                    x.get("symbol") or x.get("Symbol") or x.get("SYMBOL") or ""
                ).strip().upper()
                if sym and sym.isalnum():
                    out.append(sym)
    except Exception:
        pass
    return sorted(set(out))


def run_ipo_watch():
    ipo_symbols = []
    if IPO_LIST_FILE.exists():
        try:
            ipo_input = pd.read_csv(IPO_LIST_FILE)
            if "Symbol" in ipo_input.columns:
                ipo_symbols = (
                    ipo_input["Symbol"].dropna().astype(str).str.strip().str.upper()
                    .str.replace(r"\.NS$", "", regex=True).unique().tolist()
                )
        except Exception as e:
            print(f"Warning: could not read IPO watchlist: {e}")
    else:
        pd.DataFrame(columns=["Symbol"]).to_csv(IPO_LIST_FILE, index=False)

    discovered = discover_recent_nse_symbols()
    pd.DataFrame({"Symbol": discovered}).to_csv(IPO_AUTO_FILE, index=False)

    ipo_cols = ["Symbol", "Status", "Live Price", "Trading Days", "Plan 2 Eligible"]
    ipo_rows = []
    for ipo_symbol in ipo_symbols:
        try:
            ipo_data = yf.download(
                ipo_symbol + ".NS",
                period="2y",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if ipo_data.empty:
                ipo_rows.append({
                    "Symbol": ipo_symbol,
                    "Status": "IPO WATCH - NO DATA",
                    "Live Price": np.nan,
                    "Trading Days": 0,
                    "Plan 2 Eligible": "NO",
                })
                continue

            ipo_close = ipo_data["Close"].squeeze().dropna()
            days = len(ipo_close)
            if days < 20:
                status, eligible = "IPO WATCH - BUILDING HISTORY", "NO"
            elif days < MIN_HISTORY_ROWS:
                status, eligible = "IPO WATCH - NOT ENOUGH HISTORY", "NO"
            else:
                status, eligible = "PLAN 2 ELIGIBLE HISTORY", "YES"

            ipo_rows.append({
                "Symbol": ipo_symbol,
                "Status": status,
                "Live Price": float(ipo_close.iloc[-1]),
                "Trading Days": days,
                "Plan 2 Eligible": eligible,
            })
        except Exception:
            ipo_rows.append({
                "Symbol": ipo_symbol,
                "Status": "IPO WATCH - DATA ERROR",
                "Live Price": np.nan,
                "Trading Days": 0,
                "Plan 2 Eligible": "NO",
            })

    pd.DataFrame(ipo_rows, columns=ipo_cols).to_csv(IPO_WATCH_FILE, index=False)


def main():
    symbols = load_nifty500_universe()
    print(f"Nifty 500 universe loaded: {len(symbols)} stocks")

    close_map = fetch_close_series(symbols)

    results = []
    for sym in symbols:
        close = close_map.get(sym)
        if close is None or len(close) < MIN_HISTORY_ROWS:
            continue

        start_cut = close.index[-1] - pd.DateOffset(months=12)
        end_cut = close.index[-1] - pd.DateOffset(months=2)

        a = close.loc[close.index <= start_cut]
        b = close.loc[close.index <= end_cut]
        if a.empty or b.empty:
            continue

        p0, p1 = float(a.iloc[-1]), float(b.iloc[-1])
        if p0 <= 0 or p1 <= 0:
            continue

        momentum = p1 / p0 - 1.0

        path = close.loc[
            (close.index >= a.index[-1]) & (close.index <= b.index[-1])
        ]
        if len(path) < MIN_PATH_ROWS:
            continue

        daily = path.pct_change().dropna()
        daily = daily[daily != 0]
        if daily.empty:
            continue

        positive_days = float((daily > 0).mean())
        negative_days = float((daily < 0).mean())

        id_score = (1.0 if momentum > 0 else -1.0) * (negative_days - positive_days)

        results.append({
            "Stock": sym,
            "Live Price": float(close.iloc[-1]),
            "Momentum 12-2": float(momentum),
            "Positive Days %": positive_days,
            "Negative Days %": negative_days,
            "ID": float(id_score),
        })

    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError("No stocks produced valid results.")

    valid_ratio = len(df) / len(symbols)
    print(f"Valid-data coverage: {valid_ratio:.1%}")
    if valid_ratio < (1.0 - MAX_MISSING_VALID_RATIO):
        raise RuntimeError(
            f"DATA SAFETY STOP: valid coverage {valid_ratio:.1%} is below 80%. "
            "No BUY/SELL signals are saved."
        )

    print("\nPRICE / SIGNAL DATA")
    print("===================")
    print(f"Valid stocks: {len(df)}")

    run_ipo_watch()

    # Stage 1: Top 100 momentum
    df = df.sort_values(["Momentum 12-2", "Stock"], ascending=[False, True]).reset_index(drop=True)
    df["Momentum Rank"] = df.index + 1
    top100 = df.head(TOP_MOMENTUM).copy()

    # Stage 2: Lowest ID within Top 100
    top100 = top100.sort_values(["ID", "Stock"], ascending=[True, True]).reset_index(drop=True)
    top100["ID Rank"] = top100.index + 1

    # Propagate ID Rank back to full ranking dataframe
    id_rank_map = top100.set_index("Stock")["ID Rank"].to_dict()
    df["ID Rank"] = df["Stock"].map(id_rank_map)

    # Final Top 50
    top50 = top100.head(PORTFOLIO_SIZE).copy()
    if len(top50) < PORTFOLIO_SIZE:
        raise RuntimeError(
            f"Only {len(top50)} final stocks available; need {PORTFOLIO_SIZE}."
        )

    current_top50 = set(top50["Stock"])
    current_top100 = set(top100["Stock"])

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ---------------------------------------------------------
    # PREVIOUS SIGNAL STATE (strictly prior to today's run_date)
    # ---------------------------------------------------------
    previous = pd.DataFrame()
    if HISTORY_FILE.exists():
        try:
            history = pd.read_csv(HISTORY_FILE)
            if not history.empty and "Run Date" in history.columns:
                prior_history = history[history["Run Date"].astype(str) < run_date]
                if not prior_history.empty:
                    latest_date = prior_history["Run Date"].astype(str).max()
                    previous = prior_history[prior_history["Run Date"].astype(str) == latest_date].copy()
        except Exception as e:
            print(f"Warning: could not read signal history: {e}")

    previous_top50 = set()
    if not previous.empty and "Portfolio Signal" in previous.columns:
        previous_top50 = set(
            previous.loc[
                previous["Portfolio Signal"].isin(["BUY", "HOLD"]),
                "Stock",
            ].astype(str)
        )

    # ---------------------------------------------------------
    # SIGNALS (No duplicate rows; preserve Sell Reason & Exit Safety)
    # ---------------------------------------------------------
    df["Portfolio Signal"] = "WAIT"
    df["Sell Reason"] = ""
    df["Buy Reason"] = ""
    df["Exit Safety"] = ""
    df["Data Safety"] = "VALID"

    df.loc[df["Stock"].isin(current_top100), "Portfolio Signal"] = "ID FILTER"

    hold_mask = df["Stock"].isin(current_top50 & previous_top50)
    buy_mask = df["Stock"].isin(current_top50 - previous_top50)

    df.loc[hold_mask, "Portfolio Signal"] = "HOLD"
    df.loc[hold_mask, "Buy Reason"] = df.loc[hold_mask].apply(
        lambda r: f"Remains in Plan 2 Top 50 (Momentum Rank {int(r['Momentum Rank'])}, ID Rank {int(r['ID Rank'])})",
        axis=1,
    )

    df.loc[buy_mask, "Portfolio Signal"] = "BUY"
    df.loc[buy_mask, "Buy Reason"] = df.loc[buy_mask].apply(
        lambda r: f"New entry into Plan 2 Top 50 (Momentum Rank {int(r['Momentum Rank'])}, ID Rank {int(r['ID Rank'])})",
        axis=1,
    )

    # SELL rule: update in-place for stocks present in today's valid data
    # that dropped out of Top 50 (never force SELL on missing data; never duplicate rows).
    sell_candidates = (previous_top50 - current_top50) & set(df["Stock"])
    if sell_candidates:
        sell_mask = df["Stock"].isin(sell_candidates)
        df.loc[sell_mask, "Portfolio Signal"] = "SELL"
        df.loc[sell_mask, "Sell Reason"] = df.loc[sell_mask, "Stock"].map(
            lambda s: "SELL: dropped below Top 50; still in Top 100"
            if s in current_top100
            else "SELL: dropped out of Top 100"
        )
        df.loc[sell_mask, "Exit Safety"] = "Validated current data; no forced sell on missing data"

    # Weights
    df["Weight %"] = 0.0
    df.loc[df["Stock"].isin(current_top50), "Weight %"] = 100.0 / len(top50)

    columns = [
        "Momentum Rank",
        "ID Rank",
        "Stock",
        "Live Price",
        "Momentum 12-2",
        "Positive Days %",
        "Negative Days %",
        "ID",
        "Portfolio Signal",
        "Sell Reason",
        "Buy Reason",
        "Exit Safety",
        "Data Safety",
        "Weight %",
    ]

    # Sync top100 and top50 directly from df so signals/reasons/weights always match
    top100_output = (
        df[df["Stock"].isin(current_top100)]
        .sort_values("ID Rank", ascending=True)
        .reset_index(drop=True)[columns]
    )
    top50_output = (
        df[df["Stock"].isin(current_top50)]
        .sort_values("ID Rank", ascending=True)
        .reset_index(drop=True)[columns]
    )
    sell_output = (
        df[df["Portfolio Signal"] == "SELL"]
        .sort_values("Momentum Rank", ascending=True)
        .reset_index(drop=True)[columns]
    )

    # ---------------------------------------------------------
    # SAVE CURRENT RESULTS
    # ---------------------------------------------------------
    df[columns].to_csv("live_plan2_ranking.csv", index=False)
    top100_output.to_csv("live_plan2_top100.csv", index=False)
    top50_output.to_csv("live_plan2_top50.csv", index=False)
    df[columns].to_csv("ranking.csv", index=False)

    # ---------------------------------------------------------
    # APPEND SNAPSHOT (TOP 50 + SELL EXITS) TO SIGNAL HISTORY
    # ---------------------------------------------------------
    snapshot = pd.concat([top50_output, sell_output], ignore_index=True)
    snapshot.insert(0, "Run Date", run_date)

    if HISTORY_FILE.exists():
        try:
            old_history = pd.read_csv(HISTORY_FILE)
            old_history = old_history[old_history["Run Date"].astype(str) != run_date]
        except Exception:
            old_history = pd.DataFrame()
    else:
        old_history = pd.DataFrame()

    new_history = pd.concat([old_history, snapshot], ignore_index=True)
    new_history.to_csv(HISTORY_FILE, index=False)

    # ---------------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------------
    print("\n==========================================")
    print("PLAN 2 LIVE RESULT")
    print("==========================================")
    print(f"Universe: {len(symbols)}")
    print(f"Valid stocks: {len(results)}")
    print(f"Top momentum: {TOP_MOMENTUM}")
    print(f"Final portfolio: {len(top50_output)}")
    print(f"Previous Top 50: {len(previous_top50)}")
    print(f"BUY: {int((top50_output['Portfolio Signal'] == 'BUY').sum())}")
    print(f"HOLD: {int((top50_output['Portfolio Signal'] == 'HOLD').sum())}")
    print(f"SELL: {int((df['Portfolio Signal'] == 'SELL').sum())}")
    print("HOLD -> SELL safety: only valid current-data exits; missing data is not a SELL")

    display_df = top50_output[["Momentum Rank", "ID Rank", "Stock", "Live Price", "Momentum 12-2", "ID", "Portfolio Signal", "Weight %"]].copy()
    display_df["Live Price"] = display_df["Live Price"].map(lambda x: f"{x:.2f}")
    display_df["Momentum 12-2"] = display_df["Momentum 12-2"].map(lambda x: f"{x:.2%}")
    display_df["ID"] = display_df["ID"].map(lambda x: f"{x:.4f}")
    display_df["Weight %"] = display_df["Weight %"].map(lambda x: f"{x:.2f}%")

    print("\nTOP 50 PLAN 2")
    print("------------------------------------------")
    print(display_df.to_string(index=False))

    print("\nFiles saved:")
    for fn in [
        "ranking.csv",
        "live_plan2_ranking.csv",
        "live_plan2_top100.csv",
        "live_plan2_top50.csv",
        "signal_history.csv",
        "ipo_watch.csv",
        "ipo_auto_discovered.csv",
    ]:
        print(" ", fn)
    print("\nSTATUS: SUCCESS")


if __name__ == "__main__":
    main()
