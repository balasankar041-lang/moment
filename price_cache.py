import os
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import yfinance as yf

START_DATE = "2019-01-01"
CACHE_DIR = "data_cache"
CLOSE_FILE = os.path.join(CACHE_DIR, "close.pkl")
VOLUME_FILE = os.path.join(CACHE_DIR, "volume.pkl")

BATCH_SIZE = 10
RETRIES = 5
BATCH_DELAY = 3
MIN_COVERAGE = 0.80


def download_batches(symbols, start, end):
    close_parts = []
    volume_parts = []
    failed = []

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        tickers = [s + ".NS" for s in batch]

        frame = None
        for attempt in range(RETRIES):
            try:
                frame = yf.download(
                    tickers,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    group_by="column",
                )
                if frame is not None and not frame.empty:
                    break
            except Exception as exc:
                print(f"Batch {i//BATCH_SIZE + 1}: attempt {attempt + 1} failed: {exc}")

            if attempt < RETRIES - 1:
                time.sleep(min(60, 5 * (2 ** attempt)))

        if frame is None or frame.empty:
            failed.extend(batch)
            print(f"FAILED batch: {batch}")
            continue

        try:
            close = frame["Close"]
            volume = frame["Volume"]
        except Exception:
            failed.extend(batch)
            print(f"FAILED batch (missing Close/Volume): {batch}")
            continue

        if isinstance(close, pd.Series):
            close = close.to_frame()
        if isinstance(volume, pd.Series):
            volume = volume.to_frame()

        close.columns = [
            c.replace(".NS", "") if isinstance(c, str) else c
            for c in close.columns
        ]
        volume.columns = [
            c.replace(".NS", "") if isinstance(c, str) else c
            for c in volume.columns
        ]

        close_parts.append(close)
        volume_parts.append(volume)

        print(f"Downloaded {min(i + BATCH_SIZE, len(symbols))}/{len(symbols)}")
        time.sleep(BATCH_DELAY)

    if not close_parts:
        raise RuntimeError("No Yahoo data was downloaded.")

    close = pd.concat(close_parts, axis=1)
    volume = pd.concat(volume_parts, axis=1)

    close = close.loc[:, ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]

    return close.sort_index(), volume.sort_index(), failed


def load_cache():
    if not (os.path.exists(CLOSE_FILE) and os.path.exists(VOLUME_FILE)):
        return None, None

    close = pd.read_pickle(CLOSE_FILE)
    volume = pd.read_pickle(VOLUME_FILE)

    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)

    return close.sort_index(), volume.sort_index()


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    membership = pd.read_csv("nifty500_membership_timeline.csv")
    membership["symbol"] = membership["symbol"].astype(str).str.strip()
    symbols = sorted(membership["symbol"].dropna().unique())

    print("Historical symbols:", len(symbols))

    old_close, old_volume = load_cache()

    today = datetime.now(timezone.utc).date()
    end_date = (today + timedelta(days=1)).isoformat()

    if old_close is None or old_volume is None or old_close.empty:
        print("CACHE: no existing cache -> initial full download")
        download_start = START_DATE
    else:
        last_date = max(old_close.index.max(), old_volume.index.max()).date()
        # Small overlap protects against missing/late rows around the last date.
        download_start = (last_date - timedelta(days=7)).isoformat()
        print(f"CACHE: existing data through {last_date}")
        print(f"CACHE: incremental update from {download_start}")

    new_close, new_volume, failed = download_batches(
        symbols,
        download_start,
        end_date,
    )

    if old_close is not None:
        close = pd.concat([old_close, new_close], axis=0)
        volume = pd.concat([old_volume, new_volume], axis=0)
        close = close[~close.index.duplicated(keep="last")].sort_index()
        volume = volume[~volume.index.duplicated(keep="last")].sort_index()
    else:
        close, volume = new_close, new_volume

    # Keep the requested universe and require broad coverage.
    close = close.reindex(columns=symbols)
    volume = volume.reindex(columns=symbols)

    coverage = close.notna().any(axis=0).mean()
    print(f"Close coverage: {coverage * 100:.1f}%")

    if coverage < MIN_COVERAGE:
        raise RuntimeError(
            f"Coverage only {coverage * 100:.1f}%; cache was NOT written."
        )

    close.to_pickle(CLOSE_FILE)
    volume.to_pickle(VOLUME_FILE)

    print("CACHE UPDATED")
    print("Close rows:", len(close))
    print("Close last date:", close.index.max().date())
    print("Failed batches/symbols:", len(failed))

    if failed:
        print("Failed symbols:", ", ".join(failed))


if __name__ == "__main__":
    main()
