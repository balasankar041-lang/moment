"""
Dynamic Zerodha sector classifier for the Plan 2 Top 50.

Source:
  Zerodha Markets sector pages.

Design:
  1. Fetch all Zerodha sector pages.
  2. Extract NSE stock symbols from Zerodha stock links.
  3. Build a symbol -> sector master.
  4. Classify the current Plan 2 Top 50.
  5. Preserve Plan 2 order/ranking.
  6. NEVER silently convert an unmatched Nifty 500/Top 50 stock to "Other".
     Unmatched symbols are written to sector_mapping_missing.csv and the
     program exits non-zero so the workflow cannot publish an incomplete
     sector table.

The sector list is the current 35-sector list exposed by Zerodha.
"""

from pathlib import Path
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

TOP50_FILE = Path("live_plan2_top50.csv")
UNIVERSE_FILE = Path("universe.csv")
OUTPUT_FILE = Path("sector_wise_top50.csv")
MASTER_FILE = Path("zerodha_sector_master.csv")
MISSING_FILE = Path("sector_mapping_missing.csv")

BASE = "https://zerodha.com/markets/sector/"

SECTORS = [
    ("Agriculture", "agriculture"),
    ("Auto ancillary", "auto-ancillary"),
    ("Automobile", "automobile"),
    ("Aviation", "aviation"),
    ("Building materials", "building-materials"),
    ("Chemicals", "chemicals"),
    ("Consumer durables", "consumer-durables"),
    ("Dairy products", "dairy-products"),
    ("Defence", "defence"),
    ("Diversified", "diversified"),
    ("Education & training", "education-training"),
    ("Energy", "energy"),
    ("Engineering & capital goods", "engineering-capital-goods"),
    ("FMCG", "fmcg"),
    ("Fertilizers", "fertilizers"),
    ("Financial services", "financial-services"),
    ("Healthcare", "healthcare"),
    ("IT", "it"),
    ("Logistics", "logistics"),
    ("Media & entertainment", "media-entertainment"),
    ("Metals", "metals"),
    ("Miscellaneous", "miscellaneous"),
    ("NBFC", "nbfc"),
    ("Packaging", "packaging"),
    ("Plastic pipes", "plastic-pipes"),
    ("Real estate", "real-estate"),
    ("Retail", "retail"),
    ("Services", "services"),
    ("Silver", "silver"),
    ("Software services", "software-services"),
    ("Solar panel", "solar-panel"),
    ("Telecom", "telecom"),
    ("Textiles", "textiles"),
    ("Tourism & hospitality", "tourism-hospitality"),
    ("Trading", "trading"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Plan2SectorClassifier/1.0)"
}


def clean_symbol(value):
    s = str(value).strip().upper()
    s = re.sub(r"\.NS$", "", s)
    return s


def fetch_sector(session, sector_name, slug):
    url = f"{BASE}{slug}/"
    last_error = None

    for attempt in range(3):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()

            soup = BeautifulSoup(r.text, "html.parser")
            symbols = set()

            # Zerodha stock pages use /markets/stocks/NSE/<SYMBOL>/
            for a in soup.find_all("a", href=True):
                href = a["href"]
                m = re.search(
                    r"/markets/stocks/(?:NSE|BSE)/([A-Za-z0-9&._-]+)/?",
                    href,
                    flags=re.I,
                )
                if m and "/NSE/" in href.upper():
                    symbols.add(clean_symbol(m.group(1)))

            if not symbols:
                raise RuntimeError(
                    f"No NSE symbols extracted from {url}"
                )

            return [(symbol, sector_name) for symbol in sorted(symbols)]

        except Exception as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"{sector_name}: {last_error}")


def load_top50():
    if not TOP50_FILE.exists():
        raise FileNotFoundError(TOP50_FILE)

    df = pd.read_csv(TOP50_FILE)
    if df.empty:
        raise RuntimeError("live_plan2_top50.csv is empty")

    symbol_col = next(
        (c for c in ["Stock", "Symbol", "Ticker"] if c in df.columns),
        None,
    )
    if not symbol_col:
        raise RuntimeError("No Stock/Symbol/Ticker column in Top 50")

    df["_symbol"] = df[symbol_col].map(clean_symbol)
    return df, symbol_col


def load_nifty500_symbols():
    if not UNIVERSE_FILE.exists():
        return set()

    u = pd.read_csv(UNIVERSE_FILE)
    col = next(
        (c for c in ["Symbol", "Stock", "Ticker"] if c in u.columns),
        None,
    )
    if not col:
        return set()

    return set(u[col].map(clean_symbol))


def build_master():
    session = requests.Session()
    all_rows = []

    print(f"Fetching {len(SECTORS)} Zerodha sector pages...")

    for i, (sector_name, slug) in enumerate(SECTORS, 1):
        print(f"[{i:02d}/{len(SECTORS)}] {sector_name}")
        rows = fetch_sector(session, sector_name, slug)
        all_rows.extend(rows)
        time.sleep(0.25)

    master = pd.DataFrame(all_rows, columns=["Symbol", "Sector"])

    # A company can appear in more than one Zerodha sector page.
    # Keep the first occurrence but report duplicates for audit.
    dup = (
        master.groupby("Symbol")["Sector"]
        .agg(lambda x: sorted(set(x)))
        .reset_index()
    )
    dup["Sector Count"] = dup["Sector"].map(len)

    conflicts = dup[dup["Sector Count"] > 1].copy()
    if not conflicts.empty:
        conflicts.to_csv("zerodha_sector_conflicts.csv", index=False)
        print(
            f"WARNING: {len(conflicts)} symbols appear in multiple Zerodha sectors. "
            "First sector occurrence is used."
        )

    master = master.drop_duplicates("Symbol", keep="first")
    master.to_csv(MASTER_FILE, index=False)

    print(f"Zerodha sector master: {len(master)} NSE symbols")
    return master


def classify():
    top, symbol_col = load_top50()
    universe = load_nifty500_symbols()
    master = build_master()

    mapping = dict(zip(master["Symbol"], master["Sector"]))

    top["Sector"] = top["_symbol"].map(mapping)

    missing = top[top["Sector"].isna()].copy()
    if not missing.empty:
        missing[["Stock", "_symbol"]].rename(
            columns={"_symbol": "Symbol"}
        ).to_csv(MISSING_FILE, index=False)

        print("\nUNMATCHED TOP 50 STOCKS:")
        print(
            missing[["Stock", "_symbol"]].to_string(index=False)
        )

        raise RuntimeError(
            f"{len(missing)} Top 50 stocks have no Zerodha sector mapping. "
            f"See {MISSING_FILE}. No incomplete sector file published."
        )

    # Audit against the Nifty 500 universe when available.
    if universe:
        unexpected = sorted(
            set(top["_symbol"]) - universe
        )
        if unexpected:
            print(
                "WARNING: Top 50 contains symbols not present in universe.csv:",
                unexpected,
            )

    # Preserve exact Plan 2 order.
    top["Sector Rank"] = (
        top.groupby("Sector", sort=False).cumcount() + 1
    )
    top["Sector Stock Count"] = top.groupby("Sector")["Sector"].transform("size")

    top = top.drop(columns=["_symbol"])

    # Keep existing columns/order and add sector fields.
    cols = list(top.columns)
    sector_cols = ["Sector", "Sector Rank", "Sector Stock Count"]
    cols = [c for c in cols if c not in sector_cols] + sector_cols
    top = top[cols]

    if len(top) != len(pd.read_csv(TOP50_FILE)):
        raise RuntimeError("Sector classification changed Top 50 row count")

    top.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {OUTPUT_FILE}: {len(top)} stocks")
    print("\nSector counts:")
    print(top["Sector"].value_counts().to_string())
    print("\nNo Top 50 stock is classified as Other.")


if __name__ == "__main__":
    classify()
