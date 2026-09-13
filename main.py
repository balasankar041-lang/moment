from pathlib import Path
import pandas as pd
from screener import score_universe

DATA = Path("data")
DATA.mkdir(exist_ok=True)

prices_file = DATA / "prices.csv"
universe_file = DATA / "universe.csv"

if not prices_file.exists():
    print("Project is ready.")
    print("Next: add price data and the Nifty 500 universe.")
    raise SystemExit(0)

prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)
universe = pd.read_csv(universe_file)["symbol"].dropna().tolist() if universe_file.exists() else None
result = score_universe(prices, universe)
print(result.head(15).to_string(index=False))
