"""Build clean historical Nifty 500 membership records from the existing snapshot CSV."""
import pandas as pd
from pathlib import Path

INPUT = Path("nifty500_2019-01-01_to_2026-09-01.csv")
OUTPUT = Path("nifty500_membership_timeline_clean.csv")

df = pd.read_csv(INPUT)
df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
df = df.dropna(subset=["effective_date"])

records = []
for _, row in df.sort_values("effective_date").iterrows():
    symbols = str(row.get("symbols", "")).split(",")
    for symbol in {s.strip().upper() for s in symbols if s and s.strip()}:
        records.append({"effective_date": row["effective_date"].strftime("%Y-%m-%d"), "symbol": symbol})

out = pd.DataFrame(records).drop_duplicates().sort_values(["effective_date", "symbol"])
out.to_csv(OUTPUT, index=False)
print("Created:", OUTPUT)
print("Membership records:", len(out))
print("Unique symbols:", out["symbol"].nunique())
print("Snapshots:", out["effective_date"].nunique())
