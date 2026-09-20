from pathlib import Path
import pandas as pd

INPUT_FILE = Path("live_plan2_top50.csv")
OUTPUT_FILE = Path("sector_wise_top50.csv")

SECTOR_MAP = {
    "FEDERALBNK":"Financials","SHRIRAMFIN":"Financials","BANKINDIA":"Financials",
    "INDUSINDBK":"Financials","SBIN":"Financials","UNIONBANK":"Financials",
    "HDFCBANK":"Financials","ICICIBANK":"Financials","AXISBANK":"Financials",
    "KOTAKBANK":"Financials","PNB":"Financials","CANBK":"Financials",
    "BANKBARODA":"Financials","IDFCFIRSTB":"Financials","AUBANK":"Financials",
    "MUTHOOTFIN":"Financials","MANAPPURAM":"Financials","CHOLAFIN":"Financials",
    "BAJFINANCE":"Financials","BAJAJFINSV":"Financials",
    "NATIONALUM":"Metals & Mining","HINDALCO":"Metals & Mining","TATASTEEL":"Metals & Mining",
    "JSWSTEEL":"Metals & Mining","VEDL":"Metals & Mining","HINDZINC":"Metals & Mining",
    "JINDALSTEL":"Metals & Mining","SAIL":"Metals & Mining","NMDC":"Metals & Mining",
    "BHEL":"Capital Goods","BEL":"Capital Goods","HAL":"Capital Goods","SIEMENS":"Capital Goods",
    "ABB":"Capital Goods","THERMAX":"Capital Goods","CGPOWER":"Capital Goods",
    "POLYCAB":"Capital Goods","POWERINDIA":"Capital Goods",
    "ADANIENSOL":"Power & Utilities","NTPC":"Power & Utilities","POWERGRID":"Power & Utilities",
    "TATAPOWER":"Power & Utilities","TORNTPOWER":"Power & Utilities","JSWENERGY":"Power & Utilities",
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma","DIVISLAB":"Pharma",
    "LUPIN":"Pharma","AUROPHARMA":"Pharma","TORNTPHARM":"Pharma","GRANULES":"Pharma",
    "ZYDUSLIFE":"Pharma","MANKIND":"Pharma",
    "ASTERDM":"Healthcare","APOLLOHOSP":"Healthcare","MAXHEALTH":"Healthcare",
    "FORTIS":"Healthcare","KIMS":"Healthcare","MEDANTA":"Healthcare",
    "HSCL":"Chemicals","PIDILITIND":"Chemicals","SRF":"Chemicals","DEEPAKNTR":"Chemicals",
    "NAVINFLUOR":"Chemicals","TATACHEM":"Chemicals","PIIND":"Chemicals",
    "TCS":"IT","INFY":"IT","HCLTECH":"IT","WIPRO":"IT","TECHM":"IT","LTIM":"IT",
    "PERSISTENT":"IT","COFORGE":"IT","MPHASIS":"IT",
    "MARUTI":"Auto","TATAMOTORS":"Auto","M&M":"Auto","BAJAJ-AUTO":"Auto",
    "EICHERMOT":"Auto","HEROMOTOCO":"Auto","TVSMOTOR":"Auto","ASHOKLEY":"Auto",
    "BHARATFORG":"Auto",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "TATACONSUM":"FMCG","DABUR":"FMCG","MARICO":"FMCG","COLPAL":"FMCG",
    "GODREJCP":"FMCG","UNITDSPR":"FMCG",
    "RELIANCE":"Oil & Gas","ONGC":"Oil & Gas","IOC":"Oil & Gas","BPCL":"Oil & Gas",
    "GAIL":"Oil & Gas","HINDPETRO":"Oil & Gas","PETRONET":"Oil & Gas",
    "LT":"Infrastructure","ADANIENT":"Infrastructure","ADANIPORTS":"Infrastructure",
    "RVNL":"Infrastructure","IRCON":"Infrastructure","NBCC":"Infrastructure",
    "DLF":"Realty","PHOENIXLTD":"Realty","LODHA":"Realty","GODREJPROP":"Realty",
    "OBEROIRLTY":"Realty","PRESTIGE":"Realty",
    "BHARTIARTL":"Telecom","INDUSTOWER":"Telecom","IDEA":"Telecom",
    "DIXON":"Consumer Durables","VOLTAS":"Consumer Durables","WHIRLPOOL":"Consumer Durables",
    "HAVELLS":"Consumer Durables","BLUESTARCO":"Consumer Durables",
    "KPRMILL":"Textiles","TRIDENT":"Textiles",
    "MAZDOCK":"Defence","COCHINSHIP":"Defence","BDL":"Defence"
}

def get_symbol(row):
    for col in ["Stock", "Symbol", "Ticker"]:
        if col in row.index:
            value = str(row[col]).strip().upper()
            if value:
                return value
    return ""

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found")
    df = pd.read_csv(INPUT_FILE)
    if df.empty:
        raise RuntimeError("live_plan2_top50.csv is empty")

    df["Symbol_Check"] = df.apply(get_symbol, axis=1)
    df["Sector"] = df["Symbol_Check"].map(SECTOR_MAP).fillna("Other")
    df["Sector Rank"] = df.groupby("Sector").cumcount() + 1
    df["Sector Stock Count"] = df.groupby("Sector")["Sector"].transform("count")

    preferred = [
        "Rank","Stock","Symbol","Ticker","Sector","Sector Rank",
        "Sector Stock Count","Live Price","Momentum %","ID","Signal"
    ]
    columns = [c for c in preferred if c in df.columns]
    remaining = [c for c in df.columns if c not in columns and c != "Symbol_Check"]
    out = df[columns + remaining]

    if len(out) != len(df):
        raise RuntimeError("Sector classification lost stocks")

    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Input stocks : {len(df)}")
    print(f"Output stocks: {len(out)}")
    print(out[[c for c in ["Rank","Stock","Sector","Sector Rank"] if c in out.columns]].to_string(index=False))
    print("\nSector summary")
    print(out["Sector"].value_counts().to_string())

if __name__ == "__main__":
    main()
