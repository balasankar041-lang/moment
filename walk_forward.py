import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"
TOP_100 = 100
TOP_50 = 50
PORTFOLIO_SIZE = 50
TRADING_COST = 0.005

TEST_WINDOWS = [
    ("2023", "2023-01-31", "2023-12-31"),
    ("2024", "2024-01-31", "2024-12-31"),
    ("2025", "2025-01-31", "2025-12-31"),
    ("2026", "2026-01-31", "2026-08-31"),
]

membership = pd.read_csv(MEMBERSHIP_FILE)
membership["effective_date"] = pd.to_datetime(membership["effective_date"])
membership["symbol"] = membership["symbol"].astype(str).str.strip().str.upper()
membership = membership.dropna(subset=["effective_date","symbol"]).sort_values(["effective_date","symbol"])

def yahoo_symbol(s):
    s = str(s).strip().upper()
    return s + ".NS" if s.isalpha() and len(s) <= 20 else None

def members_at_date(date):
    x = membership[membership["effective_date"] <= date]
    if x.empty: return []
    latest = x["effective_date"].max()
    return sorted(set(yahoo_symbol(s) for s in x.loc[x["effective_date"] == latest,"symbol"] if yahoo_symbol(s)))

symbols = sorted(set(yahoo_symbol(s) for s in membership["symbol"] if yahoo_symbol(s)))
download_start = membership["effective_date"].min() - pd.DateOffset(years=2)
today = pd.Timestamp.today().normalize()
download_end = today + pd.Timedelta(days=1)

print("Downloading price data...")
print("Price period:", download_start.date(), "to", today.date())
print("Historical symbols:", len(symbols))

def download_prices(symbols, batch_size=25):
    frames = []
    for i in range(0,len(symbols),batch_size):
        batch = symbols[i:i+batch_size]
        print(f"Downloading {i+1}-{min(i+batch_size,len(symbols))}")
        try:
            data = yf.download(batch,start=download_start.strftime("%Y-%m-%d"),
                               end=download_end.strftime("%Y-%m-%d"),
                               auto_adjust=True,progress=False,threads=True)
            if data.empty: continue
            if isinstance(data.columns,pd.MultiIndex):
                if "Close" not in data.columns.get_level_values(0): continue
                close = data["Close"]
            else:
                if "Close" not in data.columns: continue
                close = data[["Close"]]; close.columns=[batch[0]]
            frames.append(close)
        except Exception:
            continue
    if not frames: return pd.DataFrame()
    return pd.concat(frames,axis=1,sort=True).loc[:,lambda x:~x.columns.duplicated()].sort_index()

prices = download_prices(symbols).dropna(axis=1,how="all")
if prices.empty: raise SystemExit("No price data downloaded.")

last_trading_day = prices.index.max()
if last_trading_day.to_period("M") == today.to_period("M"):
    completed_month = (today.to_period("M")-1).end_time.normalize()
else:
    completed_month = last_trading_day.to_period("M").end_time.normalize()

monthly = prices.resample("ME").last()
monthly = monthly[monthly.index <= completed_month]

print("\nPRICE DATA")
print("==========")
print("Stocks with actual data:",prices.shape[1])
print("Trading days:",len(prices))
print("Last available trading day:",last_trading_day.date())
print("Last completed month:",completed_month.date())

def signal(symbol,date):
    if symbol not in prices.columns: return None
    s=prices[symbol].dropna()
    start_cut=date-pd.DateOffset(months=12)
    end_cut=date-pd.DateOffset(months=2)
    a=s.loc[s.index<=start_cut]
    b=s.loc[s.index<=end_cut]
    if a.empty or b.empty: return None
    p0,p1=a.iloc[-1],b.iloc[-1]
    if p0<=0 or p1<=0: return None
    ret=p1/p0-1
    path=s.loc[(s.index>=a.index[-1])&(s.index<=b.index[-1])]
    if len(path)<100: return None
    dr=path.pct_change().dropna()
    dr=dr[dr!=0]
    if dr.empty: return None
    pos=(dr>0).mean(); neg=(dr<0).mean()
    id_score=(1 if ret>0 else -1)*(neg-pos)
    return ret,id_score

dates=[d for d in monthly.index if d<=completed_month]
dates=[d for d in dates if d>=membership["effective_date"].min()+pd.DateOffset(months=13)]

def build_portfolio(date):
    candidates=[s for s in members_at_date(date) if s in prices.columns]
    signals={}
    for s in candidates:
        z=signal(s,date)
        if z is not None: signals[s]=z
    if len(signals)<TOP_100: return None
    top100=sorted(signals.items(),key=lambda x:x[1][0],reverse=True)[:TOP_100]
    selected=[s for s,z in sorted(top100,key=lambda x:x[1][1])[:TOP_50]]
    if len(selected)<PORTFOLIO_SIZE: return None
    return {s:1.0/PORTFOLIO_SIZE for s in selected[:PORTFOLIO_SIZE]}

rows=[]
previous={}
for i,date in enumerate(dates[:-1]):
    next_date=dates[i+1]
    portfolio=build_portfolio(date)
    if portfolio is None: continue
    all_s=set(previous)|set(portfolio)
    turnover=sum(abs(portfolio.get(s,0)-previous.get(s,0)) for s in all_s)
    equity_cost=1-turnover*TRADING_COST
    period_ret=0.0
    for s,weight in portfolio.items():
        try:
            p0=monthly.loc[date,s]; p1=monthly.loc[next_date,s]
            if pd.notna(p0) and pd.notna(p1) and p0>0:
                period_ret += weight*(p1/p0-1)
        except Exception: pass
    rows.append({"rebalance_date":date,"return_date":next_date,
                 "monthly_return":equity_cost*(1+period_ret)-1})
    previous=portfolio

monthly_returns=pd.DataFrame(rows)
if monthly_returns.empty: raise SystemExit("No monthly returns generated.")

monthly_returns["return_date"]=pd.to_datetime(monthly_returns["return_date"])
monthly_returns=monthly_returns.sort_values("return_date").drop_duplicates("return_date",keep="last")

def metrics(r):
    r=pd.Series(r).dropna()
    if r.empty: return None
    equity=(1+r).cumprod()
    total=equity.iloc[-1]-1
    years=len(r)/12
    cagr=equity.iloc[-1]**(1/years)-1
    vol=r.std(ddof=1)*np.sqrt(12)
    sharpe=r.mean()/r.std(ddof=1)*np.sqrt(12) if r.std(ddof=1)>0 else np.nan
    dd=(equity/equity.cummax()-1).min()
    return total,cagr,vol,sharpe,dd,(r>0).mean(),len(r)

print("\n"+"="*80)
print("PLAN 2 — EXACT WALK-FORWARD VALIDATION")
print("="*80)
print("Fixed: Top 100 momentum -> Lowest ID Top 50 -> 50 stocks")
print(f"Transaction cost: {TRADING_COST:.2%}")
print("No parameter optimization.")

results=[]
for name,start,end in TEST_WINDOWS:
    test=monthly_returns[(monthly_returns["return_date"]>=pd.Timestamp(start))&(monthly_returns["return_date"]<=pd.Timestamp(end))]
    m=metrics(test["monthly_return"])
    if m is None: continue
    total,cagr,vol,sharpe,dd,win,n=m
    print(f"\nUNSEEN TEST — {name}")
    print("-"*40)
    print(f"Total return: {total:.2%}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Annual volatility: {vol:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max DD: {dd:.2%}")
    print(f"Winning months: {win:.2%}")
    print(f"Months: {n}")
    results.append({"period":name,"total_return":total,"cagr":cagr,"volatility":vol,
                    "sharpe":sharpe,"max_drawdown":dd,"winning_months":win,"months":n})

results_df=pd.DataFrame(results)
print("\n"+"="*80)
print("WALK-FORWARD SUMMARY")
print("="*80)
print(f"Average unseen CAGR: {results_df['cagr'].mean():.2%}")
print(f"Median unseen CAGR: {results_df['cagr'].median():.2%}")
print(f"Average unseen Sharpe: {results_df['sharpe'].mean():.2f}")
print(f"Average unseen Max DD: {results_df['max_drawdown'].mean():.2%}")
print(f"Positive unseen periods: {(results_df['cagr']>0).mean():.2%}")

results_df.to_csv("plan2_walk_forward_results_exact.csv",index=False)
monthly_returns.to_csv("plan2_walk_forward_monthly_returns_exact.csv",index=False)
print("Saved: plan2_walk_forward_results_exact.csv")
print("Saved: plan2_walk_forward_monthly_returns_exact.csv")
