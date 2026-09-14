import pandas as pd
import numpy as np
import yfinance as yf

MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"
START = "2019-01-01"
END = "2026-09-30"
TOP_MOMENTUM = 100
TOP_ID = 50
PORTFOLIO_SIZE = 15
COST = 0.005

membership = pd.read_csv(MEMBERSHIP_FILE)
membership["effective_date"] = pd.to_datetime(membership["effective_date"])
membership["symbol"] = membership["symbol"].astype(str).str.strip().str.upper()
membership = membership.drop_duplicates(["effective_date", "symbol"])
membership_by_date = {d: set(g["symbol"]) for d,g in membership.groupby("effective_date")}
membership_dates = sorted(membership_by_date)

def members_at(date):
    valid = [d for d in membership_dates if d <= date]
    return membership_by_date[valid[-1]] if valid else set()

def price_before(series, date):
    x = series.index[series.index <= date]
    return series.loc[x[-1]] if len(x) else np.nan

symbols = sorted(membership["symbol"].unique())
tickers = [s + ".NS" for s in symbols]
frames = []

print("Downloading stock prices...")
for i in range(0, len(tickers), 25):
    batch = tickers[i:i+25]
    try:
        x = yf.download(batch, start=START, end=END, auto_adjust=True,
                        progress=False, threads=False, group_by="column")
        if x.empty: continue
        if isinstance(x.columns, pd.MultiIndex):
            x = x["Close"]
        else:
            x = x[["Close"]]; x.columns = batch
        frames.append(x)
        print(f"Downloaded {min(i+25,len(tickers))}/{len(tickers)}")
    except Exception as e:
        print("Batch failed:", e)

if not frames:
    raise RuntimeError("No price data downloaded.")

close = pd.concat(frames, axis=1)
close = close.loc[:, ~close.columns.duplicated()].dropna(axis=1, how="all").sort_index()

print(f"Stocks with usable data: {close.shape[1]}")
print(f"Trading days: {len(close)}")

bench = yf.download("^CRSLDX", start=START, end=END, auto_adjust=True,
                    progress=False, threads=False)
if bench.empty:
    raise RuntimeError("Nifty 500 benchmark unavailable.")
if isinstance(bench.columns, pd.MultiIndex):
    bench = bench["Close"].iloc[:,0]
else:
    bench = bench["Close"]
bench = bench.dropna()

month_ends = close.resample("ME").last().index
last_completed = pd.Timestamp.today().to_period("M").start_time - pd.Timedelta(days=1)
month_ends = month_ends[(month_ends <= last_completed) & (month_ends >= pd.Timestamp("2020-02-29"))]

rets, dates = [], []
previous = set()
skipped = 0

for month_end in month_ends:
    members = members_at(month_end)
    end2 = month_end - pd.DateOffset(months=2)
    end12 = month_end - pd.DateOffset(months=12)
    end6 = month_end - pd.DateOffset(months=6)

    bn = price_before(bench, month_end)
    b6 = price_before(bench, end6)
    if pd.isna(bn) or pd.isna(b6) or b6 <= 0:
        skipped += 1; continue
    bench6 = bn/b6 - 1

    mom, ids, rs = {}, {}, {}

    for symbol in members:
        t = symbol + ".NS"
        if t not in close.columns: continue
        s = close[t].dropna()
        if len(s) < 270: continue

        d12s = s.index[s.index <= end12]
        d2s = s.index[s.index <= end2]
        if not len(d12s) or not len(d2s): continue
        d12, d2 = d12s[-1], d2s[-1]
        p12, p2 = s.loc[d12], s.loc[d2]
        if p12 <= 0: continue

        m = p2/p12 - 1
        path = s.loc[d12:d2].pct_change().dropna()
        pos, neg = (path > 0).sum(), (path < 0).sum()
        directional = pos + neg
        if directional == 0: continue
        idv = np.sign(m) * (neg/directional - pos/directional)

        sn, s6 = price_before(s, month_end), price_before(s, end6)
        if pd.isna(sn) or pd.isna(s6) or s6 <= 0: continue
        rsval = (sn/s6 - 1) - bench6

        mom[symbol], ids[symbol], rs[symbol] = m, idv, rsval

    if len(mom) < TOP_MOMENTUM:
        skipped += 1; continue

    top100 = pd.Series(mom).sort_values(ascending=False).head(TOP_MOMENTUM)
    top100 = top100[top100 > 0]
    if len(top100) < PORTFOLIO_SIZE:
        skipped += 1; continue

    top50 = pd.Series({s:ids[s] for s in top100.index if s in ids}).sort_values().head(TOP_ID)
    if len(top50) < PORTFOLIO_SIZE:
        skipped += 1; continue

    # Relative-strength confirmation: stock 6M return must beat Nifty 500.
    passing = [s for s in top50.index if rs.get(s, -np.inf) > 0]
    if len(passing) < PORTFOLIO_SIZE:
        skipped += 1; continue

    selected = top50.loc[passing].head(PORTFOLIO_SIZE)
    portfolio = set(selected.index)

    next_month = month_end + pd.offsets.MonthEnd(1)
    if next_month not in month_ends: continue

    stock_rets = []
    for symbol in portfolio:
        s = close[symbol + ".NS"].dropna()
        starts, ends = s.index[s.index > month_end], s.index[s.index <= next_month]
        if not len(starts) or not len(ends): continue
        a, b = starts[0], ends[-1]
        if b <= a: continue
        r = s.loc[b]/s.loc[a] - 1
        if pd.notna(r): stock_rets.append(r)

    if len(stock_rets) < max(5, PORTFOLIO_SIZE//2):
        skipped += 1; continue

    gross = np.mean(stock_rets)
    turnover = len(portfolio.symmetric_difference(previous))
    turnover_fraction = turnover/(len(portfolio)+len(previous)) if previous else 1.0
    rets.append(gross - COST*turnover_fraction)
    dates.append(next_month)
    previous = portfolio

returns = pd.Series(rets, index=pd.DatetimeIndex(dates)).sort_index()
if returns.empty: raise RuntimeError("No strategy returns produced.")

equity = (1+returns).cumprod()
total = equity.iloc[-1]-1
years = len(returns)/12
cagr = equity.iloc[-1]**(1/years)-1
vol = returns.std(ddof=1)*np.sqrt(12)
sharpe = returns.mean()/returns.std(ddof=1)*np.sqrt(12)
dd = (equity/equity.cummax()-1).min()
win = (returns>0).mean()

print("\n"+"="*60)
print("PLAN 2 + RELATIVE STRENGTH RESULTS")
print("="*60)
print("6M relative strength vs Nifty 500")
print("Filter: stock 6M return > Nifty 500 6M return")
print(f"Cost assumption: {COST:.2%}")
print(f"Total return: {total:.2%}")
print(f"CAGR: {cagr:.2%}")
print(f"Annual volatility: {vol:.2%}")
print(f"Sharpe ratio: {sharpe:.2f}")
print(f"Maximum drawdown: {dd:.2%}")
print(f"Winning months: {win:.2%}")
print(f"Months tested: {len(returns)}")
print(f"Rebalances: {len(returns)}")
print(f"Skipped months: {skipped}")
pd.DataFrame({"date":returns.index,"monthly_return":returns.values,"equity":equity.values}).to_csv(
    "plan2_relative_strength_monthly_returns.csv", index=False)
print("Saved: plan2_relative_strength_monthly_returns.csv")
