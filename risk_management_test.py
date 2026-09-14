import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"

TOP_100 = 100
TOP_50 = 50
TRADING_COST = 0.005

# ---------------------------------------------------------
# MEMBERSHIP
# ---------------------------------------------------------

membership = pd.read_csv(MEMBERSHIP_FILE)

membership["effective_date"] = pd.to_datetime(
    membership["effective_date"]
)

membership["symbol"] = (
    membership["symbol"]
    .astype(str)
    .str.strip()
    .str.upper()
)

membership = membership.dropna(
    subset=["effective_date", "symbol"]
).sort_values(
    ["effective_date", "symbol"]
)


def yahoo_symbol(s):
    s = str(s).strip().upper()

    if s.isalpha() and len(s) <= 20:
        return s + ".NS"

    return None


def members_at_date(date):
    x = membership[
        membership["effective_date"] <= date
    ]

    if x.empty:
        return []

    latest = x["effective_date"].max()

    return sorted(
        set(
            yahoo_symbol(s)
            for s in x.loc[
                x["effective_date"] == latest,
                "symbol"
            ]
            if yahoo_symbol(s)
        )
    )


symbols = sorted(
    set(
        yahoo_symbol(s)
        for s in membership["symbol"]
        if yahoo_symbol(s)
    )
)

download_start = (
    membership["effective_date"].min()
    - pd.DateOffset(years=2)
)

today = pd.Timestamp.today().normalize()

download_end = today + pd.Timedelta(days=1)


# ---------------------------------------------------------
# DOWNLOAD STOCK DATA
# ---------------------------------------------------------

def download_prices(symbols, batch_size=25):

    frames = []

    for i in range(0, len(symbols), batch_size):

        batch = symbols[i:i + batch_size]

        print(
            f"Downloading "
            f"{i + 1}-{min(i + batch_size, len(symbols))}"
        )

        try:

            data = yf.download(
                batch,
                start=download_start.strftime("%Y-%m-%d"),
                end=download_end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if data.empty:
                continue

            if isinstance(
                data.columns,
                pd.MultiIndex
            ):

                if (
                    "Close"
                    not in data.columns
                    .get_level_values(0)
                ):
                    continue

                close = data["Close"]

            else:

                if "Close" not in data.columns:
                    continue

                close = data[["Close"]]
                close.columns = [batch[0]]

            frames.append(close)

        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return (
        pd.concat(
            frames,
            axis=1,
            sort=True
        )
        .loc[
            :,
            lambda x:
                ~x.columns.duplicated()
        ]
        .sort_index()
    )


print("\nDOWNLOADING STOCK DATA")
print("======================")

prices = download_prices(symbols)

prices = prices.dropna(
    axis=1,
    how="all"
)

if prices.empty:
    raise SystemExit(
        "No price data downloaded."
    )


last_trading_day = prices.index.max()

if (
    last_trading_day.to_period("M")
    == today.to_period("M")
):

    completed_month = (
        today.to_period("M") - 1
    ).end_time.normalize()

else:

    completed_month = (
        last_trading_day
        .to_period("M")
        .end_time.normalize()
    )


monthly = prices.resample("ME").last()

monthly = monthly[
    monthly.index <= completed_month
]


print("\nPRICE DATA")
print("==========")
print(
    "Stocks with actual data:",
    prices.shape[1]
)
print(
    "Trading days:",
    len(prices)
)
print(
    "Last trading day:",
    last_trading_day.date()
)
print(
    "Last completed month:",
    completed_month.date()
)


# ---------------------------------------------------------
# PLAN 2 SIGNAL
# ---------------------------------------------------------

def signal(symbol, date):

    if symbol not in prices.columns:
        return None

    s = prices[symbol].dropna()

    start_cut = (
        date - pd.DateOffset(months=12)
    )

    end_cut = (
        date - pd.DateOffset(months=2)
    )

    a = s.loc[
        s.index <= start_cut
    ]

    b = s.loc[
        s.index <= end_cut
    ]

    if a.empty or b.empty:
        return None

    p0 = a.iloc[-1]
    p1 = b.iloc[-1]

    if p0 <= 0 or p1 <= 0:
        return None

    momentum = p1 / p0 - 1

    path = s.loc[
        (s.index >= a.index[-1])
        &
        (s.index <= b.index[-1])
    ]

    if len(path) < 100:
        return None

    daily_returns = (
        path.pct_change()
        .dropna()
    )

    daily_returns = daily_returns[
        daily_returns != 0
    ]

    if daily_returns.empty:
        return None

    positive_days = (
        daily_returns > 0
    ).mean()

    negative_days = (
        daily_returns < 0
    ).mean()

    id_score = (
        (1 if momentum > 0 else -1)
        *
        (
            negative_days
            - positive_days
        )
    )

    return momentum, id_score


# ---------------------------------------------------------
# NIFTY 500
# ---------------------------------------------------------

print("\nDOWNLOADING NIFTY 500")
print("=====================")

nifty = yf.download(
    "^CRSLDX",
    start=download_start.strftime("%Y-%m-%d"),
    end=download_end.strftime("%Y-%m-%d"),
    auto_adjust=True,
    progress=False
)

if nifty.empty:
    raise SystemExit(
        "Nifty 500 data unavailable."
    )

if isinstance(
    nifty.columns,
    pd.MultiIndex
):

    nifty_close = nifty[
        "Close"
    ].squeeze()

else:

    nifty_close = nifty[
        "Close"
    ]


nifty_monthly = (
    nifty_close
    .resample("ME")
    .last()
)

nifty_return = (
    nifty_monthly
    .pct_change()
)


# ---------------------------------------------------------
# 200 DMA REGIME
# ---------------------------------------------------------

nifty_daily = nifty_close.dropna()

nifty_200dma = (
    nifty_daily
    .rolling(200)
    .mean()
)

# Month-end regime:
# ON  = Nifty close >= 200 DMA
# OFF = Nifty close < 200 DMA

regime_daily = (
    nifty_daily >= nifty_200dma
)

regime_monthly = (
    regime_daily
    .resample("ME")
    .last()
)

regime_monthly = regime_monthly.astype(bool)


# ---------------------------------------------------------
# BUILD PLAN 2
# ---------------------------------------------------------

dates = [
    d
    for d in monthly.index
    if d >= (
        membership["effective_date"].min()
        + pd.DateOffset(months=13)
    )
]

dates = [
    d
    for d in dates
    if d < completed_month
]


previous = {}

rows = []


for i, date in enumerate(dates):

    if i + 1 >= len(monthly.index):
        break

    next_date = monthly.index[
        monthly.index.get_loc(date) + 1
    ]

    candidates = [
        s
        for s in members_at_date(date)
        if s in prices.columns
    ]

    signals = {}

    for s in candidates:

        z = signal(s, date)

        if z is not None:
            signals[s] = z

    if len(signals) < TOP_100:
        continue

    # PLAN 2
    # First: Top 100 momentum
    top100 = sorted(
        signals.items(),
        key=lambda x: x[1][0],
        reverse=True
    )[:TOP_100]

    # Second: Lowest ID
    selected = [
        s
        for s, z in sorted(
            top100,
            key=lambda x: x[1][1]
        )[:TOP_50]
    ]

    weight = 1.0 / len(selected)

    new = {
        s: weight
        for s in selected
    }

    all_symbols = (
        set(previous)
        | set(new)
    )

    turnover = sum(
        abs(
            new.get(s, 0)
            - previous.get(s, 0)
        )
        for s in all_symbols
    )

    cost_multiplier = (
        1
        - turnover * TRADING_COST
    )

    portfolio_return = 0.0

    for s, w in new.items():

        try:

            p0 = monthly.loc[
                date, s
            ]

            p1 = monthly.loc[
                next_date, s
            ]

            if (
                pd.notna(p0)
                and pd.notna(p1)
                and p0 > 0
            ):

                portfolio_return += (
                    w * (p1 / p0 - 1)
                )

        except Exception:
            pass

    original_return = (
        cost_multiplier
        * (1 + portfolio_return)
        - 1
    )

    # -----------------------------------------------------
    # RISK OVERLAY
    #
    # If Nifty 500 is below its 200 DMA at month-end,
    # hold cash for the next month.
    # -----------------------------------------------------

    regime_on = regime_monthly.get(
        date,
        True
    )

    if regime_on:

        overlay_return = original_return

    else:

        overlay_return = 0.0

    rows.append(
        {
            "signal_date": date,
            "return_date": next_date,
            "plan2_return": original_return,
            "regime_on": regime_on,
            "risk_managed_return": overlay_return,
            "nifty500_return": nifty_return.get(
                next_date,
                np.nan
            )
        }
    )

    previous = new


result = pd.DataFrame(rows)

result = result.dropna(
    subset=["nifty500_return"]
)

if result.empty:
    raise SystemExit(
        "No results generated."
    )


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def metrics(returns):

    returns = pd.Series(
        returns
    ).dropna()

    if len(returns) < 2:
        return None

    total = (
        (1 + returns).prod()
        - 1
    )

    years = len(returns) / 12

    cagr = (
        (1 + total)
        ** (1 / years)
        - 1
    )

    volatility = (
        returns.std()
        * np.sqrt(12)
    )

    sharpe = (
        returns.mean()
        * 12
        / volatility
        if volatility > 0
        else np.nan
    )

    equity = (
        1 + returns
    ).cumprod()

    drawdown = (
        equity
        / equity.cummax()
        - 1
    ).min()

    win = (
        returns > 0
    ).mean()

    return {
        "total": total,
        "cagr": cagr,
        "vol": volatility,
        "sharpe": sharpe,
        "dd": drawdown,
        "win": win,
        "months": len(returns)
    }


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

plan2 = metrics(
    result["plan2_return"]
)

managed = metrics(
    result["risk_managed_return"]
)


print("\n")
print("=" * 80)
print("PLAN 2 — RISK MANAGEMENT TEST")
print("=" * 80)

print(
    "Overlay: Nifty 500 below 200 DMA = CASH"
)

print(
    "Transaction cost:",
    f"{TRADING_COST * 100:.2f}%"
)


print("\nORIGINAL PLAN 2")
print("----------------------------------------")

print(
    "Total return:",
    f"{plan2['total'] * 100:.2f}%"
)

print(
    "CAGR:",
    f"{plan2['cagr'] * 100:.2f}%"
)

print(
    "Annual volatility:",
    f"{plan2['vol'] * 100:.2f}%"
)

print(
    "Sharpe:",
    f"{plan2['sharpe']:.2f}"
)

print(
    "Max DD:",
    f"{plan2['dd'] * 100:.2f}%"
)

print(
    "Winning months:",
    f"{plan2['win'] * 100:.2f}%"
)


print("\nRISK-MANAGED PLAN 2")
print("----------------------------------------")

print(
    "Total return:",
    f"{managed['total'] * 100:.2f}%"
)

print(
    "CAGR:",
    f"{managed['cagr'] * 100:.2f}%"
)

print(
    "Annual volatility:",
    f"{managed['vol'] * 100:.2f}%"
)

print(
    "Sharpe:",
    f"{managed['sharpe']:.2f}"
)

print(
    "Max DD:",
    f"{managed['dd'] * 100:.2f}%"
)

print(
    "Winning months:",
    f"{managed['win'] * 100:.2f}%"
)


cash_months = (
    result["regime_on"] == False
).sum()

print("\nREGIME INFORMATION")
print("----------------------------------------")

print(
    "Cash months:",
    cash_months
)

print(
    "Invested months:",
    len(result) - cash_months
)


# ---------------------------------------------------------
# YEARLY COMPARISON
# ---------------------------------------------------------

result["year"] = (
    result["return_date"]
    .dt.year
)

yearly = []

for year, group in result.groupby("year"):

    p2 = (
        (1 + group["plan2_return"])
        .prod()
        - 1
    )

    rm = (
        (1 + group["risk_managed_return"])
        .prod()
        - 1
    )

    yearly.append(
        {
            "year": year,
            "plan2": p2,
            "risk_managed": rm,
            "cash_months": (
                ~group["regime_on"]
            ).sum()
        }
    )


yearly_df = pd.DataFrame(yearly)


print("\nYEARLY COMPARISON")
print("----------------------------------------")

for _, row in yearly_df.iterrows():

    print(
        int(row["year"]),
        "| Plan 2:",
        f"{row['plan2'] * 100:.2f}%",
        "| Risk managed:",
        f"{row['risk_managed'] * 100:.2f}%",
        "| Cash months:",
        int(row["cash_months"])
    )


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

result.to_csv(
    "plan2_risk_management_monthly.csv",
    index=False
)

yearly_df.to_csv(
    "plan2_risk_management_yearly.csv",
    index=False
)


print("\nSaved:")
print(
    "plan2_risk_management_monthly.csv"
)

print(
    "plan2_risk_management_yearly.csv"
)

print("\nRISK MANAGEMENT TEST COMPLETE")
