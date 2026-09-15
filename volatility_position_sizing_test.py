import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"

TOP_100 = 100
TOP_50 = 50
TRADING_COST = 0.005

# Volatility targets
TARGET_VOL_LOW = 0.20
TARGET_VOL_HIGH = 0.30

# Exposure limits
MIN_EXPOSURE = 0.50
MAX_EXPOSURE = 1.00


# =========================================================
# MEMBERSHIP
# =========================================================

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


# =========================================================
# PRICE DATA
# =========================================================

download_start = (
    membership["effective_date"].min()
    - pd.DateOffset(years=2)
)

today = pd.Timestamp.today().normalize()

download_end = today + pd.Timedelta(days=1)


def download_prices(symbols, batch_size=25):

    frames = []

    for i in range(
        0,
        len(symbols),
        batch_size
    ):

        batch = symbols[
            i:i + batch_size
        ]

        print(
            f"Downloading "
            f"{i + 1}-"
            f"{min(i + batch_size, len(symbols))}"
        )

        try:

            data = yf.download(
                batch,
                start=download_start.strftime(
                    "%Y-%m-%d"
                ),
                end=download_end.strftime(
                    "%Y-%m-%d"
                ),
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
                    not in
                    data.columns
                    .get_level_values(0)
                ):
                    continue

                close = data["Close"]

            else:

                if "Close" not in data.columns:
                    continue

                close = data[["Close"]]
                close.columns = [
                    batch[0]
                ]

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


print("\nDOWNLOADING PRICE DATA")
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
        today.to_period("M")
        - 1
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


# =========================================================
# PLAN 2 SIGNAL
# =========================================================

def signal(symbol, date):

    if symbol not in prices.columns:
        return None

    s = prices[
        symbol
    ].dropna()

    start_cut = (
        date
        - pd.DateOffset(months=12)
    )

    end_cut = (
        date
        - pd.DateOffset(months=2)
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
        path
        .pct_change()
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


# =========================================================
# BUILD PLAN 2
# =========================================================

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

    # -----------------------------------------------------
    # PLAN 2
    # -----------------------------------------------------

    top100 = sorted(
        signals.items(),
        key=lambda x: x[1][0],
        reverse=True
    )[:TOP_100]

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

    # -----------------------------------------------------
    # TURNOVER
    # -----------------------------------------------------

    all_symbols = (
        set(previous)
        | set(new)
    )

    turnover = sum(
        abs(
            new.get(s, 0)
            -
            previous.get(s, 0)
        )
        for s in all_symbols
    )

    cost_multiplier = (
        1
        -
        turnover * TRADING_COST
    )

    # -----------------------------------------------------
    # PORTFOLIO RETURN
    # -----------------------------------------------------

    portfolio_return = 0.0

    for s, w in new.items():

        try:

            p0 = monthly.loc[
                date,
                s
            ]

            p1 = monthly.loc[
                next_date,
                s
            ]

            if (
                pd.notna(p0)
                and pd.notna(p1)
                and p0 > 0
            ):

                portfolio_return += (
                    w
                    *
                    (p1 / p0 - 1)
                )

        except Exception:
            pass

    original_return = (
        cost_multiplier
        *
        (1 + portfolio_return)
        - 1
    )

    # =====================================================
    # REALIZED PORTFOLIO VOLATILITY
    #
    # Use previous 12 monthly portfolio returns.
    # =====================================================

    historical_returns = [
        r["plan2_return"]
        for r in rows
        if r["return_date"] < date
    ]

    if len(historical_returns) >= 6:

        realized_vol = (
            pd.Series(
                historical_returns[-12:]
            )
            .std()
            *
            np.sqrt(12)
        )

    else:

        realized_vol = TARGET_VOL_LOW

    if (
        pd.isna(realized_vol)
        or realized_vol <= 0
    ):
        realized_vol = TARGET_VOL_LOW

    # -----------------------------------------------------
    # EXPOSURE
    #
    # 20% vol -> 100% exposure
    # 30%+ vol -> 50% exposure
    # Between them -> linear reduction
    # -----------------------------------------------------

    if realized_vol <= TARGET_VOL_LOW:

        exposure = MAX_EXPOSURE

    elif realized_vol >= TARGET_VOL_HIGH:

        exposure = MIN_EXPOSURE

    else:

        exposure = (
            MAX_EXPOSURE
            -
            (
                (
                    realized_vol
                    - TARGET_VOL_LOW
                )
                /
                (
                    TARGET_VOL_HIGH
                    - TARGET_VOL_LOW
                )
            )
            *
            (
                MAX_EXPOSURE
                - MIN_EXPOSURE
            )
        )

    risk_managed_return = (
        exposure
        * original_return
    )

    rows.append(
        {
            "signal_date": date,
            "return_date": next_date,
            "plan2_return": original_return,
            "realized_vol": realized_vol,
            "exposure": exposure,
            "risk_managed_return":
                risk_managed_return
        }
    )

    previous = new


result = pd.DataFrame(rows)

if result.empty:
    raise SystemExit(
        "No results generated."
    )


# =========================================================
# METRICS
# =========================================================

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
        /
        equity.cummax()
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


plan2 = metrics(
    result["plan2_return"]
)

managed = metrics(
    result["risk_managed_return"]
)


# =========================================================
# RESULTS
# =========================================================

print("\n")
print("=" * 80)
print(
    "PLAN 2 — VOLATILITY POSITION SIZING TEST"
)
print("=" * 80)

print(
    "Base: Top 100 momentum -> "
    "Lowest ID Top 50 -> 50 stocks"
)

print(
    "Transaction cost:",
    f"{TRADING_COST * 100:.2f}%"
)

print(
    "Exposure:",
    "100% at <=20% vol, "
    "50% at >=30% vol"
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


print("\nVOLATILITY-MANAGED PLAN 2")
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


print("\nEXPOSURE")
print("----------------------------------------")

print(
    "Average exposure:",
    f"{result['exposure'].mean() * 100:.2f}%"
)

print(
    "Minimum exposure:",
    f"{result['exposure'].min() * 100:.2f}%"
)

print(
    "Maximum exposure:",
    f"{result['exposure'].max() * 100:.2f}%"
)

print(
    "Months below 100% exposure:",
    int(
        (
            result["exposure"]
            < 1.0
        ).sum()
    )
)


# =========================================================
# YEARLY
# =========================================================

result["year"] = (
    result["return_date"].dt.year
)

yearly = []

for year, group in result.groupby(
    "year"
):

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
            "average_exposure":
                group["exposure"].mean()
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
        "| Managed:",
        f"{row['risk_managed'] * 100:.2f}%",
        "| Avg exposure:",
        f"{row['average_exposure'] * 100:.1f}%"
    )


# =========================================================
# SAVE
# =========================================================

result.to_csv(
    "plan2_volatility_sizing_monthly.csv",
    index=False
)

yearly_df.to_csv(
    "plan2_volatility_sizing_yearly.csv",
    index=False
)


print("\nSaved:")
print(
    "plan2_volatility_sizing_monthly.csv"
)

print(
    "plan2_volatility_sizing_yearly.csv"
)

print(
    "\nVOLATILITY POSITION SIZING TEST COMPLETE"
)
