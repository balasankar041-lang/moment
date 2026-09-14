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
# PRICE DOWNLOAD
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
# BUILD PLAN 2 MONTHLY RETURNS
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

monthly_rows = []


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

        result = signal(s, date)

        if result is not None:
            signals[s] = result

    if len(signals) < TOP_100:
        continue

    # First ranking: strongest momentum
    top100 = sorted(
        signals.items(),
        key=lambda x: x[1][0],
        reverse=True
    )[:TOP_100]

    # Second ranking: lowest ID
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

    # Turnover cost
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

    net_return = (
        cost_multiplier
        * (1 + portfolio_return)
        - 1
    )

    monthly_rows.append(
        {
            "signal_date": date,
            "return_date": next_date,
            "strategy_return": net_return
        }
    )

    previous = new


strategy = pd.DataFrame(
    monthly_rows
)

if strategy.empty:
    raise SystemExit(
        "No strategy results generated."
    )


# ---------------------------------------------------------
# NIFTY 500 BENCHMARK
# ---------------------------------------------------------

print("\nDOWNLOADING NIFTY 500")
print("=====================")

benchmark = yf.download(
    "^CRSLDX",
    start=download_start.strftime("%Y-%m-%d"),
    end=download_end.strftime("%Y-%m-%d"),
    auto_adjust=True,
    progress=False
)

if benchmark.empty:
    raise SystemExit(
        "Nifty 500 benchmark unavailable."
    )


if isinstance(
    benchmark.columns,
    pd.MultiIndex
):

    benchmark_close = benchmark[
        "Close"
    ].squeeze()

else:

    benchmark_close = benchmark[
        "Close"
    ]


benchmark_monthly = (
    benchmark_close
    .resample("ME")
    .last()
)


benchmark_returns = (
    benchmark_monthly
    .pct_change()
)


# ---------------------------------------------------------
# MERGE
# ---------------------------------------------------------

strategy["nifty500_return"] = (
    strategy["return_date"]
    .map(benchmark_returns)
)

strategy = strategy.dropna(
    subset=["nifty500_return"]
)


strategy["strategy_equity"] = (
    1 + strategy["strategy_return"]
).cumprod()

strategy["nifty_equity"] = (
    1 + strategy["nifty500_return"]
).cumprod()


# ---------------------------------------------------------
# MARKET DOWN MONTHS
# ---------------------------------------------------------

bad_months = strategy[
    strategy["nifty500_return"] < 0
].copy()


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def metrics(returns):

    returns = pd.Series(
        returns
    ).dropna()

    if len(returns) == 0:
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
        if years > 0
        else np.nan
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
# OUTPUT
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("PLAN 2 — BEAR / DIFFICULT MARKET TEST")
print("=" * 80)

print(
    "Strategy: Top 100 momentum -> "
    "Lowest ID Top 50 -> 50 stocks"
)

print(
    "Transaction cost:",
    f"{TRADING_COST * 100:.2f}%"
)

print(
    "\nNifty 500 negative months:",
    len(bad_months)
)


if not bad_months.empty:

    print("\nMONTH-BY-MONTH BAD MARKET TEST")
    print("----------------------------------------")

    for _, row in bad_months.iterrows():

        print(
            row["return_date"].strftime("%Y-%m"),
            "| Nifty 500:",
            f"{row['nifty500_return'] * 100:.2f}%",
            "| Plan 2:",
            f"{row['strategy_return'] * 100:.2f}%"
        )


    strategy_bad = metrics(
        bad_months["strategy_return"]
    )

    nifty_bad = metrics(
        bad_months["nifty500_return"]
    )


    print("\n")
    print("=" * 80)
    print("NEGATIVE NIFTY 500 MONTHS — SUMMARY")
    print("=" * 80)

    print(
        "Plan 2 total return:",
        f"{strategy_bad['total'] * 100:.2f}%"
    )

    print(
        "Plan 2 winning months:",
        f"{strategy_bad['win'] * 100:.2f}%"
    )

    print(
        "Nifty 500 total return:",
        f"{nifty_bad['total'] * 100:.2f}%"
    )

    print(
        "Nifty 500 winning months:",
        f"{nifty_bad['win'] * 100:.2f}%"
    )


    # Count months where strategy beat the market
    beat_count = (
        bad_months["strategy_return"]
        >
        bad_months["nifty500_return"]
    ).sum()

    print(
        "\nPlan 2 beat Nifty 500 in",
        beat_count,
        "of",
        len(bad_months),
        "negative-market months."
    )

    print(
        "Outperformance rate:",
        f"{beat_count / len(bad_months) * 100:.2f}%"
    )


# ---------------------------------------------------------
# WORST MONTHS
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("WORST PLAN 2 MONTHS")
print("=" * 80)

worst = strategy.nsmallest(
    10,
    "strategy_return"
)

for _, row in worst.iterrows():

    print(
        row["return_date"].strftime("%Y-%m"),
        "| Plan 2:",
        f"{row['strategy_return'] * 100:.2f}%",
        "| Nifty 500:",
        f"{row['nifty500_return'] * 100:.2f}%"
    )


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

bad_months.to_csv(
    "plan2_bear_market_results.csv",
    index=False
)

strategy.to_csv(
    "plan2_bear_market_monthly.csv",
    index=False
)


print("\nSaved:")
print(
    "plan2_bear_market_results.csv"
)

print(
    "plan2_bear_market_monthly.csv"
)

print("\nBEAR MARKET TEST COMPLETE")
