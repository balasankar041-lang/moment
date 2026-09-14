import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"

TOP_100 = 100
TOP_50 = 50

TRADING_COST = 0.005
TRAIN_END = pd.Timestamp("2023-12-31")


# ============================================================
# HISTORICAL MEMBERSHIP
# ============================================================

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

membership = (
    membership
    .dropna(subset=["effective_date", "symbol"])
    .sort_values(["effective_date", "symbol"])
)

print("\nHISTORICAL MEMBERSHIP")
print("=====================")
print("Records:", len(membership))
print(
    "Snapshots:",
    membership["effective_date"].nunique()
)
print(
    "Date range:",
    membership["effective_date"].min().date(),
    "to",
    membership["effective_date"].max().date()
)


def yahoo_symbol(s):

    s = str(s).strip().upper()

    return (
        s + ".NS"
        if s.isalpha() and len(s) <= 20
        else None
    )


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


# ============================================================
# DOWNLOAD PRICE DATA
# ============================================================

download_start = (
    membership["effective_date"].min()
    - pd.DateOffset(years=2)
)

today = pd.Timestamp.today().normalize()

download_end = today + pd.Timedelta(days=1)

print("\nDownloading price data...")
print(
    "Price period:",
    download_start.date(),
    "to",
    today.date()
)
print(
    "Historical symbols:",
    len(symbols)
)


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
            f"Downloading {i+1}-"
            f"{min(i+batch_size, len(symbols))}"
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
                threads=True
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


prices = download_prices(symbols)

prices = prices.dropna(
    axis=1,
    how="all"
)

if prices.empty:

    raise SystemExit(
        "No price data downloaded."
    )


# ============================================================
# COMPLETED MONTH
# ============================================================

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
        .end_time
        .normalize()
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
    "Last available trading day:",
    last_trading_day.date()
)
print(
    "Last completed month:",
    completed_month.date()
)


# ============================================================
# 12-2 MOMENTUM + INFORMATION DISCRETION
# ============================================================

def signal(symbol, date):

    if symbol not in prices.columns:
        return None

    s = prices[symbol].dropna()

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

    ret = p1 / p0 - 1

    path = s.loc[
        (s.index >= a.index[-1]) &
        (s.index <= b.index[-1])
    ]

    if len(path) < 100:
        return None

    dr = path.pct_change().dropna()

    dr = dr[dr != 0]

    if dr.empty:
        return None

    pos = (dr > 0).mean()

    neg = (dr < 0).mean()

    id_score = (
        (1 if ret > 0 else -1)
        * (neg - pos)
    )

    return ret, id_score


# ============================================================
# REBALANCE DATES
# ============================================================

dates = [
    d
    for d in monthly.index
    if d <= completed_month
]

dates = [
    d
    for d in dates
    if d >= (
        membership["effective_date"].min()
        + pd.DateOffset(months=13)
    )
]


# ============================================================
# BACKTEST
# ============================================================

equity = 1.0

rows = []

previous = {}

rebalance_count = 0

skipped = 0

audit = []


for i, date in enumerate(
    dates[:-1]
):

    next_date = dates[i + 1]

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

        skipped += 1
        continue


    # --------------------------------------------------------
    # FIRST SORT: Top 100 momentum
    # --------------------------------------------------------

    top100 = sorted(
        signals.items(),
        key=lambda x: x[1][0],
        reverse=True
    )[:TOP_100]


    # --------------------------------------------------------
    # SECOND SORT: Lowest ID
    # --------------------------------------------------------

    selected = [
        s
        for s, z in sorted(
            top100,
            key=lambda x: x[1][1]
        )[:TOP_50]
    ]


    # --------------------------------------------------------
    # Equal weight
    # --------------------------------------------------------

    w = 1.0 / len(selected)

    new = {
        s: w
        for s in selected
    }


    # --------------------------------------------------------
    # Transaction cost
    # --------------------------------------------------------

    all_s = (
        set(previous)
        | set(new)
    )

    turnover = sum(
        abs(
            new.get(s, 0)
            - previous.get(s, 0)
        )
        for s in all_s
    )

    equity *= (
        1
        - turnover * TRADING_COST
    )


    # --------------------------------------------------------
    # Monthly return
    # --------------------------------------------------------

    period_ret = 0.0

    for s, weight in new.items():

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

                period_ret += (
                    weight
                    * (p1 / p0 - 1)
                )

        except Exception:
            pass


    equity *= (
        1 + period_ret
    )

    rows.append(
        (
            next_date,
            equity
        )
    )


    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    for s in selected:

        audit.append({

            "rebalance_date": date,

            "symbol": s,

            "momentum_12_2":
                signals[s][0],

            "ID":
                signals[s][1]

        })


    previous = new

    rebalance_count += 1


# ============================================================
# EQUITY CURVE
# ============================================================

if not rows:

    raise SystemExit(
        "No backtest results generated."
    )


eq = pd.Series(
    dict(rows)
).sort_index()


# ============================================================
# METRICS
# ============================================================

def metrics(
    x,
    start=None,
    end=None
):

    x = x.copy()

    if start is not None:
        x = x[
            x.index >= start
        ]

    if end is not None:
        x = x[
            x.index <= end
        ]

    r = x.pct_change().dropna()

    if len(x) < 2:
        return None

    total = (
        x.iloc[-1]
        / x.iloc[0]
        - 1
    )

    years = max(
        (
            x.index[-1]
            - x.index[0]
        ).days / 365.25,
        1 / 12
    )

    cagr = (
        x.iloc[-1]
        / x.iloc[0]
    ) ** (1 / years) - 1

    vol = (
        r.std()
        * np.sqrt(12)
    )

    sharpe = (
        r.mean() * 12 / vol
        if vol > 0
        else np.nan
    )

    dd = (
        x
        / x.cummax()
        - 1
    ).min()

    win = (
        r > 0
    ).mean()

    return (
        total,
        cagr,
        vol,
        sharpe,
        dd,
        win,
        len(r)
    )


def show(title, m):

    print("\n" + title)

    print(
        "-" * len(title)
    )

    if m is None:

        print(
            "Insufficient data"
        )

        return

    (
        total,
        cagr,
        vol,
        sharpe,
        dd,
        win,
        n
    ) = m

    print(
        "Total return:",
        f"{total * 100:.2f}%"
    )

    print(
        "CAGR:",
        f"{cagr * 100:.2f}%"
    )

    print(
        "Annual volatility:",
        f"{vol * 100:.2f}%"
    )

    print(
        "Sharpe ratio:",
        f"{sharpe:.2f}"
    )

    print(
        "Maximum drawdown:",
        f"{dd * 100:.2f}%"
    )

    print(
        "Winning months:",
        f"{win * 100:.2f}%"
    )

    print(
        "Months tested:",
        n
    )


# ============================================================
# RESULTS
# ============================================================

print(
    "\nNIFTY 500 HISTORICAL-MEMBERSHIP"
)

print(
    "12-2 MOMENTUM + INFORMATION DISCRETION"
)

print(
    "=========================================="
)

print(
    "First sort: Top 100 by 12-2 momentum"
)

print(
    "Second sort: Lowest ID"
)

print(
    "Final portfolio: Top 50"
)

print(
    "Trading cost:",
    f"{TRADING_COST * 100:.2f}%"
)

print(
    "Rebalances:",
    rebalance_count
)

print(
    "Skipped months:",
    skipped
)


show(
    "TRAINING (2020-2023)",
    metrics(
        eq,
        pd.Timestamp("2020-03-31"),
        TRAIN_END
    )
)


show(
    "UNSEEN TEST (2024-2026)",
    metrics(
        eq,
        pd.Timestamp("2024-01-31"),
        completed_month
    )
)


show(
    "FULL PERIOD",
    metrics(eq)
)


# ============================================================
# SAVE AUDIT
# ============================================================

pd.DataFrame(
    audit
).to_csv(
    "id_strategy_selections.csv",
    index=False
)

print(
    "\nSelection audit saved:"
    " id_strategy_selections.csv"
)

print(
    "BACKTEST COMPLETE"
)
