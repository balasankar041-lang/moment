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

# ============================================================
# LOAD HISTORICAL NIFTY 500 MEMBERSHIP
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

membership = membership.drop_duplicates(
    ["effective_date", "symbol"]
)

membership_by_date = {
    d: set(g["symbol"])
    for d, g in membership.groupby("effective_date")
}

membership_dates = sorted(membership_by_date)


def members_at(date):
    valid = [
        d for d in membership_dates
        if d <= date
    ]

    if not valid:
        return set()

    return membership_by_date[valid[-1]]


# ============================================================
# HELPER
# ============================================================

def price_before(series, date):
    x = series.index[
        series.index <= date
    ]

    if len(x) == 0:
        return np.nan

    return series.loc[x[-1]]


# ============================================================
# DOWNLOAD STOCK DATA
# ============================================================

symbols = sorted(
    membership["symbol"].unique()
)

tickers = [
    s + ".NS"
    for s in symbols
]

frames = []

print("Downloading stock prices...")

for i in range(0, len(tickers), 25):

    batch = tickers[i:i + 25]

    try:

        x = yf.download(
            batch,
            start=START,
            end=END,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column"
        )

        if x.empty:
            continue

        if isinstance(
            x.columns,
            pd.MultiIndex
        ):

            x = x["Close"]

        else:

            x = x[["Close"]]
            x.columns = batch

        frames.append(x)

        print(
            f"Downloaded "
            f"{min(i + 25, len(tickers))}/"
            f"{len(tickers)}"
        )

    except Exception as e:

        print(
            "Batch failed:",
            e
        )


if not frames:

    raise RuntimeError(
        "No price data downloaded."
    )


close = pd.concat(
    frames,
    axis=1
)

close = (
    close
    .loc[:, ~close.columns.duplicated()]
    .dropna(axis=1, how="all")
    .sort_index()
)

print(
    f"Stocks with usable data: "
    f"{close.shape[1]}"
)

print(
    f"Trading days: "
    f"{len(close)}"
)


# ============================================================
# MONTH END DATES
# ============================================================

month_ends = (
    close
    .resample("ME")
    .last()
    .index
)

last_completed = (
    pd.Timestamp.today()
    .to_period("M")
    .start_time
    - pd.Timedelta(days=1)
)

month_ends = month_ends[
    (month_ends <= last_completed)
    &
    (month_ends >= pd.Timestamp("2020-02-29"))
]


# ============================================================
# PLAN 2
#
# 1. 12–2 momentum
# 2. Keep positive momentum
# 3. Top 100 momentum
# 4. Information Discreteness
# 5. Lowest 50 ID
# 6. Select Top 15
# 7. Hold for next month
# ============================================================

rets = []
dates = []

previous = set()
skipped = 0

for month_end in month_ends:

    members = members_at(
        month_end
    )

    # 12–2 momentum window
    end2 = (
        month_end
        - pd.DateOffset(months=2)
    )

    end12 = (
        month_end
        - pd.DateOffset(months=12)
    )

    mom = {}
    ids = {}

    # --------------------------------------------------------
    # Calculate momentum + information discreteness
    # --------------------------------------------------------

    for symbol in members:

        ticker = symbol + ".NS"

        if ticker not in close.columns:
            continue

        s = close[ticker].dropna()

        if len(s) < 270:
            continue

        d12s = s.index[
            s.index <= end12
        ]

        d2s = s.index[
            s.index <= end2
        ]

        if (
            not len(d12s)
            or not len(d2s)
        ):
            continue

        d12 = d12s[-1]
        d2 = d2s[-1]

        p12 = s.loc[d12]
        p2 = s.loc[d2]

        if p12 <= 0:
            continue

        # 12–2 momentum
        momentum = (
            p2 / p12
        ) - 1

        # Daily returns over same 12–2 window
        path = (
            s.loc[d12:d2]
            .pct_change()
            .dropna()
        )

        positive_days = (
            path > 0
        ).sum()

        negative_days = (
            path < 0
        ).sum()

        directional_days = (
            positive_days
            + negative_days
        )

        if directional_days == 0:
            continue

        # Information Discreteness
        #
        # ID = sign(momentum) *
        #      (% negative days - % positive days)
        #
        # Lower ID = smoother momentum
        id_value = (
            np.sign(momentum)
            *
            (
                negative_days
                / directional_days
                -
                positive_days
                / directional_days
            )
        )

        mom[symbol] = momentum
        ids[symbol] = id_value

    # --------------------------------------------------------
    # Need enough stocks
    # --------------------------------------------------------

    if len(mom) < TOP_MOMENTUM:

        skipped += 1
        continue

    # --------------------------------------------------------
    # FIRST RANK:
    # Top 100 momentum
    # --------------------------------------------------------

    top100 = (
        pd.Series(mom)
        .sort_values(
            ascending=False
        )
        .head(TOP_MOMENTUM)
    )

    # Positive momentum only
    top100 = top100[
        top100 > 0
    ]

    if len(top100) < PORTFOLIO_SIZE:

        skipped += 1
        continue

    # --------------------------------------------------------
    # SECOND RANK:
    # Lowest Information Discreteness
    # --------------------------------------------------------

    top50 = (
        pd.Series(
            {
                s: ids[s]
                for s in top100.index
                if s in ids
            }
        )
        .sort_values(
            ascending=True
        )
        .head(TOP_ID)
    )

    if len(top50) < PORTFOLIO_SIZE:

        skipped += 1
        continue

    # --------------------------------------------------------
    # FINAL PORTFOLIO:
    # Top 15 from lowest ID group
    # --------------------------------------------------------

    selected = top50.head(
        PORTFOLIO_SIZE
    )

    portfolio = set(
        selected.index
    )

    # --------------------------------------------------------
    # NEXT MONTH
    # --------------------------------------------------------

    next_month = (
        month_end
        + pd.offsets.MonthEnd(1)
    )

    if next_month not in month_ends:
        continue

    stock_rets = []

    for symbol in portfolio:

        ticker = symbol + ".NS"

        s = close[ticker].dropna()

        starts = s.index[
            s.index > month_end
        ]

        ends = s.index[
            s.index <= next_month
        ]

        if (
            not len(starts)
            or not len(ends)
        ):
            continue

        a = starts[0]
        b = ends[-1]

        if b <= a:
            continue

        r = (
            s.loc[b]
            /
            s.loc[a]
        ) - 1

        if pd.notna(r):
            stock_rets.append(r)

    if len(stock_rets) < max(
        5,
        PORTFOLIO_SIZE // 2
    ):

        skipped += 1
        continue

    # --------------------------------------------------------
    # PORTFOLIO RETURN
    # --------------------------------------------------------

    gross = np.mean(
        stock_rets
    )

    # Turnover calculation
    turnover = len(
        portfolio.symmetric_difference(
            previous
        )
    )

    if previous:

        turnover_fraction = (
            turnover
            /
            (
                len(portfolio)
                +
                len(previous)
            )
        )

    else:

        turnover_fraction = 1.0

    net_return = (
        gross
        -
        COST * turnover_fraction
    )

    rets.append(
        net_return
    )

    dates.append(
        next_month
    )

    previous = portfolio


# ============================================================
# RESULTS
# ============================================================

returns = pd.Series(
    rets,
    index=pd.DatetimeIndex(dates)
).sort_index()

if returns.empty:

    raise RuntimeError(
        "No strategy returns produced."
    )


equity = (
    1 + returns
).cumprod()

total = (
    equity.iloc[-1]
    - 1
)

years = (
    len(returns)
    / 12
)

cagr = (
    equity.iloc[-1]
    ** (1 / years)
) - 1

vol = (
    returns.std(ddof=1)
    * np.sqrt(12)
)

sharpe = (
    returns.mean()
    /
    returns.std(ddof=1)
    *
    np.sqrt(12)
)

drawdown = (
    equity
    /
    equity.cummax()
    - 1
)

max_dd = drawdown.min()

winning_months = (
    returns > 0
).mean()


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 60)
print("PLAN 2 — INFORMATION DISCRETENESS")
print("=" * 60)

print(
    "Strategy: 12–2 Momentum → "
    "Top 100 → Lowest ID Top 50 → Top 15"
)

print(
    f"Cost assumption: {COST:.2%}"
)

print(
    f"Total return: {total:.2%}"
)

print(
    f"CAGR: {cagr:.2%}"
)

print(
    f"Annual volatility: {vol:.2%}"
)

print(
    f"Sharpe ratio: {sharpe:.2f}"
)

print(
    f"Maximum drawdown: {max_dd:.2%}"
)

print(
    f"Winning months: "
    f"{winning_months:.2%}"
)

print(
    f"Months tested: "
    f"{len(returns)}"
)

print(
    f"Rebalances: "
    f"{len(returns)}"
)

print(
    f"Skipped months: "
    f"{skipped}"
)


# ============================================================
# SAVE MONTHLY RETURNS
# ============================================================

output = pd.DataFrame(
    {
        "date": returns.index,
        "monthly_return": returns.values,
        "equity": equity.values
    }
)

output.to_csv(
    "plan2_monthly_returns.csv",
    index=False
)

print(
    "Saved: plan2_monthly_returns.csv"
)
