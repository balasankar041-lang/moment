import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PLAN 2 ROBUSTNESS TEST — EXACT BASE IMPLEMENTATION
#
# Base implementation is aligned to the recovered original
# backtest_id.py:
#
#   Top 100 by 12-2 momentum
#   -> lowest ID Top 50
#   -> 50-stock equal-weight portfolio
#   -> month-end close to next month-end close
#   -> original weight-based turnover cost
#
# The base case is validated FIRST.
# The robustness grid is interpreted ONLY after PASS.
#
# Research only. No trading.
# ============================================================

MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"

TOP_MOMENTUM_POOLS = [75, 100, 150]
TOP_ID_POOLS = [30, 50, 75]
PORTFOLIO_SIZES = [30, 40, 50, 60, 75]
COSTS = [0.0025, 0.0050, 0.0075, 0.0100]

BASE_TOP_MOMENTUM = 100
BASE_TOP_ID = 50
BASE_PORTFOLIO = 50
BASE_COST = 0.005

EXPECTED_CAGR = 0.3342
EXPECTED_SHARPE = 1.52
EXPECTED_DD = -0.2765
EXPECTED_WIN = 0.7273

TRAIN_END = pd.Timestamp("2023-12-31")

# ------------------------------------------------------------
# Membership
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Download
# ------------------------------------------------------------

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


prices = (
    download_prices(symbols)
    .dropna(axis=1, how="all")
)

if prices.empty:
    raise SystemExit(
        "No price data downloaded."
    )


# ------------------------------------------------------------
# Completed month
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Signal
# EXACT original implementation
# ------------------------------------------------------------

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
        (s.index >= a.index[-1])
        &
        (s.index <= b.index[-1])
    ]

    if len(path) < 100:
        return None

    dr = (
        path
        .pct_change()
        .dropna()
    )

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


# ------------------------------------------------------------
# Dates
# ------------------------------------------------------------

dates = [
    d for d in monthly.index
    if d <= completed_month
]

dates = [
    d for d in dates
    if d >= (
        membership["effective_date"].min()
        + pd.DateOffset(months=13)
    )
]


# ------------------------------------------------------------
# Run exact strategy
# ------------------------------------------------------------

def run_strategy(
    top_100,
    top_50,
    portfolio_size,
    trading_cost
):

    equity = 1.0

    rows = []

    previous = {}

    rebalance_count = 0
    skipped = 0

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

        if len(signals) < top_100:
            skipped += 1
            continue

        # EXACT original:
        # sort only by momentum
        top100 = sorted(
            signals.items(),
            key=lambda x: x[1][0],
            reverse=True
        )[:top_100]

        # EXACT original:
        # lowest ID from top momentum group
        selected = [
            s
            for s, z in sorted(
                top100,
                key=lambda x: x[1][1]
            )[:top_50]
        ]

        if len(selected) < portfolio_size:
            skipped += 1
            continue

        # For the base strategy portfolio size is 50.
        # For robustness tests, select the first N from
        # the same lowest-ID ranking.
        selected = selected[:portfolio_size]

        w = 1.0 / len(selected)

        new = {
            s: w
            for s in selected
        }

        # EXACT original turnover calculation
        all_s = set(previous) | set(new)

        turnover = sum(
            abs(
                new.get(s, 0)
                -
                previous.get(s, 0)
            )
            for s in all_s
        )

        equity *= (
            1
            -
            turnover * trading_cost
        )

        # EXACT original return calculation:
        # month-end close -> next month-end close
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
                        *
                        (p1 / p0 - 1)
                    )

            except Exception:
                pass

        equity *= (
            1 + period_ret
        )

        rows.append(
            (next_date, equity)
        )

        previous = new
        rebalance_count += 1

    if not rows:
        return None

    eq = pd.Series(
        dict(rows)
    ).sort_index()

    # Same metric methodology as original
    r = (
        eq
        .pct_change()
        .dropna()
    )

    if len(eq) < 2:
        return None

    total = (
        eq.iloc[-1]
        /
        eq.iloc[0]
        - 1
    )

    years = max(
        (
            eq.index[-1]
            -
            eq.index[0]
        ).days
        / 365.25,
        1 / 12
    )

    cagr = (
        eq.iloc[-1]
        /
        eq.iloc[0]
    ) ** (1 / years) - 1

    vol = (
        r.std()
        *
        np.sqrt(12)
    )

    sharpe = (
        r.mean()
        * 12
        /
        vol
        if vol > 0
        else np.nan
    )

    dd = (
        eq
        /
        eq.cummax()
        - 1
    ).min()

    win = (
        r > 0
    ).mean()

    return {
        "top_momentum": top_100,
        "top_id": top_50,
        "portfolio_size": portfolio_size,
        "cost": trading_cost,
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "winning_months": win,
        "months": len(r),
        "rebalances": rebalance_count,
        "skipped": skipped
    }


# ------------------------------------------------------------
# BASE VALIDATION
# ------------------------------------------------------------

print()
print("=" * 80)
print("BASE CASE VALIDATION")
print("=" * 80)

base = run_strategy(
    BASE_TOP_MOMENTUM,
    BASE_TOP_ID,
    BASE_PORTFOLIO,
    BASE_COST
)

if base is None:
    raise SystemExit(
        "BASE CASE FAILED: no results."
    )

print(
    f"CAGR: {base['cagr']:.2%}"
)
print(
    f"Sharpe: {base['sharpe']:.2f}"
)
print(
    f"Max DD: {base['max_drawdown']:.2%}"
)
print(
    f"Winning months: "
    f"{base['winning_months']:.2%}"
)
print(
    f"Months: {base['months']}"
)
print(
    f"Rebalances: {base['rebalances']}"
)
print(
    f"Skipped: {base['skipped']}"
)

# Tight validation:
# CAGR within 2 pp, Sharpe within 0.10,
# DD within 2 pp, win rate within 2 pp.
base_ok = (
    abs(
        base["cagr"]
        -
        EXPECTED_CAGR
    ) <= 0.02
    and
    abs(
        base["sharpe"]
        -
        EXPECTED_SHARPE
    ) <= 0.10
    and
    abs(
        base["max_drawdown"]
        -
        EXPECTED_DD
    ) <= 0.02
    and
    abs(
        base["winning_months"]
        -
        EXPECTED_WIN
    ) <= 0.02
)

print()

if base_ok:

    print("BASE CASE STATUS: PASS")
    print(
        "Base case reproduces the "
        "recorded Plan 2 benchmark closely."
    )

else:

    print("BASE CASE STATUS: FAIL")
    print(
        "STOP: robustness grid will NOT "
        "be interpreted."
    )
    print(
        "The base case does not reproduce "
        "the recorded Plan 2 benchmark."
    )


# ------------------------------------------------------------
# ROBUSTNESS GRID
# ------------------------------------------------------------

results = []

if base_ok:

    print()
    print("=" * 80)
    print("RUNNING ROBUSTNESS GRID")
    print("=" * 80)

    for cost in COSTS:

        for momentum_pool in TOP_MOMENTUM_POOLS:

            for id_pool in TOP_ID_POOLS:

                for portfolio_size in PORTFOLIO_SIZES:

                    if portfolio_size > id_pool:
                        continue

                    result = run_strategy(
                        momentum_pool,
                        id_pool,
                        portfolio_size,
                        cost
                    )

                    if result is not None:
                        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        "plan2_robustness_results_validated.csv",
        index=False
    )

    print()
    print("=" * 80)
    print("BEST 10 CONFIGURATIONS BY SHARPE")
    print("=" * 80)

    print(
        results_df
        .sort_values(
            ["sharpe", "cagr"],
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print()
    print("=" * 80)
    print("MEDIAN RESULTS")
    print("=" * 80)

    print(
        f"Median CAGR: "
        f"{results_df['cagr'].median():.2%}"
    )

    print(
        f"Median Sharpe: "
        f"{results_df['sharpe'].median():.2f}"
    )

    print(
        f"Median Max DD: "
        f"{results_df['max_drawdown'].median():.2%}"
    )

    print(
        f"Median Winning Months: "
        f"{results_df['winning_months'].median():.2%}"
    )

    print()
    print("=" * 80)
    print("ROBUSTNESS CHECK")
    print("=" * 80)

    print(
        f"Positive CAGR: "
        f"{(results_df['cagr'] > 0).mean():.2%}"
    )

    print(
        f"Sharpe >= 1: "
        f"{(results_df['sharpe'] >= 1).mean():.2%}"
    )

    print(
        f"CAGR >= 20%: "
        f"{(results_df['cagr'] >= 0.20).mean():.2%}"
    )

    print()
    print(
        "Saved: "
        "plan2_robustness_results_validated.csv"
    )

else:

    pd.DataFrame(
        [base]
    ).to_csv(
        "plan2_robustness_base_validation_failed.csv",
        index=False
    )

    print()
    print(
        "Saved: "
        "plan2_robustness_base_validation_failed.csv"
    )
