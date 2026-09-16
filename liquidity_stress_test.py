import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PLAN 2 LIQUIDITY STRESS TEST
#
# Purpose:
# Test whether Plan 2 performance survives when stocks with
# insufficient average daily traded value are excluded BEFORE
# portfolio selection.
#
# Plan 2:
#   Historical Nifty 500 membership
#   12-2 month momentum -> Top 100
#   Lowest ID -> Top 50
#   Equal weight 50 stocks
#   Monthly rebalance
#   0.50% transaction cost
#
# Liquidity filter:
#   Prior 100 trading-day average traded value
#   >= threshold
#
# Thresholds:
#   ₹25 lakh/day
#   ₹50 lakh/day
#   ₹1 crore/day
#   ₹5 crore/day
#
# Research only. No trading.
# ============================================================

MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"

TOP_MOMENTUM = 100
TOP_ID = 50
PORTFOLIO_SIZE = 50
TRADING_COST = 0.005

LIQUIDITY_THRESHOLDS = {
    "No filter": 0,
    "₹25L/day": 25_00_000,
    "₹50L/day": 50_00_000,
    "₹1Cr/day": 1_00_00_000,
    "₹5Cr/day": 5_00_00_000,
}

START = "2017-01-01"
END = None


# ------------------------------------------------------------
# Membership
# ------------------------------------------------------------

membership = pd.read_csv(MEMBERSHIP_FILE)
membership["effective_date"] = pd.to_datetime(membership["effective_date"])
membership["symbol"] = (
    membership["symbol"].astype(str).str.strip().str.upper()
)
membership = (
    membership
    .dropna(subset=["effective_date", "symbol"])
    .drop_duplicates(["effective_date", "symbol"])
    .sort_values(["effective_date", "symbol"])
)

membership_dates = sorted(membership["effective_date"].unique())
membership_by_date = {
    d: set(g["symbol"])
    for d, g in membership.groupby("effective_date")
}


def members_at(date):
    valid = [d for d in membership_dates if d <= date]
    if not valid:
        return set()
    return membership_by_date[valid[-1]]


# ------------------------------------------------------------
# Yahoo ticker conversion
# ------------------------------------------------------------

def yahoo_symbol(symbol):
    symbol = str(symbol).strip().upper()
    if symbol.isalpha() and len(symbol) <= 20:
        return symbol + ".NS"
    return None


symbols = sorted(
    {
        yahoo_symbol(s)
        for s in membership["symbol"]
        if yahoo_symbol(s) is not None
    }
)


# ------------------------------------------------------------
# Download close + volume
# ------------------------------------------------------------

download_start = (
    membership["effective_date"].min()
    - pd.DateOffset(years=2)
)

today = pd.Timestamp.today().normalize()
download_end = today + pd.Timedelta(days=1)

print("\nPLAN 2 LIQUIDITY STRESS TEST")
print("=" * 75)
print("Membership records:", len(membership))
print(
    "Membership date range:",
    membership["effective_date"].min().date(),
    "to",
    membership["effective_date"].max().date(),
)
print("Historical symbols:", len(symbols))
print("Price period:", download_start.date(), "to", today.date())


def download_data(tickers, batch_size=25):
    close_frames = []
    volume_frames = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(
            f"Downloading {i + 1}-{min(i + batch_size, len(tickers))}"
        )

        try:
            data = yf.download(
                batch,
                start=download_start.strftime("%Y-%m-%d"),
                end=download_end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
            )

            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                levels = data.columns.get_level_values(0)

                if "Close" not in levels or "Volume" not in levels:
                    continue

                close = data["Close"]
                volume = data["Volume"]

            else:
                if "Close" not in data.columns or "Volume" not in data.columns:
                    continue

                close = data[["Close"]]
                volume = data[["Volume"]]
                close.columns = [batch[0]]
                volume.columns = [batch[0]]

            close_frames.append(close)
            volume_frames.append(volume)

        except Exception as exc:
            print("Batch failed:", str(exc)[:160])

    if not close_frames:
        return pd.DataFrame(), pd.DataFrame()

    close = pd.concat(close_frames, axis=1, sort=True)
    volume = pd.concat(volume_frames, axis=1, sort=True)

    close = close.loc[:, ~close.columns.duplicated()].sort_index()
    volume = volume.loc[:, ~volume.columns.duplicated()].sort_index()

    common = sorted(set(close.columns) & set(volume.columns))

    close = close[common].dropna(axis=1, how="all")
    volume = volume[common]

    return close, volume


close, volume = download_data(symbols)

if close.empty or volume.empty:
    raise SystemExit("No usable price/volume data downloaded.")

print("\nPRICE DATA")
print("=" * 75)
print("Stocks with close data:", close.shape[1])
print("Trading days:", len(close))
print("Last available trading day:", close.index.max().date())


# ------------------------------------------------------------
# Monthly dates
# ------------------------------------------------------------

monthly_close = close.resample("ME").last()

last_completed_month = (
    close.index.max().to_period("M").to_timestamp("M")
)

month_ends = [
    d for d in monthly_close.index
    if d <= last_completed_month
]

first_test_date = (
    membership["effective_date"].min()
    + pd.DateOffset(months=13)
)

month_ends = [
    d for d in month_ends
    if d >= first_test_date
]

# We need a next completed month to calculate the return.
test_months = month_ends[:-1]


# ------------------------------------------------------------
# Liquidity calculation
# ------------------------------------------------------------
#
# IMPORTANT:
# Liquidity is calculated only from data available ON or BEFORE
# the rebalance date. This avoids using future information.
#
# ADV = average of daily Close * Volume over previous 100
# trading observations.
# ------------------------------------------------------------

traded_value = close * volume

liquidity_cache = {}


def average_daily_traded_value(ticker, date, window=100):
    key = (ticker, date)

    if key in liquidity_cache:
        return liquidity_cache[key]

    if ticker not in traded_value.columns:
        liquidity_cache[key] = np.nan
        return np.nan

    s = traded_value[ticker]
    s = s.loc[s.index <= date].dropna()

    if len(s) < window:
        liquidity_cache[key] = np.nan
        return np.nan

    value = s.iloc[-window:].mean()

    if not np.isfinite(value) or value <= 0:
        liquidity_cache[key] = np.nan
        return np.nan

    liquidity_cache[key] = float(value)
    return float(value)


# ------------------------------------------------------------
# Plan 2 signal
# ------------------------------------------------------------

def plan2_signal(ticker, date):
    if ticker not in close.columns:
        return None

    s = close[ticker].dropna()

    start_cut = date - pd.DateOffset(months=12)
    end_cut = date - pd.DateOffset(months=2)

    a = s.loc[s.index <= start_cut]
    b = s.loc[s.index <= end_cut]

    if a.empty or b.empty:
        return None

    p0 = a.iloc[-1]
    p1 = b.iloc[-1]

    if p0 <= 0 or p1 <= 0:
        return None

    momentum = p1 / p0 - 1

    path = s.loc[
        (s.index >= a.index[-1])
        & (s.index <= b.index[-1])
    ]

    if len(path) < 100:
        return None

    daily_returns = path.pct_change().dropna()
    daily_returns = daily_returns[daily_returns != 0]

    if daily_returns.empty:
        return None

    positive_pct = (daily_returns > 0).mean()
    negative_pct = (daily_returns < 0).mean()

    id_value = (
        np.sign(momentum)
        * (negative_pct - positive_pct)
    )

    return momentum, id_value


# ------------------------------------------------------------
# Run one liquidity configuration
# ------------------------------------------------------------

def run_strategy(liquidity_threshold):
    equity = 1.0
    previous = set()

    rows = []
    selection_audit = []

    rebalances = 0
    skipped = 0
    filter_skips = 0

    for i, date in enumerate(test_months):
        next_date = test_months[i + 1]

        candidates = [
            yahoo_symbol(s)
            for s in members_at(date)
            if yahoo_symbol(s) in close.columns
        ]

        signals = {}

        for ticker in candidates:
            signal = plan2_signal(ticker, date)

            if signal is None:
                continue

            momentum, id_value = signal

            adv = average_daily_traded_value(ticker, date)

            if not np.isfinite(adv):
                continue

            signals[ticker] = {
                "momentum": momentum,
                "id": id_value,
                "adv": adv,
            }

        if len(signals) < TOP_MOMENTUM:
            skipped += 1
            continue

        # ----------------------------------------------------
        # Liquidity filter BEFORE momentum ranking
        # ----------------------------------------------------

        if liquidity_threshold > 0:
            liquid = {
                ticker: values
                for ticker, values in signals.items()
                if values["adv"] >= liquidity_threshold
            }
        else:
            liquid = signals

        if len(liquid) < TOP_MOMENTUM:
            skipped += 1
            filter_skips += 1
            continue

        # ----------------------------------------------------
        # Plan 2 ranking
        # ----------------------------------------------------

        top_momentum = sorted(
            liquid.items(),
            key=lambda x: (
                x[1]["momentum"],
                x[0],
            ),
            reverse=True,
        )[:TOP_MOMENTUM]

        # Keep positive momentum, matching Plan 2.
        top_momentum = [
            item for item in top_momentum
            if item[1]["momentum"] > 0
        ]

        if len(top_momentum) < TOP_ID:
            skipped += 1
            continue

        top_id = sorted(
            top_momentum,
            key=lambda x: (
                x[1]["id"],
                x[0],
            ),
        )[:TOP_ID]

        if len(top_id) < PORTFOLIO_SIZE:
            skipped += 1
            continue

        selected = [
            ticker
            for ticker, _ in top_id[:PORTFOLIO_SIZE]
        ]

        # Equal weight.
        weight = 1.0 / len(selected)
        new = {ticker: weight for ticker in selected}

        # Transaction cost.
        all_symbols = set(previous) | set(new)

        turnover = sum(
            abs(new.get(ticker, 0) - previous.get(ticker, 0))
            for ticker in all_symbols
        )

        equity *= 1 - turnover * TRADING_COST

        # Next month's return.
        period_return = 0.0

        for ticker, weight in new.items():
            try:
                p0 = monthly_close.loc[date, ticker]
                p1 = monthly_close.loc[next_date, ticker]

                if (
                    pd.notna(p0)
                    and pd.notna(p1)
                    and p0 > 0
                ):
                    period_return += (
                        weight * (p1 / p0 - 1)
                    )
            except Exception:
                pass

        equity *= 1 + period_return

        rows.append(
            {
                "date": next_date,
                "equity": equity,
                "turnover": turnover,
                "selected_count": len(selected),
            }
        )

        for ticker in selected:
            selection_audit.append(
                {
                    "rebalance_date": date,
                    "symbol": ticker,
                    "momentum_12_2": signals[ticker]["momentum"],
                    "ID": signals[ticker]["id"],
                    "avg_daily_traded_value": signals[ticker]["adv"],
                    "liquidity_threshold": liquidity_threshold,
                }
            )

        previous = new
        rebalances += 1

    if not rows:
        return None, pd.DataFrame()

    result = pd.DataFrame(rows).set_index("date")
    return {
        "equity": result["equity"],
        "rebalances": rebalances,
        "skipped": skipped,
        "filter_skips": filter_skips,
    }, pd.DataFrame(selection_audit)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def calculate_metrics(equity):
    if equity is None or len(equity) < 2:
        return None

    returns = equity.pct_change().dropna()

    if returns.empty:
        return None

    total_return = equity.iloc[-1] / equity.iloc[0] - 1

    years = max(
        (equity.index[-1] - equity.index[0]).days / 365.25,
        1 / 12,
    )

    cagr = (
        equity.iloc[-1] / equity.iloc[0]
    ) ** (1 / years) - 1

    volatility = returns.std() * np.sqrt(12)

    sharpe = (
        returns.mean() * 12 / volatility
        if volatility > 0
        else np.nan
    )

    max_drawdown = (
        equity / equity.cummax() - 1
    ).min()

    winning_months = (
        returns > 0
    ).mean()

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "winning_months": winning_months,
        "months": len(returns),
    }


# ------------------------------------------------------------
# Execute all thresholds
# ------------------------------------------------------------

results = []
all_audits = []

print("\nRUNNING LIQUIDITY STRESS GRID")
print("=" * 75)

for label, threshold in LIQUIDITY_THRESHOLDS.items():

    print(f"\nTesting: {label}")

    run, audit = run_strategy(threshold)

    if run is None:
        print("No usable result.")
        continue

    metrics = calculate_metrics(run["equity"])

    if metrics is None:
        continue

    results.append(
        {
            "liquidity_filter": label,
            "threshold_rupees_per_day": threshold,
            "total_return": metrics["total_return"],
            "cagr": metrics["cagr"],
            "volatility": metrics["volatility"],
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "winning_months": metrics["winning_months"],
            "months": metrics["months"],
            "rebalances": run["rebalances"],
            "skipped": run["skipped"],
            "filter_skips": run["filter_skips"],
        }
    )

    if not audit.empty:
        all_audits.append(audit)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

summary = pd.DataFrame(results)

if summary.empty:
    raise SystemExit("No stress-test results generated.")

display_summary = summary.copy()

for col in [
    "total_return",
    "cagr",
    "volatility",
    "max_drawdown",
    "winning_months",
]:
    display_summary[col] = (
        display_summary[col] * 100
    ).round(2)

display_summary["sharpe"] = display_summary["sharpe"].round(2)

print("\n" + "=" * 75)
print("LIQUIDITY STRESS TEST SUMMARY")
print("=" * 75)
print(display_summary.to_string(index=False))


# ------------------------------------------------------------
# Relative impact versus no-filter baseline
# ------------------------------------------------------------

base = summary.iloc[0]

comparison = summary.copy()

comparison["CAGR_change_vs_base"] = (
    comparison["cagr"] - base["cagr"]
)

comparison["Sharpe_change_vs_base"] = (
    comparison["sharpe"] - base["sharpe"]
)

comparison["MaxDD_change_vs_base"] = (
    comparison["max_drawdown"] - base["max_drawdown"]
)

comparison_display = comparison[
    [
        "liquidity_filter",
        "CAGR_change_vs_base",
        "Sharpe_change_vs_base",
        "MaxDD_change_vs_base",
        "skipped",
        "filter_skips",
    ]
].copy()

for col in [
    "CAGR_change_vs_base",
    "MaxDD_change_vs_base",
]:
    comparison_display[col] *= 100

comparison_display = comparison_display.round(2)

print("\n" + "=" * 75)
print("CHANGE VS NO-FILTER BASE CASE")
print("=" * 75)
print(comparison_display.to_string(index=False))


# ------------------------------------------------------------
# Simple diagnostic — not a trading recommendation
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("DIAGNOSTIC")
print("=" * 75)

base_cagr = base["cagr"]

for _, row in summary.iterrows():

    label = row["liquidity_filter"]

    if label == "No filter":
        print("No filter: baseline.")
        continue

    cagr_change = row["cagr"] - base_cagr

    print(
        f"{label}: "
        f"CAGR change {cagr_change * 100:+.2f} percentage points, "
        f"skipped {int(row['skipped'])} months."
    )

print("\nIMPORTANT:")
print("- This is a stress test, not proof of future returns.")
print("- Liquidity is estimated from historical Close × Volume.")
print("- A 100-day average does not guarantee execution at any price.")
print("- Plan 2 parameters are NOT changed by this script.")
print("- No trading orders are placed.")


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

summary.to_csv(
    "liquidity_stress_results.csv",
    index=False,
)

comparison.to_csv(
    "liquidity_stress_comparison.csv",
    index=False,
)

if all_audits:
    pd.concat(
        all_audits,
        ignore_index=True,
    ).to_csv(
        "liquidity_stress_selection_audit.csv",
        index=False,
    )

print("\nFiles saved:")
print("  liquidity_stress_results.csv")
print("  liquidity_stress_comparison.csv")
print("  liquidity_stress_selection_audit.csv")
print("\nLIQUIDITY STRESS TEST COMPLETE")
