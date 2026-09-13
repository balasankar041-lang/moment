import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"

TOP_N = 15

W3 = 0.20
W6 = 0.30
W12 = 0.50

TRADING_COST = 0.005   # 0.50% of turnover

MIN_HISTORY_MONTHS = 36
MIN_VOL_DAYS = 200


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

membership = membership.dropna(
    subset=["effective_date", "symbol"]
)

membership = membership.sort_values(
    ["effective_date", "symbol"]
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


# ============================================================
# CONVERT NSE SYMBOL TO YAHOO SYMBOL
# ============================================================

def yahoo_symbol(symbol):
    """
    Basic NSE -> Yahoo Finance conversion.

    Historical ticker changes are not automatically guessed.
    """
    if not symbol:
        return None

    symbol = str(symbol).strip().upper()

    # Remove common unwanted characters
    if not symbol.isalpha():
        return None

    if len(symbol) > 20:
        return None

    return symbol + ".NS"


# ============================================================
# GET HISTORICAL MEMBERS FOR A DATE
# ============================================================

def members_at_date(date):
    """
    Use the latest available historical Nifty 500 snapshot
    on or before the rebalance date.
    """

    available = membership[
        membership["effective_date"] <= date
    ]

    if available.empty:
        return []

    latest_date = available["effective_date"].max()

    symbols = available.loc[
        available["effective_date"] == latest_date,
        "symbol"
    ].tolist()

    result = []

    for symbol in symbols:
        yahoo = yahoo_symbol(symbol)

        if yahoo:
            result.append(yahoo)

    return sorted(set(result))


# ============================================================
# GET DATE RANGE
# ============================================================

latest_membership_date = membership["effective_date"].max()

download_start = (
    membership["effective_date"].min()
    - pd.DateOffset(years=2)
)

download_end = (
    pd.Timestamp.today()
    + pd.Timedelta(days=1)
)

print("\nDownloading price data...")
print(
    "Price period:",
    download_start.date(),
    "to",
    download_end.date()
)


# ============================================================
# COLLECT ALL HISTORICAL SYMBOLS
# ============================================================

all_symbols = sorted(
    set(
        yahoo_symbol(s)
        for s in membership["symbol"]
        if yahoo_symbol(s) is not None
    )
)

print("Historical symbols:", len(all_symbols))


# ============================================================
# DOWNLOAD PRICES IN BATCHES
# ============================================================

def download_prices(symbols, batch_size=50):

    frames = []

    for i in range(0, len(symbols), batch_size):

        batch = symbols[i:i + batch_size]

        print(
            f"Downloading {i + 1}-{min(i + batch_size, len(symbols))}"
        )

        try:
            data = yf.download(
                batch,
                start=download_start.strftime("%Y-%m-%d"),
                end=download_end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True
            )

            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):

                if "Close" in data.columns.levels[0]:
                    close = data["Close"]
                else:
                    continue

            else:
                if "Close" not in data.columns:
                    continue

                close = data[["Close"]]
                close.columns = [batch[0]]

            frames.append(close)

        except Exception as e:
            print("Batch error:", e)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)

    prices = prices.loc[
        :,
        ~prices.columns.duplicated()
    ]

    prices = prices.sort_index()

    return prices


prices = download_prices(all_symbols)

print("\nPRICE DATA")
print("==========")
print("Stocks with data:", prices.shape[1])
print(
    "Trading days:",
    len(prices)
)


# ============================================================
# MONTH-END PRICES
# ============================================================

monthly_prices = prices.resample("ME").last()

monthly_returns = monthly_prices.pct_change()


# ============================================================
# MOMENTUM SCORE
# ============================================================

def momentum_score(symbol, date):

    try:

        daily = prices[symbol].dropna()

        if daily.empty:
            return None

        # Need approximately 12 months of daily data
        past_12m = daily.loc[
            date - pd.DateOffset(months=12):date
        ]

        if len(past_12m) < MIN_VOL_DAYS:
            return None

        # Need monthly prices
        if date not in monthly_prices.index:
            return None

        current_price = monthly_prices.loc[
            date, symbol
        ]

        if pd.isna(current_price) or current_price <= 0:
            return None

        # 3 month
        d3 = date - pd.DateOffset(months=3)

        # 6 month
        d6 = date - pd.DateOffset(months=6)

        # 12 month
        d12 = date - pd.DateOffset(months=12)

        previous_3 = monthly_prices.loc[
            monthly_prices.index <= d3, symbol
        ].dropna()

        previous_6 = monthly_prices.loc[
            monthly_prices.index <= d6, symbol
        ].dropna()

        previous_12 = monthly_prices.loc[
            monthly_prices.index <= d12, symbol
        ].dropna()

        if (
            previous_3.empty
            or previous_6.empty
            or previous_12.empty
        ):
            return None

        p3 = previous_3.iloc[-1]
        p6 = previous_6.iloc[-1]
        p12 = previous_12.iloc[-1]

        if min(p3, p6, p12) <= 0:
            return None

        r3 = current_price / p3 - 1
        r6 = current_price / p6 - 1
        r12 = current_price / p12 - 1

        # Annualized volatility
        daily_returns = past_12m.pct_change().dropna()

        if len(daily_returns) < MIN_VOL_DAYS - 1:
            return None

        volatility = (
            daily_returns.std() * np.sqrt(252)
        )

        if (
            pd.isna(volatility)
            or volatility <= 0
        ):
            return None

        # Composite momentum
        momentum = (
            W3 * r3
            + W6 * r6
            + W12 * r12
        )

        # Positive momentum only
        if momentum <= 0:
            return None

        # Risk-adjusted momentum
        score = momentum / volatility

        return score

    except Exception:
        return None


# ============================================================
# BACKTEST
# ============================================================

rebalance_dates = monthly_prices.index

portfolio_value = 1.0

portfolio_values = []

previous_weights = {}

trade_count = 0

start_date = (
    rebalance_dates[0]
    + pd.DateOffset(months=MIN_HISTORY_MONTHS)
)

rebalance_dates = [
    d for d in rebalance_dates
    if d >= start_date
]


print("\nBACKTEST")
print("========")
print(
    "Rebalance dates:",
    len(rebalance_dates)
)

for date in rebalance_dates:

    historical_members = members_at_date(date)

    if not historical_members:
        continue

    candidates = [
        s for s in historical_members
        if s in prices.columns
    ]

    scores = {}

    for symbol in candidates:

        score = momentum_score(
            symbol,
            date
        )

        if score is not None:
            scores[symbol] = score

    if len(scores) < TOP_N:
        continue

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    selected = [
        symbol
        for symbol, score in ranked[:TOP_N]
    ]

    # Equal weight
    new_weight = 1.0 / len(selected)

    new_weights = {
        symbol: new_weight
        for symbol in selected
    }

    # ========================================================
    # TURNOVER
    # ========================================================

    all_symbols_weights = set(
        previous_weights
    ) | set(new_weights)

    turnover = 0.0

    for symbol in all_symbols_weights:

        old_w = previous_weights.get(
            symbol, 0.0
        )

        new_w = new_weights.get(
            symbol, 0.0
        )

        turnover += abs(
            new_w - old_w
        )

    # Half-turnover convention:
    # turnover is already sum(abs changes).
    transaction_cost = (
        turnover * TRADING_COST
    )

    portfolio_value *= (
        1 - transaction_cost
    )

    # ========================================================
    # HOLD UNTIL NEXT MONTH
    # ========================================================

    current_index = rebalance_dates.index(date)

    if current_index + 1 < len(rebalance_dates):

        next_date = rebalance_dates[
            current_index + 1
        ]

        period_return = 0.0

        for symbol, weight in new_weights.items():

            if (
                symbol not in monthly_prices.columns
            ):
                continue

            try:

                p0 = monthly_prices.loc[
                    date, symbol
                ]

                p1 = monthly_prices.loc[
                    next_date, symbol
                ]

                if (
                    pd.isna(p0)
                    or pd.isna(p1)
                    or p0 <= 0
                ):
                    continue

                stock_return = (
                    p1 / p0 - 1
                )

                period_return += (
                    weight * stock_return
                )

            except Exception:
                continue

        portfolio_value *= (
            1 + period_return
        )

        portfolio_values.append(
            (
                next_date,
                portfolio_value
            )
        )

    previous_weights = new_weights

    trade_count += 1


# ============================================================
# RESULTS
# ============================================================

if not portfolio_values:

    print("\nNo backtest results generated.")
    raise SystemExit


equity = pd.Series(
    dict(portfolio_values)
).sort_index()

monthly_equity_returns = equity.pct_change().dropna()

start_value = equity.iloc[0]
end_value = equity.iloc[-1]

years = (
    equity.index[-1]
    - equity.index[0]
).days / 365.25

if years <= 0:
    years = 1

total_return = (
    end_value / start_value - 1
)

cagr = (
    (end_value / start_value)
    ** (1 / years)
    - 1
)

annual_volatility = (
    monthly_equity_returns.std()
    * np.sqrt(12)
)

if annual_volatility > 0:

    sharpe = (
        monthly_equity_returns.mean()
        * 12
        / annual_volatility
    )

else:
    sharpe = np.nan

rolling_max = equity.cummax()

drawdown = (
    equity / rolling_max - 1
)

max_drawdown = drawdown.min()

winning_months = (
    (monthly_equity_returns > 0).mean()
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("NIFTY 500 HISTORICAL-MEMBERSHIP")
print("RISK-ADJUSTED TOP-15 BACKTEST")
print("======================================")

print(
    "Backtest period:",
    equity.index[0].date(),
    "to",
    equity.index[-1].date()
)

print(
    "Top stocks:",
    TOP_N
)

print(
    "Momentum:",
    "20% 3M + 30% 6M + 50% 12M"
)

print(
    "Risk adjustment:",
    "Momentum / annualized volatility"
)

print(
    "Trading cost:",
    f"{TRADING_COST * 100:.2f}%"
)

print(
    "Rebalances:",
    trade_count
)

print(
    f"Total return: {total_return * 100:.2f}%"
)

print(
    f"CAGR: {cagr * 100:.2f}%"
)

print(
    f"Annual volatility: {annual_volatility * 100:.2f}%"
)

print(
    f"Sharpe ratio: {sharpe:.2f}"
)

print(
    f"Maximum drawdown: {max_drawdown * 100:.2f}%"
)

print(
    f"Winning months: {winning_months * 100:.2f}%"
)

print(
    f"Months tested: {len(monthly_equity_returns)}"
)

print("\nBACKTEST COMPLETE")
