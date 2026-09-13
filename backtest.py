import yfinance as yf
import pandas as pd
import numpy as np

TOP_N = 15
YEARS = 10
COSTS = [0.0015, 0.0030, 0.0050, 0.0100]
MIN_HISTORY_MONTHS = 36

print("NIFTY 500 TOP-15 COST STRESS TEST")
print("=================================")

# -----------------------------
# NIFTY 500
# -----------------------------
url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

universe = pd.read_csv(
    url,
    storage_options={"User-Agent": "Mozilla/5.0"}
)

symbols = (
    universe["Symbol"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.upper()
)

symbols = [
    s + ".NS"
    for s in symbols
    if s.isalpha() and len(s) <= 15
]

symbols = list(dict.fromkeys(symbols))

print(f"Universe candidates: {len(symbols)}")
print(f"Portfolio: Top {TOP_N}")
print("Downloading data...")

# -----------------------------
# DOWNLOAD
# -----------------------------
all_data = []
BATCH_SIZE = 50

for start in range(0, len(symbols), BATCH_SIZE):

    batch = symbols[start:start + BATCH_SIZE]

    print(
        f"Batch {start + 1}-"
        f"{min(start + BATCH_SIZE, len(symbols))}"
    )

    try:
        data = yf.download(
            batch,
            period=f"{YEARS}y",
            auto_adjust=True,
            progress=False,
            threads=True
        )

        if data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):

            if "Close" not in data.columns.levels[0]:
                continue

            data = data["Close"]

        else:

            if "Close" not in data.columns:
                continue

            data = data[["Close"]]
            data.columns = [batch[0]]

        all_data.append(data)

    except Exception as e:
        print("Batch error:", e)

if not all_data:
    raise RuntimeError("No price data downloaded.")

prices = pd.concat(all_data, axis=1)

prices = prices.loc[
    :,
    ~prices.columns.duplicated()
]

prices = prices.dropna(
    axis=1,
    how="all"
)

prices = prices.ffill()

print(
    f"Stocks with usable data: "
    f"{len(prices.columns)}"
)

if len(prices.columns) < 450:
    raise RuntimeError(
        "Too few stocks downloaded."
    )

monthly = prices.resample("ME").last()

# -----------------------------
# RUN STRATEGY ONCE
# -----------------------------
raw_returns = []
previous_stocks = set()

for i in range(MIN_HISTORY_MONTHS, len(monthly) - 1):

    current = monthly.iloc[i]

    p3 = monthly.iloc[i - 3]
    p6 = monthly.iloc[i - 6]
    p12 = monthly.iloc[i - 12]

    ret3 = current / p3 - 1
    ret6 = current / p6 - 1
    ret12 = current / p12 - 1

    daily = prices.loc[
        monthly.index[i - 12]:
        monthly.index[i]
    ]

    valid_days = daily.count()

    eligible = valid_days[
        valid_days >= 200
    ].index

    if len(eligible) < TOP_N:
        continue

    daily_returns = daily[eligible].pct_change()

    vol = (
        daily_returns.std()
        * np.sqrt(252)
    )

    score = (
        0.20 * ret3 +
        0.30 * ret6 +
        0.50 * ret12
    )

    score = score / vol

    score = score.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    score = score[score > 0]

    if len(score) < TOP_N:
        continue

    selected = score.nlargest(TOP_N).index

    next_month = monthly.iloc[i + 1]

    stock_returns = (
        next_month[selected]
        / current[selected]
        - 1
    ).dropna()

    if len(stock_returns) < 10:
        continue

    portfolio_return = stock_returns.mean()

    current_stocks = set(
        stock_returns.index
    )

    turnover = 0.0

    if previous_stocks:

        changed = len(
            current_stocks.symmetric_difference(
                previous_stocks
            )
        )

        turnover = (
            changed / (2 * TOP_N)
        )

    raw_returns.append({
        "Date": monthly.index[i + 1],
        "GrossReturn": portfolio_return,
        "Turnover": turnover
    })

    previous_stocks = current_stocks

raw = pd.DataFrame(raw_returns)

if raw.empty:
    raise RuntimeError(
        "Backtest produced no results."
    )

raw = raw.set_index("Date")

# -----------------------------
# COST STRESS TEST
# -----------------------------
results = []

print()
print("================================")
print("TRANSACTION COST STRESS TEST")
print("================================")

for cost in COSTS:

    returns = (
        raw["GrossReturn"]
        - raw["Turnover"] * cost
    )

    equity = (
        1 + returns
    ).cumprod()

    years_tested = (
        equity.index[-1]
        - equity.index[0]
    ).days / 365.25

    cagr = (
        equity.iloc[-1]
        ** (1 / years_tested)
    ) - 1

    volatility = (
        returns.std()
        * np.sqrt(12)
    )

    sharpe = (
        returns.mean()
        / returns.std()
    ) * np.sqrt(12)

    drawdown = (
        equity /
        equity.cummax()
    ) - 1

    max_drawdown = drawdown.min()

    win_rate = (
        returns > 0
    ).mean()

    total_return = (
        equity.iloc[-1] - 1
    )

    results.append({
        "Trading cost": cost,
        "Total return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max drawdown": max_drawdown,
        "Winning months": win_rate
    })

# -----------------------------
# PRINT
# -----------------------------
table = pd.DataFrame(results)

for _, row in table.iterrows():

    print()
    print(
        f"Trading cost: "
        f"{row['Trading cost']:.2%}"
    )

    print(
        f"Total return: "
        f"{row['Total return']:.2%}"
    )

    print(
        f"CAGR: "
        f"{row['CAGR']:.2%}"
    )

    print(
        f"Annual volatility: "
        f"{row['Volatility']:.2%}"
    )

    print(
        f"Sharpe ratio: "
        f"{row['Sharpe']:.2f}"
    )

    print(
        f"Maximum drawdown: "
        f"{row['Max drawdown']:.2%}"
    )

    print(
        f"Winning months: "
        f"{row['Winning months']:.2%}"
    )

# -----------------------------
# SAVE
# -----------------------------
table.to_csv(
    "transaction_cost_stress_test.csv",
    index=False
)

raw.to_csv(
    "raw_strategy_returns.csv"
)

print()
print("Saved:")
print("transaction_cost_stress_test.csv")
print("raw_strategy_returns.csv")
print("COST STRESS TEST COMPLETE")
