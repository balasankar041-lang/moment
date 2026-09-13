import yfinance as yf
import pandas as pd
import numpy as np

TOP_N = 15
YEARS = 10
TRADING_COST = 0.005
MIN_HISTORY_MONTHS = 36

TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"

print("NIFTY 500 OUT-OF-SAMPLE BACKTEST")
print("================================")
print("Training: 2019-2023")
print("Testing : 2024-2026")
print(f"Portfolio: Top {TOP_N}")
print(f"Trading cost: {TRADING_COST:.2%}")

# -----------------------------
# NIFTY 500 UNIVERSE
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
print("Downloading data...")

# -----------------------------
# DOWNLOAD IN BATCHES
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
# STRATEGY FUNCTION
# -----------------------------
def run_backtest(start_date, end_date):

    data = monthly.loc[
        :end_date
    ]

    returns = []
    previous_stocks = set()

    for i in range(
        MIN_HISTORY_MONTHS,
        len(data) - 1
    ):

        current_date = data.index[i]

        if current_date < pd.Timestamp(start_date):
            continue

        current = data.iloc[i]

        p3 = data.iloc[i - 3]
        p6 = data.iloc[i - 6]
        p12 = data.iloc[i - 12]

        ret3 = current / p3 - 1
        ret6 = current / p6 - 1
        ret12 = current / p12 - 1

        daily = prices.loc[
            data.index[i - 12]:
            data.index[i]
        ]

        valid_days = daily.count()

        eligible = valid_days[
            valid_days >= 200
        ].index

        if len(eligible) < TOP_N:
            continue

        daily_returns = (
            daily[eligible].pct_change()
        )

        vol = (
            daily_returns.std()
            * np.sqrt(252)
        )

        # Fixed strategy parameters
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

        selected = score.nlargest(
            TOP_N
        ).index

        next_month = data.iloc[i + 1]

        stock_returns = (
            next_month[selected]
            / current[selected]
            - 1
        ).dropna()

        if len(stock_returns) < 10:
            continue

        portfolio_return = (
            stock_returns.mean()
        )

        current_stocks = set(
            stock_returns.index
        )

        turnover = 0

        if previous_stocks:

            changed = len(
                current_stocks.symmetric_difference(
                    previous_stocks
                )
            )

            turnover = (
                changed /
                (2 * TOP_N)
            )

            portfolio_return -= (
                turnover *
                TRADING_COST
            )

        returns.append({
            "Date": data.index[i + 1],
            "Return": portfolio_return
        })

        previous_stocks = current_stocks

    result = pd.DataFrame(returns)

    if result.empty:
        return None

    result = result.set_index("Date")

    # Keep only requested period
    result = result[
        result.index >=
        pd.Timestamp(start_date)
    ]

    result = result[
        result.index <=
        pd.Timestamp(end_date)
    ]

    if result.empty:
        return None

    equity = (
        1 + result["Return"]
    ).cumprod()

    years = (
        result.index[-1]
        - result.index[0]
    ).days / 365.25

    cagr = (
        equity.iloc[-1]
        ** (1 / years)
    ) - 1

    volatility = (
        result["Return"].std()
        * np.sqrt(12)
    )

    sharpe = (
        result["Return"].mean()
        / result["Return"].std()
    ) * np.sqrt(12)

    drawdown = (
        equity /
        equity.cummax()
    ) - 1

    max_drawdown = drawdown.min()

    win_rate = (
        result["Return"] > 0
    ).mean()

    total_return = (
        equity.iloc[-1] - 1
    )

    return {
        "result": result,
        "Total return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max drawdown": max_drawdown,
        "Winning months": win_rate
    }

# -----------------------------
# TRAINING PERIOD
# -----------------------------
print()
print("Running training period...")

train = run_backtest(
    "2019-10-31",
    TRAIN_END
)

# -----------------------------
# TEST PERIOD
# -----------------------------
print()
print("Running unseen test period...")

test = run_backtest(
    TEST_START,
    "2026-12-31"
)

if train is None:
    raise RuntimeError(
        "Training period produced no results."
    )

if test is None:
    raise RuntimeError(
        "Test period produced no results."
    )

# -----------------------------
# PRINT RESULTS
# -----------------------------
print()
print("================================")
print("TRAINING RESULT")
print("================================")

print(
    f"Period: "
    f"{train['result'].index[0].date()} "
    f"to "
    f"{train['result'].index[-1].date()}"
)

print(
    f"Total return: "
    f"{train['Total return']:.2%}"
)

print(
    f"CAGR: "
    f"{train['CAGR']:.2%}"
)

print(
    f"Annual volatility: "
    f"{train['Volatility']:.2%}"
)

print(
    f"Sharpe ratio: "
    f"{train['Sharpe']:.2f}"
)

print(
    f"Maximum drawdown: "
    f"{train['Max drawdown']:.2%}"
)

print(
    f"Winning months: "
    f"{train['Winning months']:.2%}"
)

print()
print("================================")
print("UNSEEN TEST RESULT")
print("================================")

print(
    f"Period: "
    f"{test['result'].index[0].date()} "
    f"to "
    f"{test['result'].index[-1].date()}"
)

print(
    f"Total return: "
    f"{test['Total return']:.2%}"
)

print(
    f"CAGR: "
    f"{test['CAGR']:.2%}"
)

print(
    f"Annual volatility: "
    f"{test['Volatility']:.2%}"
)

print(
    f"Sharpe ratio: "
    f"{test['Sharpe']:.2f}"
)

print(
    f"Maximum drawdown: "
    f"{test['Max drawdown']:.2%}"
)

print(
    f"Winning months: "
    f"{test['Winning months']:.2%}"
)

# -----------------------------
# COMPARISON
# -----------------------------
print()
print("================================")
print("TRAINING vs TEST")
print("================================")

print(
    f"Training CAGR: "
    f"{train['CAGR']:.2%}"
)

print(
    f"Test CAGR: "
    f"{test['CAGR']:.2%}"
)

print(
    f"Training Sharpe: "
    f"{train['Sharpe']:.2f}"
)

print(
    f"Test Sharpe: "
    f"{test['Sharpe']:.2f}"
)

print(
    f"Training Max DD: "
    f"{train['Max drawdown']:.2%}"
)

print(
    f"Test Max DD: "
    f"{test['Max drawdown']:.2%}"
)

# -----------------------------
# SAVE
# -----------------------------
train["result"].to_csv(
    "training_results.csv"
)

test["result"].to_csv(
    "out_of_sample_results.csv"
)

print()
print("Saved:")
print("training_results.csv")
print("out_of_sample_results.csv")
print("OUT-OF-SAMPLE TEST COMPLETE")
