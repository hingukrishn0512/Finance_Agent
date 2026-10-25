import os
import re
import sys
import time
import uuid
import threading
import functools
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# NOTE: TensorFlow, scikit-learn and matplotlib are imported INSIDE the forecast
# functions below, not here. They are heavy, and importing them at startup made the
# server slow to wake up. Now only the forecast tool pays that cost, when it is used.

load_dotenv(override=True)
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)


def timed(fn):
    """Print when each tool starts/ends so you can see where time is spent."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        print(f"[tool] {fn.__name__} started {args} {kwargs}", flush=True)
        try:
            return fn(*args, **kwargs)
        finally:
            print(f"[tool] {fn.__name__} finished in {time.time() - t0:.1f}s", flush=True)
    return wrapper


# --- small cache ---------------------------------------------------------------
# "Analyze X" calls several tools that all ask Yahoo for the same data. Remembering
# each answer for 5 minutes makes those requests faster and avoids rate limits.

CACHE_SECONDS = 300
_cache = {}


def _cached(key, loader):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < CACHE_SECONDS:
        return hit[1]
    value = loader()
    is_empty = value.empty if hasattr(value, "empty") else not value
    if not is_empty:                      # never remember an empty/failed answer
        if len(_cache) > 200:
            _cache.clear()
        _cache[key] = (now, value)
    return value


def get_info(ticker: str) -> dict:
    return _cached(("info", ticker.upper()), lambda: yf.Ticker(ticker).info or {})


def get_history(ticker: str, period: str) -> pd.DataFrame:
    return _cached(("hist", ticker.upper(), period), lambda: yf.Ticker(ticker).history(period=period))


# --- helpers (pure pandas/numpy) ---------------------------------------------

def fmt_num(n):
    """1234567890 -> '1.23B'. Returns 'N/A' for missing values."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    a = abs(n)
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= div:
            return f"{n / div:.2f}{suffix}"
    return f"{n:,.2f}"


def pct(n, digits=2):
    return "N/A" if n is None or (isinstance(n, float) and np.isnan(n)) else f"{n:.{digits}f}%"


def analyze_prices(close: pd.Series) -> dict:
    """Return/volatility/drawdown/moving-average/RSI stats from a close-price series."""
    close = close.dropna()
    out = {"last": float(close.iloc[-1]), "days": len(close)}

    for label, offset in (("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252), ("3y", 756), ("5y", 1260)):
        if len(close) > offset:
            out[f"ret_{label}"] = (close.iloc[-1] / close.iloc[-1 - offset] - 1) * 100

    daily = close.pct_change().dropna()
    out["volatility"] = float(daily.std() * np.sqrt(252) * 100) if len(daily) > 1 else None
    out["max_drawdown"] = float(((close / close.cummax()) - 1).min() * 100)
    out["dma50"] = float(close.tail(50).mean()) if len(close) >= 50 else None
    out["dma200"] = float(close.tail(200).mean()) if len(close) >= 200 else None

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    out["rsi"] = float(rsi.iloc[-1]) if len(rsi.dropna()) else None

    year = close.tail(252)
    out["high_52w"], out["low_52w"] = float(year.max()), float(year.min())
    out["best_day"], out["worst_day"] = float(daily.max() * 100), float(daily.min() * 100)
    return out


def statement_lines(df, wanted, max_years=4):
    """Format selected rows of a yfinance financial statement across recent years."""
    if df is None or df.empty:
        return ["  (not available)"]
    df = df.iloc[:, :max_years]
    lines = []
    for name in wanted:
        if name in df.index:
            parts = [f"{c.year}: {fmt_num(v)}" for c, v in df.loc[name].items()]
            lines.append(f"  {name}: " + " | ".join(parts))
    return lines or ["  (not available)"]


# --- forecasting (LSTM) -------------------------------------------------------

LOOKBACK = 60          # the model looks at the last 60 days to predict the next one
TEST_FRACTION = 0.2    # the last 20% of days are kept aside and never used for training
HORIZON = 7            # days to forecast


def _make_windows(scaled, lookback):
    """Turns a price series into (last 60 days -> next day) training examples."""
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    return np.array(X).reshape(-1, lookback, 1), np.array(y)


def compute_metrics(actual, pred, prev):
    """Accuracy of the model against a naive baseline ("tomorrow = today").
    actual = real price, pred = model's prediction, prev = the previous day's real price."""
    actual, pred, prev = (np.asarray(a, dtype=float).ravel() for a in (actual, pred, prev))
    rmse_model = float(np.sqrt(np.mean((pred - actual) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((prev - actual) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    direction = float(np.mean(np.sign(pred - prev) == np.sign(actual - prev)) * 100)
    improvement = (rmse_naive - rmse_model) / rmse_naive * 100 if rmse_naive else 0.0
    return {
        "test_days": int(len(actual)),
        "rmse_model": rmse_model,
        "rmse_naive": rmse_naive,
        "improvement_vs_naive": float(improvement),   # positive = LSTM better than naive
        "mape": mape,
        "direction_accuracy": direction,
    }


def run_forecast(prices, horizon: int = HORIZON) -> dict:
    """Trains an LSTM on the first 80% of the prices, tests it on the last 20% that it
    never saw, then forecasts the next `horizon` days.
    (The old version measured error on the same data it trained on, which looks much
    better than reality. This one reports honest, held-out numbers.)"""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler

    np.random.seed(42)
    tf.random.set_seed(42)          # same data -> same result every run

    prices = np.asarray(prices, dtype=float).reshape(-1, 1)
    split = int(len(prices) * (1 - TEST_FRACTION))

    scaler = MinMaxScaler()
    scaler.fit(prices[:split])       # learn the scale from training days only (no peeking)
    scaled = scaler.transform(prices)

    X, y = _make_windows(scaled, LOOKBACK)
    target_pos = np.arange(LOOKBACK, len(scaled))   # day each example is predicting
    is_train = target_pos < split
    X_train, y_train = X[is_train], y[is_train]
    X_test = X[~is_train]
    test_pos = target_pos[~is_train]

    model = Sequential([Input(shape=(LOOKBACK, 1)), LSTM(50), Dense(25), Dense(1)])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(
        X_train, y_train, batch_size=16, epochs=30, verbose=0, validation_split=0.1,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    )

    # honest accuracy: predictions on days the model never trained on
    pred = scaler.inverse_transform(model.predict(X_test, verbose=0)).ravel()
    actual = prices[test_pos].ravel()
    prev = prices[test_pos - 1].ravel()
    metrics = compute_metrics(actual, pred, prev)

    # forecast the next `horizon` days, feeding each prediction back in
    window = scaled[-LOOKBACK:, 0].tolist()
    future = []
    for _ in range(horizon):
        x_in = np.array(window[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        nxt = float(model.predict(x_in, verbose=0)[0, 0])
        future.append(nxt)
        window.append(nxt)
    future_prices = scaler.inverse_transform(np.array(future).reshape(-1, 1)).ravel()

    return {
        "metrics": metrics,
        "test_pos": test_pos,
        "test_pred": pred,
        "future_prices": future_prices,
    }


def _save_chart(ticker, dates, prices, res):
    import matplotlib
    matplotlib.use("Agg")  # no popup windows; saves to a file instead
    import matplotlib.pyplot as plt

    future_dates = pd.bdate_range(start=dates[-1] + pd.Timedelta(days=1), periods=len(res["future_prices"]))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, prices, color="blue", label="Actual price")
    ax.plot(dates[res["test_pos"]], res["test_pred"], color="red", linestyle="--",
            label="Model prediction (unseen test days)")
    ax.plot(future_dates, res["future_prices"], color="green", marker="o", label="7-day forecast")
    ax.set_title(f"{ticker.upper()} LSTM forecast (demo, not advice)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = os.path.join(CHART_DIR, f"{ticker.upper()}_{uuid.uuid4().hex[:8]}.png")
    fig.savefig(path)
    plt.close(fig)

    # keep only the 30 newest charts, so the server's disk doesn't fill up
    files = sorted(
        (os.path.join(CHART_DIR, f) for f in os.listdir(CHART_DIR) if f.endswith(".png")),
        key=os.path.getmtime,
    )
    for old in files[:-30]:
        try:
            os.remove(old)
        except OSError:
            pass
    return path, future_dates

# --- end helpers --------------------------------------------------------------


# ----------------------------- TOOLS -----------------------------
# Tools must RETURN strings (not print) so the agent can actually see the result.

@tool
@timed
def get_stock_price(ticker: str) -> str:
    """Return the latest price and daily change of a stock."""
    hist = get_history(ticker, "5d")["Close"].dropna()
    if hist.empty:
        return f"No price data found for {ticker.upper()}. Check the ticker symbol (e.g. RELIANCE.NS)."
    price = float(hist.iloc[-1])
    change = (hist.iloc[-1] / hist.iloc[-2] - 1) * 100 if len(hist) > 1 else None
    return (f"Ticker: {ticker.upper()} | Last close/price: {price:.2f} | "
            f"Daily change: {pct(change)} | As of: {hist.index[-1].date()}")


@tool
@timed
def get_stock_history(ticker: str) -> str:
    """Return the last 10 trading days of open/high/low/close/volume."""
    history = get_history(ticker, "1mo")
    if history.empty:
        return f"No history found for {ticker.upper()}."
    tail = history[["Open", "High", "Low", "Close", "Volume"]].tail(10).round(2)
    return f"Recent history for {ticker.upper()}:\n{tail.to_string()}"


@tool
@timed
def analyze_price_history(ticker: str) -> str:
    """Analyze up to 5 years of price history: returns over 1m/3m/6m/1y/3y/5y,
    annual volatility, max drawdown, 50/200-day moving averages, RSI(14), 52-week range."""
    close = get_history(ticker, "5y")["Close"]
    if close.dropna().empty:
        return f"No history found for {ticker.upper()}."
    a = analyze_prices(close)
    rets = " | ".join(f"{k[4:]}: {pct(v)}" for k, v in a.items() if k.startswith("ret_"))
    trend = ""
    if a["dma50"] and a["dma200"]:
        trend = "50-DMA above 200-DMA (bullish trend)" if a["dma50"] > a["dma200"] else "50-DMA below 200-DMA (bearish trend)"
    rsi = a["rsi"]
    rsi_note = "" if rsi is None else (" (overbought)" if rsi > 70 else " (oversold)" if rsi < 30 else " (neutral)")
    return (
        f"Price analysis for {ticker.upper()} (last close {a['last']:.2f}, {a['days']} trading days used):\n"
        f"Returns: {rets}\n"
        f"Annualized volatility: {pct(a['volatility'])}\n"
        f"Max drawdown: {pct(a['max_drawdown'])}\n"
        f"50-DMA: {fmt_num(a['dma50'])} | 200-DMA: {fmt_num(a['dma200'])} {trend}\n"
        f"RSI(14): {'N/A' if rsi is None else f'{rsi:.1f}'}{rsi_note}\n"
        f"52-week range: {a['low_52w']:.2f} - {a['high_52w']:.2f}\n"
        f"Best day: {pct(a['best_day'])} | Worst day: {pct(a['worst_day'])}"
    )


@tool
@timed
def get_company_info(ticker: str) -> str:
    """Return company name, sector, industry, country, website and business summary."""
    info = get_info(ticker)
    summary = (info.get("longBusinessSummary") or "N/A")[:600]
    return (
        f"Company info for {ticker.upper()}:\n"
        f"Name: {info.get('longName')}\n"
        f"Sector: {info.get('sector')} | Industry: {info.get('industry')}\n"
        f"Country: {info.get('country')} | Employees: {info.get('fullTimeEmployees')}\n"
        f"Website: {info.get('website')}\n"
        f"Summary: {summary}"
    )


@tool
@timed
def get_key_ratios(ticker: str) -> str:
    """Return valuation and quality ratios (P/E, P/B, ROE, margins, debt/equity, dividend yield,
    beta, market cap, 52-week range) and analyst target prices/recommendation."""
    info = get_info(ticker)
    cur = info.get("financialCurrency") or info.get("currency") or ""
    dy = info.get("dividendYield")
    # newer yfinance returns dividendYield already in percent, older returns a fraction
    dy_txt = "N/A" if dy is None else (f"{dy:.2f}%" if dy > 0.5 else f"{dy * 100:.2f}%")
    roe, pm = info.get("returnOnEquity"), info.get("profitMargins")
    return (
        f"Key ratios for {ticker.upper()} (currency: {cur}):\n"
        f"Market cap: {fmt_num(info.get('marketCap'))}\n"
        f"P/E (trailing): {info.get('trailingPE')} | P/E (forward): {info.get('forwardPE')}\n"
        f"Price/Book: {info.get('priceToBook')} | Beta: {info.get('beta')}\n"
        f"ROE: {pct(roe * 100) if roe is not None else 'N/A'} | "
        f"Profit margin: {pct(pm * 100) if pm is not None else 'N/A'}\n"
        f"Debt/Equity: {info.get('debtToEquity')} | Dividend yield: {dy_txt}\n"
        f"52-week range: {info.get('fiftyTwoWeekLow')} - {info.get('fiftyTwoWeekHigh')}\n"
        f"Analyst view: {info.get('recommendationKey')} "
        f"({info.get('numberOfAnalystOpinions')} analysts) | Target mean: {info.get('targetMeanPrice')} "
        f"(low {info.get('targetLowPrice')}, high {info.get('targetHighPrice')})"
    )


@tool
@timed
def get_financials(ticker: str) -> str:
    """Return up to 4 years of annual income statement, balance sheet and cash flow highlights."""
    stock = yf.Ticker(ticker)
    cur = get_info(ticker).get("financialCurrency") or ""
    parts = [f"Annual financials for {ticker.upper()} (currency: {cur}):"]
    parts.append("Income statement:")
    parts += statement_lines(stock.financials, ["Total Revenue", "Gross Profit", "Operating Income", "EBITDA", "Net Income"])
    parts.append("Balance sheet:")
    parts += statement_lines(stock.balance_sheet, ["Total Assets", "Total Debt", "Stockholders Equity", "Cash And Cash Equivalents"])
    parts.append("Cash flow:")
    parts += statement_lines(stock.cashflow, ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure"])
    return "\n".join(parts)


@tool
@timed
def compare_stocks(tickers: str) -> str:
    """Compare 2-4 stocks side by side. Pass tickers separated by commas, e.g. 'TCS.NS,INFY.NS'."""
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()][:4]
    if len(symbols) < 2:
        return "Please provide at least two tickers separated by commas."
    rows = []
    for sym in symbols:
        try:
            close = get_history(sym, "1y")["Close"]
            if close.dropna().empty:
                rows.append(f"{sym}: no data")
                continue
            a = analyze_prices(close)
            try:
                info = get_info(sym)
                pe, mcap = info.get("trailingPE"), info.get("marketCap")
            except Exception:
                pe, mcap = None, None
            rows.append(
                f"{sym}: price {a['last']:.2f} | 1y return {pct(a.get('ret_1y'))} | "
                f"volatility {pct(a['volatility'])} | max drawdown {pct(a['max_drawdown'])} | "
                f"P/E {pe} | market cap {fmt_num(mcap)}"
            )
        except Exception as e:
            rows.append(f"{sym}: error ({e})")
    return "Comparison (1-year basis):\n" + "\n".join(rows)


@tool
@timed
def get_financial_news(ticker: str) -> str:
    """Return recent news headlines about a company."""
    if not NEWS_API_KEY:
        return "News is unavailable: NEWS_API_KEY is not set in .env."
    base = ticker.split(".")[0]
    query = base
    try:
        name = get_info(ticker).get("shortName")
        if name:
            query = f'"{name}"'  # company name gives far better results than a ticker
    except Exception:
        pass
    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={"q": query, "language": "en", "sortBy": "publishedAt",
                "pageSize": 5, "apiKey": NEWS_API_KEY},
        timeout=10,
    ).json()
    if resp.get("status") != "ok":
        return f"News request failed: {resp.get('message', 'unknown error')}"
    articles = resp.get("articles", [])
    if not articles:
        return f"No recent news found for {ticker.upper()}."
    lines = [f"Latest news for {ticker.upper()} (search: {query}):"]
    for i, a in enumerate(articles, 1):
        lines.append(f"{i}. {a.get('title')} ({a.get('source', {}).get('name')}, {a.get('publishedAt')})")
    return "\n".join(lines)


@tool
@timed
def get_graph(ticker: str) -> str:
    """Train a small LSTM on 2 years of prices, test it on recent days it never saw, save a
    chart of actual vs predicted prices, and forecast the next 7 days. Returns the forecast,
    chart path and honest held-out accuracy (RMSE vs a naive baseline, MAPE, direction accuracy)."""
    stock_history = get_history(ticker, "2y")
    close = stock_history["Close"].dropna()
    if len(close) < 200:
        return f"Not enough price history for {ticker.upper()} to train and test a forecast."

    prices = close.values
    res = run_forecast(prices)
    chart_path, future_dates = _save_chart(ticker, close.index, prices, res)
    m = res["metrics"]

    if m["improvement_vs_naive"] > 5:
        verdict = "The LSTM beats the naive baseline on unseen days."
    elif m["improvement_vs_naive"] > -5:
        verdict = "The LSTM is about as accurate as the naive baseline, so it adds little."
    else:
        verdict = "The LSTM is WORSE than the naive baseline, so treat this forecast as a demo only."

    forecast_lines = [f"Day {i + 1} ({d.date()}): {p:.2f}"
                      for i, (d, p) in enumerate(zip(future_dates, res["future_prices"]))]
    return (
        f"Chart saved at: {chart_path}\n"
        "7-day forecast (rough estimate):\n" + "\n".join(forecast_lines) + "\n"
        f"Held-out test on the last {m['test_days']} trading days the model never trained on:\n"
        f"LSTM RMSE {m['rmse_model']:.2f} vs naive 'tomorrow = today' RMSE {m['rmse_naive']:.2f} "
        f"({m['improvement_vs_naive']:+.1f}% vs naive; positive = LSTM better) | "
        f"MAPE {m['mape']:.2f}% | direction accuracy {m['direction_accuracy']:.1f}% (50% = coin flip).\n"
        f"{verdict}"
    )


# ----------------------------- AGENT -----------------------------

# Model name comes from .env so future deprecations only need a config change.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Use ONE explicit key variable so an old key can't be picked up silently.
os.environ.pop("GOOGLE_API_KEY", None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
print("Using Gemini key ending in:", GEMINI_API_KEY[-4:])

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL, temperature=0, google_api_key=GEMINI_API_KEY,
    timeout=60, max_retries=1,
)

SYSTEM_PROMPT = """You are a careful stock research assistant.

Rules:
1. Every price, ratio, financial figure, return and news item MUST come from a tool result in this
   conversation. Never state such numbers from memory. If a tool returns nothing, say "data not available".
2. Indian companies need Yahoo tickers: .NS for NSE (RELIANCE.NS, TCS.NS), .BO for BSE. Convert company
   names to tickers yourself. If the ticker is ambiguous, ask the user.
3. For broad requests ("analyze X", "is X a good company"), call several tools: get_company_info,
   get_key_ratios, get_financials, analyze_price_history, get_financial_news. Then summarize in short
   sections (Business, Valuation, Financial health, Price trend, News, Risks).
4. For comparisons, use compare_stocks, then add get_key_ratios if more depth is needed.
5. State the currency and the date of the data when you have it.
6. For forecasts, use get_graph. Give the 7 numbers, mention the chart path, and report the held-out
   test results from the tool (RMSE vs the naive baseline, MAPE, direction accuracy) exactly as given,
   including its verdict. Say plainly that it is a weak demo, not a prediction to trade on.
7. Give balanced views (strengths and risks). Do not tell the user to buy or sell.
   End full analyses with: "This is not financial advice."
Keep answers clear and concise."""

agent = create_agent(
    model=llm,
    tools=[get_stock_price, get_stock_history, analyze_price_history, get_company_info,
           get_key_ratios, get_financials, compare_stocks, get_financial_news, get_graph],
    system_prompt=SYSTEM_PROMPT,
)

_agent_lock = threading.Lock()  # TensorFlow training isn't safe to run concurrently


def _to_text(content) -> str:
    """Gemini may return a list of content blocks instead of a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def run_agent(user_input: str, history: list | None = None) -> dict:
    """Single entry point for FastAPI / CLI.
    history: earlier [{"role": "user"|"assistant", "content": str}] messages for follow-ups.
    Returns {"reply": str, "charts": [file paths]}."""
    messages = list(history or []) + [{"role": "user", "content": user_input}]

    if not _agent_lock.acquire(timeout=120):
        raise RuntimeError("Another request is still running. Please try again shortly.")
    t0 = time.time()
    print(f"[agent] question: {user_input}", flush=True)
    try:
        result = agent.invoke({"messages": messages}, config={"recursion_limit": 25})
    finally:
        _agent_lock.release()
        print(f"[agent] done in {time.time() - t0:.1f}s", flush=True)

    reply = _to_text(result["messages"][-1].content)

    charts = []
    for m in result["messages"][len(messages):]:  # only messages from this turn
        if getattr(m, "type", "") == "tool":
            for path in re.findall(r"charts[/\\][\w\-.]+\.png", _to_text(m.content)):
                charts.append(path.replace("\\", "/"))
    return {"reply": reply, "charts": charts}


# ----------------------------- EVALUATION -------------------------
# Run:  python stock_agent.py --eval            (uses the default tickers)
#       python stock_agent.py --eval TCS.NS AAPL (or your own tickers)
# Prints honest held-out results you can quote on your resume (only if they are good).

DEFAULT_EVAL_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "MSFT"]


def evaluate_many(tickers=None):
    tickers = tickers or DEFAULT_EVAL_TICKERS
    rows = []
    print(f"{'Ticker':<14}{'Test days':>10}{'LSTM RMSE':>11}{'Naive RMSE':>12}{'vs naive':>10}{'MAPE':>8}{'Direction':>11}")
    for t in tickers:
        close = get_history(t, "2y")["Close"].dropna()
        if len(close) < 200:
            print(f"{t:<14} skipped (not enough data)")
            continue
        m = run_forecast(close.values)["metrics"]
        rows.append(m)
        print(f"{t:<14}{m['test_days']:>10}{m['rmse_model']:>11.2f}{m['rmse_naive']:>12.2f}"
              f"{m['improvement_vs_naive']:>9.1f}%{m['mape']:>7.2f}%{m['direction_accuracy']:>10.1f}%")
    if rows:
        avg = lambda k: float(np.mean([r[k] for r in rows]))
        better = sum(r["improvement_vs_naive"] > 0 for r in rows)
        print(f"\nAverage over {len(rows)} stocks: {avg('improvement_vs_naive'):+.1f}% RMSE vs naive baseline | "
              f"MAPE {avg('mape'):.2f}% | direction accuracy {avg('direction_accuracy'):.1f}%")
        print(f"LSTM beat the naive baseline on {better} of {len(rows)} stocks.")


# ----------------------------- CLI -------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        evaluate_many(sys.argv[2:] or None)
        sys.exit(0)

    print("enter 'exit' for exiting the loop")
    chat_history = []
    while True:
        user_input = input("enter the stock name : ").strip()

        if not user_input:
            print("please enter the valid statement..")
            continue
        if user_input.lower() == "exit":
            print("signing off...")
            break

        try:
            out = run_agent(user_input, chat_history)
            print(out["reply"])
            if out["charts"]:
                print("Charts:", out["charts"])
            chat_history += [{"role": "user", "content": user_input},
                             {"role": "assistant", "content": out["reply"]}]
            chat_history = chat_history[-10:]
        except Exception as e:
            print(f"Error: {e}")