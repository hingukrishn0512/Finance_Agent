import os
import re
import time
import uuid
import threading
import functools
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib
matplotlib.use("Agg")  # no popup windows; saves to file instead (works for FastAPI too)
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

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

# --- end helpers --------------------------------------------------------------


# ----------------------------- TOOLS -----------------------------
# Tools must RETURN strings (not print) so the agent can actually see the result.

@tool
@timed
def get_stock_price(ticker: str) -> str:
    """Return the latest price and daily change of a stock."""
    hist = yf.Ticker(ticker).history(period="5d")["Close"].dropna()
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
    history = yf.Ticker(ticker).history(period="1mo")
    if history.empty:
        return f"No history found for {ticker.upper()}."
    tail = history[["Open", "High", "Low", "Close", "Volume"]].tail(10).round(2)
    return f"Recent history for {ticker.upper()}:\n{tail.to_string()}"


@tool
@timed
def analyze_price_history(ticker: str) -> str:
    """Analyze up to 5 years of price history: returns over 1m/3m/6m/1y/3y/5y,
    annual volatility, max drawdown, 50/200-day moving averages, RSI(14), 52-week range."""
    close = yf.Ticker(ticker).history(period="5y")["Close"]
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
    info = yf.Ticker(ticker).info
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
    info = yf.Ticker(ticker).info
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
    cur = (stock.info or {}).get("financialCurrency") or ""
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
            close = yf.Ticker(sym).history(period="1y")["Close"]
            if close.dropna().empty:
                rows.append(f"{sym}: no data")
                continue
            a = analyze_prices(close)
            try:
                info = yf.Ticker(sym).info
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
        name = yf.Ticker(ticker).info.get("shortName")
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
    """Train a small LSTM on 1 year of prices, save a chart of actual vs predicted
    prices, and forecast the next 7 days. Returns the forecast, chart path and error stats."""
    stock_history = yf.Ticker(ticker).history(period="1y")
    close_prices = stock_history["Close"].dropna().values.reshape(-1, 1)
    dates = stock_history["Close"].dropna().index
    if len(close_prices) < 100:
        return f"Not enough price history for {ticker.upper()} to train a forecast."

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    lookback = 60
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([Input(shape=(lookback, 1)), LSTM(50), Dense(25), Dense(1)])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, batch_size=8, epochs=5, verbose=0)

    preds = scaler.inverse_transform(model.predict(X, verbose=0))
    actual = close_prices[lookback:]
    plot_dates = dates[lookback:]

    # Honest accuracy check: compare against the naive "tomorrow = today" baseline
    rmse_model = float(np.sqrt(np.mean((preds - actual) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((actual[1:] - actual[:-1]) ** 2)))

    window = scaled[-lookback:, 0].tolist()
    future = []
    for _ in range(7):
        x_in = np.array(window[-lookback:]).reshape(1, lookback, 1)
        nxt = model.predict(x_in, verbose=0)[0, 0]
        future.append(nxt)
        window.append(nxt)
    future_prices = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()
    future_dates = [dates[-1] + np.timedelta64(i, "D") for i in range(1, 8)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plot_dates, actual, color="blue", label="Actual Price")
    ax.plot(plot_dates, preds, color="red", linestyle="--", label="Model fit (training data)")
    ax.plot(future_dates, future_prices, color="green", marker="o", label="7-Day Forecast")
    ax.set_title(f"{ticker.upper()} LSTM forecast (demo, not advice)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    chart_path = os.path.join(CHART_DIR, f"{ticker.upper()}_{uuid.uuid4().hex[:8]}.png")
    fig.savefig(chart_path)
    plt.close(fig)

    forecast_lines = [f"Day {i + 1}: {p:.2f}" for i, p in enumerate(future_prices)]
    return (
        f"Chart saved at: {chart_path}\n"
        "7-day forecast (rough estimate):\n" + "\n".join(forecast_lines) + "\n"
        f"Model RMSE on training data: {rmse_model:.2f} vs naive 'tomorrow = today' baseline: {rmse_naive:.2f}. "
        "If the model is not clearly better than the baseline, the forecast has little predictive value."
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
6. For forecasts, use get_graph. Give the 7 numbers, mention the chart path, and tell the user the
   model's error versus the naive baseline. Say plainly that it is a weak demo, not a prediction to trade on.
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


# ----------------------------- CLI -------------------------------

if __name__ == "__main__":
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