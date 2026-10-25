# 📈 AI Stock Market Analyzer

An AI-powered stock research assistant that combines a **Gemini-powered LangChain tool-calling agent**, an **LSTM deep learning forecaster**, and a **FastAPI + HTML chat interface**.

🔗 **Live Demo:** [finance-agent-u1oo.onrender.com](https://finance-agent-u1oo.onrender.com)
*(Hosted on Render's free tier, so the first request after inactivity can take a minute to wake up.)*

> 🧠 Built by me as a learning project, with help from Claude AI for the FastAPI integration and web interface.

---

## 🚀 What It Does

Ask in plain English, for example *"Give me a full analysis of TCS.NS"* or *"Compare TCS.NS and INFY.NS"*. The agent picks the right tools, fetches live data, and summarizes the result.

- 💰 **Live Price**: latest price and daily % change
- 📊 **Price History Analysis**: returns over 1m / 3m / 6m / 1y / 3y / 5y, volatility, max drawdown, 50/200-day moving averages, RSI, 52-week range
- 🏢 **Company Profile**: sector, industry, country, business summary
- 📑 **Key Ratios**: P/E, P/B, ROE, margins, debt/equity, dividend yield, analyst targets
- 🧾 **Financial Statements**: 4 years of revenue, profit, debt, equity and cash flow
- ⚖️ **Stock Comparison**: 2 to 4 stocks side by side
- 📰 **Financial News**: latest headlines via NewsAPI, searched by company name
- 🔮 **LSTM Forecast**: actual vs model-fit chart plus a 7-day price forecast, reported with its error against a naive "tomorrow = today" baseline
- 💬 **Conversation Memory**: follow-up questions work ("now compare it with INFY")

---

## 📁 Project Structure

```
Finance_Agent/
├── README.md
├── .gitignore
└── project/
    ├── agent.py            # Tools, LSTM model, Gemini LangChain agent, run_agent()
    ├── main.py             # FastAPI app: /chat, /health, serves index.html and charts
    ├── index.html          # Chat interface (HTML/CSS/JavaScript)
    ├── test_gemini.py      # Checks which Gemini models respond with your key
    ├── requirements.txt
    └── .env                # API keys (not uploaded to GitHub)
```

The `charts/` folder is created automatically at runtime to store generated graphs.

---

## 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| Python | Core language |
| LangChain | Tool-calling agent framework |
| Google Gemini API | LLM for understanding questions and choosing tools |
| TensorFlow / Keras | LSTM deep learning model |
| yfinance | Live and historical stock data from Yahoo Finance |
| NewsAPI | Financial news |
| Scikit-learn | Data normalization (MinMaxScaler) |
| Pandas / NumPy | Metrics: returns, volatility, drawdown, RSI |
| Matplotlib | Chart generation |
| FastAPI + Uvicorn | REST API backend |
| HTML / CSS / JavaScript | Chat interface |
| Render | Deployment |
| python-dotenv | API key management |

---

## 🧰 Agent Tools

| Tool | Purpose |
|------|---------|
| `get_stock_price` | Latest price and daily change |
| `get_stock_history` | Last 10 trading days of OHLCV |
| `analyze_price_history` | 5-year returns, volatility, drawdown, moving averages, RSI |
| `get_company_info` | Company profile and business summary |
| `get_key_ratios` | Valuation, quality ratios and analyst targets |
| `get_financials` | Income statement, balance sheet, cash flow |
| `compare_stocks` | Side-by-side comparison of 2 to 4 tickers |
| `get_financial_news` | Latest headlines about the company |
| `get_graph` | Trains the LSTM, saves the chart, returns the 7-day forecast |

The system prompt tells the agent to use numbers only from tool results and to answer "data not available" otherwise.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/hingukrishn0512/Finance_Agent.git
cd Finance_Agent/project
```

### 2. Create a virtual environment
Python 3.11 or 3.12 is recommended (TensorFlow does not support every Python version).
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file in the `project` folder
```env
GEMINI_API_KEY=your_gemini_api_key
NEWS_API_KEY=your_newsapi_key
GEMINI_MODEL=your_working_gemini_model
TF_ENABLE_ONEDNN_OPTS=0
```
Google retires Gemini model names often. Run `python test_gemini.py` to find a model that works with your key and put its name in `GEMINI_MODEL`.

### 5. Run the app
```bash
# Web app (open http://127.0.0.1:8000)
uvicorn main:app --reload

# Or the terminal agent only
python agent.py
```

---

## 🔑 API Keys Required

| API | Free Tier | Get Key |
|-----|-----------|---------|
| Google Gemini | ✅ Free tier (rate limited) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| NewsAPI | ✅ Free tier (delayed articles) | [newsapi.org](https://newsapi.org) |
| yfinance | ✅ No key needed | Built-in |

---

## 🌐 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat interface |
| `POST` | `/chat` | Body: `{"message": "...", "session_id": "..."}`. Returns `{"reply": "...", "chart_urls": [...]}` |
| `GET` | `/health` | Health check |
| `GET` | `/charts/<file>.png` | Generated forecast charts |
| `GET` | `/docs` | Interactive Swagger docs |

Requests that share a `session_id` share conversation memory (the last 10 messages, kept in server RAM).

---

## 💡 Example Questions

```
What is the price of RELIANCE.NS?
Give me a full analysis of TCS.NS
Compare TCS.NS and INFY.NS
Show graph and 7-day prediction for INFY.NS
How has HDFC Bank done over the last 5 years, and what's the latest news?
```

For Indian stocks use `.NS` (NSE) or `.BO` (BSE) tickers, for example `RELIANCE.NS`.

---

## 🧠 How the LSTM Model Works

1. Downloads 1 year of daily closing prices
2. Normalizes prices to the 0 to 1 range with MinMaxScaler
3. Builds 60-day sliding windows (60 days → predict day 61)
4. Trains a model with one LSTM layer (50 units) followed by two Dense layers
5. Plots actual prices against the model's fit on its training data
6. Forecasts the next 7 days by feeding each prediction back into the window
7. Compares the model's error (RMSE) with a naive "tomorrow = today" baseline and reports both

---

## ☁️ Deployment (Render)

| Setting | Value |
|---------|-------|
| Root Directory | `project` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Environment variables | `GEMINI_API_KEY`, `NEWS_API_KEY`, `GEMINI_MODEL`, `PYTHON_VERSION` |

---

## 🚧 Known Limitations

- The LSTM forecast is a demo. Short training on one year of prices often does not beat the naive baseline, so treat it as an illustration of the technique, not a prediction to trade on.
- Data comes from `yfinance`, an unofficial Yahoo Finance wrapper, so some fields can be missing, slow, or rate limited, especially for Indian stocks.
- Conversation memory and generated charts are stored on the server and are cleared on restart.
- On Render's free tier the service sleeps when idle, and TensorFlow training is memory heavy.

---

## ⚠️ Disclaimer

> This tool is for **educational purposes only**. Stock predictions made by this model are not financial advice. Always consult a certified financial advisor before making investment decisions.

---

## 👨‍💻 Author

**Krishn Hingu**: built as a learning project combining ML, Deep Learning and Generative AI.

- LinkedIn: [linkedin.com/in/krishnhingu](https://www.linkedin.com/in/krishnhingu/)
- GitHub: [github.com/hingukrishn0512](https://github.com/hingukrishn0512)

---

## ⭐ If you found this helpful, give it a star!
