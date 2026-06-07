# 📈 AI Stock Market Analyzer

An AI-powered stock market analysis tool built with **LSTM Deep Learning**, **LangChain AI Agent**, and **Streamlit UI**.

> 🧠 Core agent logic written by me | 🎨 Streamlit UI integrated with help of Claude AI

---

## 🚀 What It Does

- 📊 **Live Stock Price** — fetches real-time price and % change
- 🏢 **Company Profile** — sector, industry, market cap, website
- 📰 **Financial News** — latest 1-month news with working article links
- 📈 **LSTM Prediction Chart** — actual vs predicted price using deep learning
- 🔮 **7-Day Forecast** — predicts next 7 days of stock price

---

## 📁 Project Structure

```
├── agent.py        # Core agent logic — tools, LSTM model, LangChain agent
├── app.py          # Full Streamlit UI — interactive dashboard
├── .env            # API keys (not uploaded to GitHub)
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| Python | Core language |
| LangChain + LangGraph | AI Agent framework |
| Groq (Llama 3) | LLM for natural language understanding |
| tensorflow.keras | LSTM deep learning model |
| yfinance | Live stock data from Yahoo Finance |
| NewsAPI | Financial news fetching |
| Scikit-learn | Data normalization (MinMaxScaler) |
| Matplotlib | Chart plotting |
| Streamlit | Web UI dashboard |
| Python-dotenv | API key management |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/hingukrishn0512/Finanace-Agent.git
cd Finanace-Agent
```

### 2. Create virtual environment
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

### 4. Create `.env` file
```env
GROQ_API_KEY=your_groq_api_key
NEWS_API_KEY=your_newsapi_key
```

### 5. Run the app
```bash
# Run Streamlit UI
streamlit run app.py

# Run base agent (terminal only)
python agent.py
```

---

## 🔑 API Keys Required

| API | Free Tier | Get Key |
|-----|-----------|---------|
| Groq | ✅ Free | [console.groq.com](https://console.groq.com) |
| NewsAPI | ✅ Free | [newsapi.org](https://newsapi.org) |
| yfinance | ✅ No key needed | Built-in |

---

## 💡 How to Use

**Terminal agent (`agent.py`):**
```
enter the stock name: RELIANCE.NS
enter the stock name: AAPL
enter the stock name: TCS.NS
```

**Streamlit app (`app.py`):**
1. Enter a stock ticker in the search box (e.g. `RELIANCE.NS`, `AAPL`, `TCS.NS`)
2. Click **Analyze Stock** to see price, company info and LSTM chart

## 🧠 How the LSTM Model Works

1. Downloads 1-2 years of historical closing prices
2. Normalizes prices to 0-1 range using MinMaxScaler
3. Creates 60-day sliding windows (60 days → predict day 61)
4. Trains a 2-layer LSTM neural network
5. Predicts on test data and plots actual vs predicted
6. Forecasts next 7 days by sliding the window forward

---

## ⚠️ Disclaimer

> This tool is for **educational purposes only**. Stock predictions made by this model are not financial advice. Always consult a certified financial advisor before making investment decisions.

---

## 👨‍💻 Author

**Krishn** — Built as a learning project combining ML, Deep Learning and Generative AI.

- LinkedIn: (https://www.linkedin.com/in/krishnhingu/)

---

## ⭐ If you found this helpful, give it a star!
