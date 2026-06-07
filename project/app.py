import os
import re
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent

#  this is the base of the agent which is written by me ..

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# --- Streamlit Theme & Page Layout Options ---
st.set_page_config(page_title="MarketIntel | AI Finance Terminal", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a modern, sleek interface design
st.markdown("""
    <style>
    /* Main background */
    .main { background-color: #0f172a; }

    /* Metric Card */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b, #0f172a) !important;
        padding: 20px !important;
        border-radius: 12px !important;
        border: 1px solid #334155 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    }

    /* Label — bright white so always visible */
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] div,
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    /* Value — bright white */
    div[data-testid="stMetricValue"] div,
    div[data-testid="stMetricValue"] span,
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }

    /* Delta — green for positive, red for negative */
    div[data-testid="stMetricDelta"] svg { display: none !important; }
    div[data-testid="stMetricDelta"] div,
    [data-testid="stMetricDelta"] {
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    /* Headers */
    h1 { color: #38bdf8 !important; font-weight: 700 !important; }
    h2 { color: #e2e8f0 !important; }
    h3 { color: #7dd3fc !important; }

    /* Tabs */
    div.stTabs [data-baseweb="tab"] {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
    }
    div.stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Controls Layout ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.info("This system uses a hybrid routing layer: quick tickers bind directly via native APIs, while complex conceptual descriptions scale to the Llama 3.3 ReAct core.")
    st.markdown("---")
    st.caption("🤖 Powered by Llama 3.3 & Keras LSTM")

# --- Application Header Window ---
st.title("📈 AI Market Intelligence Dashboard")
st.caption("Real-time pipeline analysis, news sentiment monitoring, and deep long short-term memory (LSTM) neural predictions.")
st.markdown("---")

# Initialize persistent session variables
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = ""

# --- Safe Ticker Extractor Engine ---
def clean_ticker(input_string: str) -> str:
    input_string = str(input_string).strip().upper()
    if "Q=" in input_string:
        match = re.search(r'Q=([A-Z0-9.\-]+)', input_string)
        if match: return match.group(1)
    match = re.search(r'([A-Z0-9.\-]+)', input_string.split('/')[-1])
    if match: return match.group(1)
    return input_string

# --- Refactored Tool Definitions ---
@tool
def get_stock_price(ticker: str) -> str:
    """Returns live stock valuation numbers."""
    ticker = clean_ticker(ticker)
    news = yf.Ticker(ticker)
    info = news.info
    return f"PRICE|{info.get('currentPrice', 'N/A')}|{info.get('regularMarketChangePercent', 0)}"

@tool
def get_stock_history(ticker: str) -> str:
    """Returns recent historical transaction tables."""
    ticker = clean_ticker(ticker)
    stock = yf.Ticker(ticker)
    history = stock.history(period="1y")
    return history[["Open", "High", "Low", "Close", "Volume"]].tail().to_string()


import datetime

@tool
def get_financial_news(ticker: str) -> str:
    """Returns past 1 month news for a stock ticker."""
    ticker       = clean_ticker(ticker)
    search_query = ticker.split(".")[0].strip()  # RELIANCE from RELIANCE.NS

    today         = datetime.date.today()
    one_month_ago = today - datetime.timedelta(days=30)

    # ✅ correct URL with full endpoint
    url = f"https://newsapi.org/v2/everything?q={search_query}&from={one_month_ago}&to={today}&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"

    try:
        response = requests.get(url)
        res      = response.json()

        if response.status_code != 200 or "articles" not in res:
            return "NEWS_EMPTY"

        articles = res.get("articles", [])
        if not articles:
            return "NEWS_EMPTY"

        output = ""
        for art in articles:
            clean_title  = str(art.get("title")).replace("||", " ")
            clean_source = str(art.get("source", {}).get("name")).replace("||", " ")
            clean_date   = str(art.get("publishedAt")).replace("||", " ")
            output += f"ITEM||{clean_title}||{clean_source}||{clean_date}\n"

        return output

    except:
        return "NEWS_EMPTY"
@tool
def get_company_info(ticker: str) -> str:
    """Returns profile data segments."""
    ticker = clean_ticker(ticker)
    stock = yf.Ticker(ticker)
    info = stock.info
    return f"INFO|{info.get('longName')}|{info.get('sector')}|{info.get('industry')}|{info.get('marketCap', 0)}|{info.get('website')}"

@tool
def get_graph(ticker: str) -> str:
    """Activates the machine learning visualizer window."""
    ticker = clean_ticker(ticker)
    st.session_state.show_graph = True
    st.session_state.current_ticker = ticker
    return f"GRAPH_TRIGGERED_FOR_{ticker}"

# --- Deep Memory LSTM Architecture Logic ---
def render_lstm_graph(ticker):
    stock = yf.Ticker(ticker)
    stock_history = stock.history(period="1y")
    if stock_history.empty:
        st.warning("Insufficient history sequence metrics for model generation.")
        return
        
    close_prices = stock_history["Close"].values.reshape(-1, 1)
    dates = stock_history.index
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    lookback = 60
    if len(scaled_data) <= lookback:
        st.warning("Not enough trading history to process sequence memory windows.")
        return
        
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(lookback, 1)),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=8, epochs=5, verbose=0)
    
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    
    actual_prices = close_prices[lookback:]
    plot_dates = dates[lookback:]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_dates, actual_prices, color='#1e3a8a', linewidth=2, label='Actual Historical Price')
    ax.plot(plot_dates, predictions, color='#dc2626', linestyle='--', linewidth=2, label='LSTM Predicted Price Sequence')
    ax.set_title(f"{ticker.upper()} Actual vs. LSTM Prediction Curve (1-Year Context Window)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Timeline")
    ax.set_ylabel("Price")
    ax.legend(frameon=True, facecolor='#ffffff')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=35)
    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close(fig)

# --- Dynamic Beautiful Layout Assembler ---
def build_structured_ui(ticker_symbol):
    ticker = clean_ticker(ticker_symbol)
    
    # 1. Fetch RAW data outputs through tool invoke blocks
    try:
        raw_price = get_stock_price.invoke({"ticker": ticker})
        raw_info = get_company_info.invoke({"ticker": ticker})
        raw_news = get_financial_news.invoke({"ticker": ticker})
        raw_hist = get_stock_history.invoke({"ticker": ticker})
    except Exception as e:
        st.error(f"Failed to gather real-time analytical vectors: {str(e)}")
        return

    # 2. Build Core Company Heading Block
    if "INFO|" in raw_info:
        _, name, sector, industry, cap, website = raw_info.split("|")
        st.markdown(f"## 🏢 {name}")
        st.caption(f"📍 **Sector:** {sector}  |  🏭 **Industry:** {industry}  |  🌐 [Visit Corporate Website]({website})")
    else:
        st.markdown(f"## 🏢 Asset Profile: {ticker}")

    st.markdown("### 📊 Market Metrics")
    # 3. Form Beautiful Metric Columns
    col1, col2, col3 = st.columns(3)
    
    if "PRICE|" in raw_price:
        _, price_val, change_val = raw_price.split("|")
        change_num = float(change_val)
        delta_str = f"{round(change_num, 2)}%"
        col1.metric("Live Valuation Price", f"${float(price_val):,}", delta=delta_str)
    
    if "INFO|" in raw_info and int(cap) > 0:
        col2.metric("Market Capitalisation", f"${int(cap):,}")
    else:
        col2.metric("Market Capitalisation", "N/A")
        
    col3.metric("Data Context Range", "1 Year (Daily)")
    
    st.markdown("<br>", unsafe_allow_html=True) 

    # 4. Tabular Workspace Layout Architecture
    tab1, tab2, tab3 = st.tabs(["📌 Technical Evaluation Chart", "📰 Latest News Feed", "🗃️ Raw Historical Feed"])
    
    with tab1:
        st.markdown("#### 📈 Machine Learning Sequence Modeling")
        with st.spinner("Initializing Deep Keras Engine & Adjusting Sliding Windows..."):
            try:
                render_lstm_graph(ticker)
            except Exception as chart_err:
                st.error(f"Chart plotting error caught: {str(chart_err)}")
                
    with tab2:
        st.markdown("#### 📰 Financial News Analytics")
        if "ITEM||" in raw_news:
            items = raw_news.strip().split("\n")
            for item in items:
                if "ITEM||" in item:
                    _, title, source, pub = item.split("||")
                    with st.container():
                        st.markdown(f"##### 🎯 {title}")
                        st.caption(f"📢 **Source:** {source} | 🗓️ **Published At:** {pub}")
                        st.markdown("---")
        else:
            st.info("No active news streams discovered via your NewsAPI tracking keys.")
            
    with tab3:
        st.markdown("#### 🗃️ Last 5 Trading Days Transaction Record")
        st.code(raw_hist, language="text")

# --- Agent Core Setup ---
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
agent = create_agent(
    model=llm,
    tools=[get_stock_price, get_stock_history, get_financial_news, get_company_info, get_graph],
    system_prompt="you are a helpful financial assistant which analyze the data and give all the knowledge you got about the stocks.",
)

# --- User Entry Query Window Area ---
# --- User Entry Query Window Area ---
user_input = st.text_input("🔍 Search Asset Symbol or Describe Request (e.g., RELIANCE.NS, AAPL):", "", key="stock_ticker_input")

if user_input:
    cleaned_input = user_input.strip().upper()
    
    # 1. Direct Ticker Route (Standard clean input)
    if cleaned_input.endswith(".NS") or (len(cleaned_input) <= 6 and " " not in cleaned_input):
        build_structured_ui(cleaned_input)
        
    # 2. Conversational & Analytical Route (Complex text queries)
    else:
        # Try to automatically extract a ticker code out of the sentence (e.g., "Tell me about AAPL")
        found_ticker = None
        words = cleaned_input.replace(",", "").replace("?", "").split()
        for word in words:
            if word.endswith(".NS") or len(word) <= 5:
                # Basic check to filter out common conversational filler words
                if word not in ["ABOUT", "STOCK", "PRICE", "GRAPH", "TREND", "SHARE", "WHAT", "SHOW"]:
                    found_ticker = word
                    break
        
        # Run the conversational AI workflow inside a clean spinner
        with st.spinner("🤖 Running deep market intelligence report workflow..."):
            try:
                # Run the conversational agent
                result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
                
                st.markdown("### 🤖 Financial Analysis Agent Response")
                st.info(result['messages'][-1].content)
                st.markdown("<br>", unsafe_index=True)
                
            except Exception as e:
                st.error(f"Conversational model routing error: {str(e)}")
        
        # If the agent found a ticker hidden inside the user's text sentence, build the UI beneath it
        if found_ticker:
            st.markdown("---")
            st.markdown(f"### 📊 Deep Analytics Dashboard for **{found_ticker}**")
            build_structured_ui(found_ticker)
        else:
            # Fallback message if the user typed a general question with no specific stock token
            st.markdown("---")
            st.caption("💡 **Tip:** If you want to see interactive metric cards and trained Keras LSTM graphs, make sure to explicitly include a valid ticker symbol (e.g., *RELIANCE.NS*, *AAPL*, *TSLA*) in your sentence!")
