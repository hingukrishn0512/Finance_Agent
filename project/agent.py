#  now with help of the claude AI i integrate streamlit in these project for the agent interface 
# all the streamlit or desgining stuff did by the AI 

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
import pandas as pd
import yfinance as yf
import requests
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
# LSTM = Long Short Term Memory
# It's a type of neural network designed specifically for sequences — data where the order matters.

# Why not a normal neural network?
# A normal neural network looks at each input independently. It has no memory.
# Example:
# Normal NN sees: [day60 price]  → predicts day61
# It ignores:      day1, day2, day3... day59
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# get_stock_price(ticker)        → live price from yfinance
# get_stock_history(ticker)      → last 1 year of data
# get_financial_news(ticker)     → latest news from NewsAPI
# get_company_info(ticker)       → name, sector, market cap

@tool
def get_stock_price(ticker):
    """return the current price of the stocks"""
    news = yf.Ticker(ticker)
    info = news.info

    price = info.get("currentPrice")
    change = info.get("regularMarketChangePercent")

    print(f"Ticker  : {ticker.upper()}")
    print(f"Price   : ${price}")
    print(f"Change  : {round(change, 2)}%")

@tool
def get_stock_history(ticker):
    """return the stock history """
    stock = yf.Ticker(ticker)
    history = stock.history(period = "1y") # 1 year ni history

    print(f"the history of the {ticker.upper()}\n")
    print(history[["Open", "High", "Low", "Close", "Volume"]].tail())

@tool
def get_financial_news(ticker):
    """retrun recent financial news about the stocks"""
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"

    response = requests.get(url)
    data = response.json()

    article = data.get("articles",[])
    
    if not article:
        print(f"there no news or articels about {ticker}")

    print(f"\nLatest news for {ticker.upper()}:")
    for i, article in enumerate(article):
        title = article.get("title")
        source = article.get("source", {}).get("name")
        published = article.get("publishedAt")
        print(f"\n{i+1}. {title}") # ***
        print(f"   Source: {source} | Published: {published}")
 
@tool
def get_company_info(ticker):
    """return the company information of that stock"""
    stock = yf.Ticker(ticker)
    info = stock.info

    name        = info.get("longName")
    sector      = info.get("sector")
    industry    = info.get("industry")
    market_cap  = info.get("marketCap")
    country     = info.get("country")
    website     = info.get("website")
 
    print(f"\nCompany Info for {ticker.upper()}:")
    print(f"  Name       : {name}")
    print(f"  Sector     : {sector}")
    print(f"  Industry   : {industry}")
    print(f"  Market Cap : ${market_cap:,}")
    print(f"  Country    : {country}")
    print(f"  Website    : {website}")


@tool
def get_graph(ticker):
    """return the acutal price and predicted price graph"""
    # 1. Fetch 1 year of historical data
    stock = yf.Ticker(ticker)
    stock_history = stock.history(period="1y")       # row , column
    close_prices = stock_history["Close"].values.reshape(-1, 1)
    dates = stock_history.index
    # reshape(-1,1) converts [1300, 1310, 1320] → [[1300],[1310],[1320]]
    # LSTM needs 2D input, not 1D
    # 2. Scale data between 0 and 1 (Crucial for LSTMs)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # 3. Create sequences (Using past 60 days to predict the next day)
    lookback = 60
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0]) # sliding window
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y) # dimensions equal ( , )
    
    # Reshape input to 3D: [samples, time_steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
                    #row       column  step by step 1 by 1
    # 4. Build and Train a quick LSTM model
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(lookback, 1)), # simple ann type layer
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=8, epochs=5, verbose=0)  # Hidden progress for speed
    
    # 5. Generate predictions
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions) # Revert scaling for the acutal prices
    
    # 6. Plot the results
    actual_prices = close_prices[lookback:]
    plot_dates = dates[lookback:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(plot_dates, actual_prices, color='blue', label='Actual Price')
    plt.plot(plot_dates, predictions, color='red', linestyle='--', label='Predicted Price')
    
    plt.title(f"{ticker} Price Prediction using LSTM")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0  )

agent = create_agent(model=llm,
                     tools = [get_stock_price , get_stock_history , get_financial_news , get_company_info , get_graph],
                     system_prompt="you are a helpful financial assitant which analyze the data and give all the " \
                     "knowledge you got about the stocks ,  all the graph and also the prediction of next 7 day stock price " \
                     "on the basis of the past 1 year performance of that stock",
        
                     )

print("enter 'exit' for exiting the loop")
while True:
    user_input = input("enter the stock name : ")

    if not user_input:
        print("please enter the valid statement..")
    elif user_input == "exit":
        print("signing off...")
        break

    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}],
    })
    print(result['messages'][-1].content)  # directly retrive the ai meesage 