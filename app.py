import streamlit as st
import yfinance as yf
from newsapi import NewsApiClient
from transformers import pipeline
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("Stock Sentiment & Return Predictor")

# User inputs ticker symbol
ticker_input = st.text_input("Enter NIFTY 50 ticker symbol (e.g. INFY.NS):", "INFY.NS")

# Initialize NewsAPI and Hugging Face sentiment pipeline once
newsapi = NewsApiClient(api_key='YOUR_NEWSAPI_KEY')
sentiment_pipeline = pipeline("sentiment-analysis")

if st.button("Analyze"):

    # Fetch stock data for last 6 months
    ticker = yf.Ticker(ticker_input)
    df_stock = ticker.history(period="6mo").reset_index()
    df_stock['Date'] = pd.to_datetime(df_stock['Date']).dt.tz_localize(None)
    df_stock['Return'] = df_stock['Close'].pct_change()
    df_stock.dropna(inplace=True)

    # Define news date range (last 30 days)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)

    st.write(f"Fetching news from {start_date.date()} to {end_date.date()}")

    # Collect news headlines with sentiment
    results = []
    current_date = start_date

    while current_date <= end_date:
        from_date = current_date.strftime('%Y-%m-%d')
        to_date = (current_date + timedelta(days=1)).strftime('%Y-%m-%d')

        try:
            articles = newsapi.get_everything(
                q=ticker_input.split('.')[0],  # e.g. INFY from INFY.NS
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='relevancy',
                page_size=30
            )
            for article in articles['articles']:
                headline = article['title']
                date_published = article['publishedAt'][:10]

                result = sentiment_pipeline(headline)[0]
                score = result['score']
                if result['label'] == 'NEGATIVE':
                    score *= -1

                results.append({'Date': date_published, 'Headline': headline, 'Sentiment': score})

        except Exception as e:
            st.warning(f"Error fetching news for {from_date}: {e}")

        time.sleep(1)
        current_date += timedelta(days=1)

    # Create news DataFrame
    df_news = pd.DataFrame(results)
    df_news['Date'] = pd.to_datetime(df_news['Date'])

    # Aggregate daily sentiment as mean sentiment per day
    daily_sentiment = df_news.groupby('Date')['Sentiment'].mean().reset_index()

   
    df_stock = pd.merge(df_stock, daily_sentiment, on='Date', how='left')


    df_stock['Sentiment'].fillna(0, inplace=True)


    st.write("Sample data with sentiment:")
    st.dataframe(df_stock[['Date', 'Close', 'Return', 'Sentiment']].tail(10))

    # Prepare features and target for ML
    X = df_stock[['Sentiment']]
    y = df_stock['Return']

  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   
    model = LinearRegression()
    model.fit(X_train, y_train)

    # test data
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Model Mean Squared Error (MSE): {mse:.6f}")
    st.write(f"Model R-squared (R²): {r2:.6f}")

    # Plot actual vs predicted returns
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_test.values, label='Actual Return', marker='o')
    ax.plot(y_pred, label='Predicted Return', marker='x')
    ax.set_title('Stock Return: Actual vs Predicted')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Return')
    ax.legend()
    st.pyplot(fig)
