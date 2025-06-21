import streamlit as st
import yfinance as yf
from newsapi import NewsApiClient
from transformers import pipeline
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from transformers import pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("Stock Sentiment & Return Predictor")

nifty_50_tickers = [
    "RELIANCE.NS",
    "INFY.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "KOTAKBANK.NS",
    "ITC.NS",
    "LT.NS",
    "SBIN.NS",
    "AXISBANK.NS"
]

ticker_input = st.selectbox("Select NIFTY 50 ticker symbol:", nifty_50_tickers, index=1)

newsapi = NewsApiClient(api_key='d43fa502e3dd43fabfdebb0e33145ad9')
sentiment_pipeline = pipeline("sentiment-analysis")

if st.button("Analyze"):

    ticker = yf.Ticker(ticker_input)
    df_stock = ticker.history(period="6mo").reset_index()
    df_stock['Date'] = pd.to_datetime(df_stock['Date']).dt.tz_localize(None)
    df_stock['Return'] = df_stock['Close'].pct_change()
    df_stock.dropna(inplace=True)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)

    st.write(f"Fetching news from {start_date.date()} to {end_date.date()}")

    results = []
    current_date = start_date

    while current_date <= end_date:
        from_date = current_date.strftime('%Y-%m-%d')
        to_date = (current_date + timedelta(days=1)).strftime('%Y-%m-%d')

        try:
            articles = newsapi.get_everything(
                q=ticker_input.split('.')[0],
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

    df_news = pd.DataFrame(results)
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    daily_sentiment = df_news.groupby('Date')['Sentiment'].mean().reset_index()

    df_stock = pd.merge(df_stock, daily_sentiment, on='Date', how='left')
    df_stock['Sentiment'].fillna(0, inplace=True)

    st.write("Sample data with sentiment:")
    st.dataframe(df_stock[['Date', 'Close', 'Return', 'Sentiment']].tail(10))

    st.info(
        "ℹ️ **Sentiment Guide:**\n"
        "- ➕ Positive score = Positive sentiment\n"
        "- ➖ Negative score = Negative sentiment\n"
        "- 0 means no relevant news for that date"
    )

    X = df_stock[['Sentiment']]
    y = df_stock['Return']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Model Mean Squared Error (MSE): {mse:.6f}")
    st.write(f"Model R-squared (R²): {r2:.6f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test.values, label='Actual Return', marker='o')
    ax.plot(y_pred, label='Predicted Return', marker='x')
    ax.set_title('Stock Return: Actual vs Predicted')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Return')
    ax.legend()
    st.pyplot(fig)
