Stock Sentiment and Return Predictor
This project is an interactive Streamlit web app that allows users to explore the relationship between recent news sentiment and stock returns for selected NIFTY 50 companies. 
It demonstrates how Natural Language Processing (NLP) can be combined with financial data to provide additional insights.

**The application performs these main tasks:**

1. Allows the user to select a NIFTY 50 stock from a dropdown.
   
2.Fetches recent stock prices and the latest news articles for the selected stock.

3.Uses a sentiment analysis model from Hugging Face Transformers to score each news headline as positive or negative.

4.Aggregates daily sentiment and merges it with daily stock returns.

5.Trains a simple Linear Regression model to test whether daily sentiment scores can help predict short-term stock returns.

**Key Takeaway**

While it is possible to assign sentiment scores to headlines, this project shows that using daily news sentiment alone is not a reliable predictor of short-term price movements.
In real-world scenarios, stock prices are influenced by multiple complex factors beyond just news sentiment.


**Features**

-Select any popular NIFTY 50 stock ticker symbol from a dropdown.

-View the most recent news headlines and their sentiment scores.

-Understand how sentiment is quantified

-Visualize actual vs. predicted returns using a simple machine learning model.

**How to Run**

-Clone or download this repository.

-Create and activate a virtual environment (recommended).

-Install the required packages listed in requirements.txt.

-Run the app using Streamlit.


You can also deploy this app directly on Streamlit Cloud by connecting this GitHub repository.


**Disclaimer
For educational use only — not financial advice.**






