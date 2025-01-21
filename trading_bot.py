import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import requests

# Flask App for Dashboard
app = Flask(__name__)

# Step 1: Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    if not ticker.endswith(('.NS', '.BO')):
        ticker += '.NS'  # Default to NSE for Indian stocks
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        print(f"No data found for ticker: {ticker}")
        return pd.DataFrame()
    return data

# Step 2: Train ML model
def train_ml_model(data):
    if data.empty:
        print("No data to train the model.")
        return None, 0.0

    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0)
    target = data['Target']

    if target.isnull().all():
        print("Not enough data to train the model.")
        return None, 0.0

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model, accuracy

# Step 3: Sentiment analysis
def analyze_sentiment(news_api_url):
    response = requests.get(news_api_url)
    if response.status_code != 200:
        print(f"Error fetching news: {response.status_code}")
        return 0

    news_data = response.json()
    articles = news_data.get('articles', [])
    sentiment_scores = []

    for article in articles:
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        analysis = TextBlob(title + " " + description)
        sentiment_scores.append(analysis.sentiment.polarity)

    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    print(f"Average Sentiment Score: {avg_sentiment:.2f}")
    return avg_sentiment

# Step 4: Simulated trade
def simulated_trade(ticker, model, latest_data, sentiment_score):
    if model is None or latest_data.empty:
        return f"No model or data available for simulated trade for {ticker}."

    latest_features = latest_data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(latest_features)

    if prediction == 1 and sentiment_score > 0:
        return f"Strong Buy signal for {ticker} based on positive sentiment and price trends."
    elif prediction == 1:
        return f"Buy signal for {ticker} based on price trends but sentiment is neutral/negative."
    else:
        return f"No Buy signal for {ticker}. Consider monitoring further."

# Step 5: Visualization
def plot_performance(data, ticker):
    if data.empty:
        print("No data to plot performance.")
        return
    plt.figure(figsize=(10, 6))
    data['Close'].plot(label='Close Price', color='blue')
    plt.title(f'{ticker} Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f'static/{ticker}_performance.png')
    plt.close()

# Flask Routes
@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form.get('ticker').upper()
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    stock_data = fetch_stock_data(ticker, start_date, end_date)
    if stock_data.empty:
        return render_template('result.html', message="No data found for the given ticker.")

    plot_performance(stock_data, ticker)

    model, accuracy = train_ml_model(stock_data)
    sentiment_score = analyze_sentiment("https://newsapi.org/v2/everything?q=stocks&language=en&apiKey=478d37maf9849519228815fb6581326")
    trade_signal = simulated_trade(ticker, model, stock_data, sentiment_score)

    return render_template(
        'result.html',
        ticker=ticker,
        accuracy=accuracy,
        sentiment_score=sentiment_score,
        trade_signal=trade_signal,
        chart_path=f'static/{ticker}_performance.png'
    )

if __name__ == "__main__":
    app.run(debug=True)
