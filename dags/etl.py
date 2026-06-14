import os
import requests
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import json

load_dotenv()

TICKERS = ["NVDA", "TSM", "AMD", "INTC"]
NEWS_QUERY = "semiconductor OR microchip OR NVIDIA OR AMD OR INTC"
DB_PATH = 'C:\\Users\\Krishna\\pulsestream\\include\\pulsestream.db'

def fetch_stock_data():
    
    all_data = pd.DataFrame()

    for ticker in TICKERS:
        try:
            print(f"🔄 Fetching data for {ticker}...")
            stock = yf.download(
                ticker,
                period="1mo",
                interval="1d",
                progress=False
            )

            if stock.empty:
                print(f"⚠️ No data for {ticker}")
                continue

            if isinstance(stock.columns, pd.MultiIndex):
                stock.columns = [col[0] for col in stock.columns]

            stock = stock.reset_index()
            stock['Ticker'] = ticker
            all_data = pd.concat([all_data, stock], ignore_index=True)

            time.sleep(2)

        except Exception as e:
            print(f"❌ Error downloading {ticker}: {e}")

    if all_data.empty:
        raise ValueError("No stock data fetched.")

    # Normalize column names
    all_data.columns = [c.lower().replace(' ', '_') for c in all_data.columns]

    # Convert date column to string for JSON safety
    all_data['date'] = pd.to_datetime(all_data['date'], errors='coerce').dt.strftime("%Y-%m-%d")

    return all_data.to_dict(orient='records')


def fetch_news_data():
    api_key = os.getenv("NEWS_API_KEY")
    url = f"https://newsapi.org/v2/everything?q={NEWS_QUERY}&from={(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])

    cleaned_news = []
    for a in articles:
        cleaned_news.append({
            "publishedAt": a.get("publishedAt"),
            "source_name": a.get("source", {}).get("name"),
            "title": a.get("title"),
            "description": a.get("description"),
            "url": a.get("url")
        })
    return cleaned_news



def transform_data(stock_data, news_data):
     # Required for parsing the cleaned data string
    
    stock_df = pd.DataFrame(stock_data)
    news_df = pd.DataFrame(news_data)
    
    # Catch empty edge cases gracefully to avoid merge crashes
    if stock_df.empty:
        return []
    if news_df.empty:
        for col in ['publishedAt', 'published_date', 'source_name', 'title', 'description', 'url']:
            stock_df[col] = None
        return stock_df.to_dict('records')
        
    # Process and align date attributes
    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
    news_df['published_date'] = news_df['publishedAt'].dt.date
    stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
    
    # Execute structural combining join 
    merged_df = pd.merge(stock_df, news_df, left_on='date', right_on='published_date', how='left')
    
    
    return merged_df.to_dict(orient='records')

def load_data(transformed_data):
    df = pd.DataFrame(transformed_data)
    if df.empty:
        print("No data to load.")
        return

    conn = sqlite3.connect(DB_PATH)
    df.to_sql('stock_news', conn, if_exists='append', index=False)
    conn.close()
    print(f"✅ Loaded {len(df)} rows into SQLite!")


# Run the pipeline
if __name__ == "__main__":
    stock_res = fetch_stock_data()
    news_res = fetch_news_data()
    transformed_res = transform_data(stock_res, news_res)
    load_data(transformed_res)
