import sqlite3
import pandas as pd
DB_PATH = 'C:\\Users\\Krishna\\pulsestream\\include\\pulsestream.db'
import os
import yfinance as yf
TARGET_TICKERS = ["NVDA", "TSM", "AMD", "INTC"]
def seed_hitorical_data():
    print("⏳ Seeding historical stock data...")
    raw_data = pd.read_csv('raw_partner_headlines.csv',index_col=0)
    raw_data.columns = raw_data.columns.str.lower().str.replace(' ', '_')

    print("cleaning data...")
    raw_data = raw_data.rename(columns=
                               {
                                     'headline': 'description',
                                      'stock':'ticker',
                                      'publisher':'source_name',
                               })
    raw_data = raw_data[raw_data['ticker'].isin(TARGET_TICKERS)]
    raw_data['date'] = pd.to_datetime(raw_data['date'], errors='coerce').dt.strftime("%Y-%m-%d")
    start_date = raw_data['date'].min()
    end_date = pd.to_datetime(pd.to_datetime(raw_data['date'].max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Fetching stock data from {start_date} to {end_date}...")
    try:
        stock_data = yf.download(
            TARGET_TICKERS,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False
        )
        stock_data = stock_data.stack(level=1).reset_index()
        stock_data.columns = stock_data.columns.str.lower()
        stock_data['date'] = stock_data['date'].dt.strftime('%Y-%m-%d')

    except Exception as e:
        print(f"failed to fetch stock data from yfiniance :{e}")
    unified_df = pd.merge(stock_data,raw_data,on=['date','ticker'],how='inner')
    unified_df = unified_df.dropna(subset=['description','close'])
    total_rows = len(unified_df)
    print(f"📊 DATA VERIFICATION: There are exactly {total_rows:,} rows successfully unified and ready for export.")
    


    conn = sqlite3.connect(DB_PATH)
    unified_df.to_sql('stock_news',conn,if_exists='append',index=False)
    conn.close()
    print('data loading complete..😊')

if __name__=="__main__":
    seed_hitorical_data()



                                                         
