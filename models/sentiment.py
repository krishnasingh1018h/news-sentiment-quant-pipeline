import sqlite3
import pandas as pd
import torch
from transformers import pipeline
import torch.nn.functional as f

DB_PATH = 'C:\\Users\\Krishna\\pulsestream\\include\\pulsestream.db'

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM stock_news",conn)
    conn.close()
    return df


def run_sentiment_analysis(text):
    pipe = pipeline(task="text-classification",model="ProsusAI/finbert",top_k=None)
    pos_scores = []
    neg_scores = []
    neu_scores = []

    for index,row in text.iterrows():
        new_text = str(row.get('description',''))
        if not new_text or new_text == 'none' or new_text.strip() == '':
            pos_scores.append(0.0)
            neg_scores.append(0.0)
            neu_scores.append(1.0)
            continue
        result = pipe(new_text,truncation=True)[0]

        score = { res['label'].lower(): res['score'] for res in result}
        pos_scores.append(score.get('positive',0.0))
        neg_scores.append(score.get('negative',0.0))
        neu_scores.append(score.get('neutral',0.0))

    text['sentiment_positive'] = pos_scores
    text['sentiment_negative'] = neg_scores
    text['sentiment_neutral'] = neu_scores

    return text


def save_results(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('stock_news_sentiment',conn,if_exists='replace',index=False)
    conn.close()
    print("Sentiment analysis results saved to database.")

if __name__ == "__main__":
    df = load_data()
    df_with_sentiment = run_sentiment_analysis(df)
    save_results(df_with_sentiment)

    