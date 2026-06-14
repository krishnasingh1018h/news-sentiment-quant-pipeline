import optuna
import sqlite3
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

DB_PATH = 'C:\\Users\\Krishna\\pulsestream\\include\\pulsestream.db'
# 1. Load your consolidated data from the DB
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM stock_news_sentiment", conn)
    conn.close()
    # Ensure all your feature columns are numeric here!
    columns_to_drop = ['title', 'description', 'url', 'source_name', 'publishedAt', 'published_date']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # 2. Aggregate multiple articles per day into a single daily grain
    # (If using hourly stock data, change 'date' to your datetime column)
    daily_df = df.groupby(['date', 'ticker']).agg({
        'close': 'first',
        'high': 'first',
        'low': 'first',
        'open': 'first',
        'volume': 'first',
        'sentiment_positive': 'mean',
        'sentiment_negative': 'mean',
        'sentiment_neutral': 'mean'
    }).reset_index()

    # Sort chronologically to maintain time-series integrity
    daily_df = daily_df.sort_values(by=['ticker', 'date']).reset_index(drop=True)

    # 3. Feature Engineering
    print("🧠 Building classification features...")
    daily_df['rolling_3d_pos'] = daily_df.groupby('ticker')['sentiment_positive'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    daily_df['rolling_3d_neg'] = daily_df.groupby('ticker')['sentiment_negative'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    daily_df['price_momentum'] = daily_df.groupby('ticker')['close'].pct_change()

    # 4. Create the Binary Classification Target
    # Shift close price back by 1: Will tomorrow's close be higher than today's close?
    daily_df['next_day_close'] = daily_df.groupby('ticker')['close'].shift(-1)
    daily_df['target'] = (daily_df['next_day_close'] > daily_df['close']).astype(int)

    # Drop rows with NaN caused by shifting/momentum
    clean_df = daily_df.dropna().copy()
    return clean_df


df = load_data()
X = df[['sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 
                'rolling_3d_pos', 'rolling_3d_neg', 'price_momentum']] # Add your features
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. The Tuning Logic
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    # Use cross-validation for a reliable score
    score = cross_val_score(model, X_train, y_train, cv=3).mean()
    return score

# 3. Execution
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"✅ Tuning complete!")
print(f"Best Params: {study.best_params}")