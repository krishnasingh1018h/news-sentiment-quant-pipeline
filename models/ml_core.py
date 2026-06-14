import sqlite3
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import os
import joblib

DB_PATH = 'C:\\Users\\Krishna\\pulsestream\\include\\pulsestream.db'
def load_and_preprocess_classification():
    print("⏳ Loading unified data from database...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM stock_news_sentiment", conn)
    conn.close()

    # 1. Clear out the raw text and metadata columns
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

def train_classification_layer(df):
    features = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 
                'rolling_3d_pos', 'rolling_3d_neg', 'price_momentum']
    
    X = df[features]
    y = df['target']
    
    # Chronological Split (80% Train, 20% Test) - strictly NO random shuffling!
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"🚀 Training XGBoost Classifier on {len(X_train)} rows...")
    
    # Initialize Classifier with logloss to avoid deprecation warnings
    model = XGBClassifier(
        max_depth= 10,
        learning_rate=0.09784131061542206,
        n_estimators= 214,
        colsample_bytree= 0.6808137628153629,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    prob_predictions = model.predict_proba(X_test)[:, 1]
    
    # --- EVALUATION METRICS ---
    acc = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, prob_predictions)
    
    print("\n📊 Classification Model Evaluation:")
    print(f"Accuracy Score: {acc * 100:.2f}%")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Down/Flat (0)', 'Up (1)']))
    
    # Feature Importance Tracking
    importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    print("\n🔍 Feature Weighting:")
    print(importances.sort_values(by='Importance', ascending=False).to_string(index=False))
    
    return model

if __name__ == "__main__":
    processed_data = load_and_preprocess_classification()
    if not processed_data.empty:
        model = train_classification_layer(processed_data)
        

# 1. Define where you want to save it
        MODEL_DIR = 'C:\\Users\\Krishna\\pulsestream\\models'
        MODEL_FILENAME = 'xgb_model.pkl'
        MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

        # 2. Ensure the directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)

        # 3. Dump the model
        # 'model' here is the variable name of your trained XGBoost object
        joblib.dump(model, MODEL_PATH)

        print(f"✅ Model successfully dumped to {MODEL_PATH}")
                
                

