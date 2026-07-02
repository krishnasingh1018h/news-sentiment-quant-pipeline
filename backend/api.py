import joblib
import numpy as np
import sqlite3
import os
import datetime
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

# Initialize FastAPI App
app = FastAPI(
    title="Pulsestream Hybrid Quant API",
    description="Production Quant Pipeline - Database Lookups (Historical Backtesting) + Real-Time Inference (Today's Forecasts)",
    version="4.0.0"
)

# Paths
DB_PATH = "C:\\Users\\Krishna\\pulsestream\\include\\pulsestream.db"
MODEL_PATH = "C:\\Users\\Krishna\\pulsestream\\models\\xgb_model.pkl"

# 1. Global Setup: Load XGBoost and the high-level Transformers Pipeline
try:
    model = joblib.load(MODEL_PATH)
    print("✅ XGBoost Model Loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load XGBoost model file. Error: {e}")
    model = None

try:
    print("⏳ Initializing FinBERT Sentiment Pipeline (downloading if running for the first time)...")
    # top_k=None forces the pipeline to return scores for ALL classes (pos, neg, neu)
    nlp_pipeline = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
    print("✅ FinBERT NLP Pipeline Engine ready.")
except Exception as e:
    print(f"⚠️ Warning: FinBERT pipeline failed to initialize: {e}")
    nlp_pipeline = None


# 2. Pydantic Schemas matching the Frontend Canvas interface
class PredictionRequest(BaseModel):
    ticker: str = Field(..., examples=["NVDA"], description="Stock ticker symbol")
    date: str = Field(..., examples=["2026-06-20"], description="Target prediction date (YYYY-MM-DD)")

class PredictionResponse(BaseModel):
    ticker: str
    date: str
    model_probability_up: float
    sentiment_score_finbert: float
    combined_probability: float
    final_decision: str
    live_features_extracted: dict


# 3. Fetch Real-Time Price Momentum (for Live Routing)
def fetch_live_price_momentum(ticker: str) -> float:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if len(hist) < 2:
            return 0.0
        close_today = hist['Close'].iloc[-1]
        close_prev = hist['Close'].iloc[-2]
        momentum = (close_today - close_prev) / close_prev
        return float(momentum)
    except Exception as e:
        print(f"⚠️ Error fetching yfinance data for {ticker}: {e}")
        return 0.0


# 4. Process Live News Sentiment via Pipeline Helper (for Live Routing)
def compute_live_sentiment(ticker: str):
    if nlp_pipeline is None:
        return 0.05, 0.10, 0.70, 0.15, 0.65  # Safety fallback values
    
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news
        
        if not news_items:
            return 0.1, 0.1, 0.8, 0.1, 0.1
        
        headlines = [item['title'] for item in news_items[:5]]
        pos_scores, neg_scores, neu_scores = [], [], []
        
        # Pass all 5 headlines into the pipeline at once
        pipeline_outputs = nlp_pipeline(headlines)
        
        for result in pipeline_outputs:
            scores_dict = {item['label'].lower(): item['score'] for item in result}
            pos_scores.append(scores_dict.get('positive', 0.0))
            neg_scores.append(scores_dict.get('negative', 0.0))
            neu_scores.append(scores_dict.get('neutral', 0.0))
                
        # Calculate final means across all sampled headlines
        s_pos = float(np.mean(pos_scores))
        s_neg = float(np.mean(neg_scores))
        s_neu = float(np.mean(neu_scores))
        
        # Appoximated rolling elements for live tracking
        rolling_3d_pos = s_pos * 0.95 
        rolling_3d_neg = s_neg * 1.05
        
        return s_neg, s_pos, s_neu, rolling_3d_neg, rolling_3d_pos
        
    except Exception as e:
        print(f"❌ Sentiment pipeline analysis failed: {e}")
        return 0.1, 0.1, 0.8, 0.1, 0.1


# 5. Helper function to query pre-calculated historical features from SQLite database
def get_historical_features_from_db(ticker: str, date: str):
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail=f"Database file missing at {DB_PATH}")
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check what tables are available to resolve dynamically
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    
    # Auto-resolve table name in case rolling has historical suffix names
    target_table = "stock_news_sentiment_rolling"
    for table in tables:
        if "rolling" in table:
            target_table = table
            break

    query = f"""
        SELECT 
            rolling_3d_neg, 
            sentiment_negative, 
            sentiment_positive, 
            rolling_3d_pos, 
            sentiment_neutral, 
            price_momentum 
        FROM {target_table}
        WHERE LOWER(ticker) = LOWER(?) AND date = ?
    """
    
    cursor.execute(query, (ticker, date))
    row = cursor.fetchone()
    conn.close()
    return row


# 6. Core Hybrid Prediction Route
@app.post("/predict", response_model=PredictionResponse, tags=["Hybrid Inference"])
async def predict_hybrid(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="XGBoost Prediction model file is missing or broken.")
    
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    is_live_request = (request.date == today_str)
    
    # --- ROUTING SYSTEM ---
    if is_live_request:
        print(f"📡 Routing NVDA/INTC to LIVE Pipeline (Scraping & Transformer Engine)")
        price_mom = fetch_live_price_momentum(request.ticker)
        s_neg, s_pos, s_neu, r_3d_neg, r_3d_pos = compute_live_sentiment(request.ticker)
    else:
        print(f"📂 Routing to HISTORICAL Database Lookup for Date: {request.date}")
        db_row = get_historical_features_from_db(request.ticker, request.date)
        
        if not db_row:
            raise HTTPException(
                status_code=404, 
                detail=f"No pre-calculated historical features found in database for ticker '{request.ticker}' on date '{request.date}'."
            )
        
        # Unpack SQL historical values
        (r_3d_neg, s_neg, s_pos, r_3d_pos, s_neu, price_mom) = db_row

    # --- INFERENCE STEP ---
    try:
        # Construct XGBoost input vector using exact training alignment order
        feature_matrix = np.array([[
            r_3d_neg, 
            s_neg, 
            s_pos, 
            r_3d_pos, 
            s_neu, 
            price_mom
        ]])
        
        probabilities = model.predict_proba(feature_matrix)[0]
        prob_up = float(probabilities[1])
        
        # Alpha consolidated score formula
        combined_prob = (prob_up * 0.6) + (s_pos * 0.2) - (s_neg * 0.2)
        combined_prob = max(0.0, min(1.0, combined_prob))
        
        decision = "Up" if combined_prob >= 0.5 else "Down/Flat"
        
        return PredictionResponse(
            ticker=request.ticker.upper(),
            date=request.date,
            model_probability_up=round(prob_up, 4),
            sentiment_score_finbert=round(s_pos - s_neg, 4),            
            combined_probability=round(combined_prob, 4),
            final_decision=decision,
            live_features_extracted={
                "price_momentum": round(price_mom, 6),
                "sentiment_positive": round(s_pos, 4),
                "sentiment_negative": round(s_neg, 4),
                "sentiment_neutral": round(s_neu, 4),
                "rolling_3d_pos": round(r_3d_pos, 4),
                "rolling_3d_neg": round(r_3d_neg, 4)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")