import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="Pulsestream ML Inference API",
    description="Production API for Stock Movement Predictions using FinBERT and XGBoost",
    version="1.0.0"
)

# 1. Load your serialized XGBoost model globally at startup
try:
    # Adjust this path if your model file name or folder is different
    model = joblib.load("C:\\Users\\Krishna\\pulsestream\\models\\xgb_model.pkl")
except Exception as e:
    print(f"⚠️ Warning: Could not load model file locally. Error: {e}")
    model = None

# 2. Define the Pydantic Input Schema for Data Validation
class PredictionRequest(BaseModel):
    ticker: str = Field(...,examples='NVDIA',description="Stock ticker symbol")
    date: str = Field(...,examples="2020-12-12",description="tareget date")
    price_momentum: float = Field(...,examples=0.012,description="daily price percentage change")
    rolling_3d_pos: float = Field(..., example=0.45, description="3-day rolling positive sentiment average")
    sentiment_pos: float 
    sentiment_neg: float
    sentiment_neu: float
# 3. Define the Structured Output Schema matching your exact style
class PredictionResponse(BaseModel):
    ticker: str
    date: str
    model_probability_up: float
    sentiment_score_finbert: float
    combined_probability: float
    final_decision: str

# 4. Health Check Endpoint
@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# 5. The Core Prediction Endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def check_prediction(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Machine Learning model is not initialized on the server.")
    
    try:
        # Prepare features in the exact order your XGBoost model was trained on
        # Array shape must be (1, number_of_features)
        feature_matrix = np.array([[
        request.rolling_3d_neg,
        request.sentiment_pos, 
        request.sentiment_neg, 
        request.sentiment_neu, 
        request.price_momentum,
        request.rolling_3d_pos
        ]])
        
        # Calculate probabilities instead of just hard 0 or 1 classes
        # predict_proba returns [[prob_of_class_0, prob_of_class_1]]
        probabilities = model.predict_proba(feature_matrix)[0]
        prob_up = float(probabilities[1])  # Probability of class 1 (Up)
        
        # Custom Logic: Your combined probability math rule
        # Example: Ensembling model confidence with raw headline sentiment
        combined_prob = (prob_up * 0.6) + (request.sentiment_pos * 0.2) - (request.sentiment_neg * 0.2)
        # Keep boundary safety limits between 0.0 and 1.0
        combined_prob = max(0.0, min(1.0, combined_prob))
        
        # Final Classification Decision threshold
        decision = "Up" if combined_prob >= 0.5 else "Down/Flat"
        
        # Return the response mapped cleanly to your schema
        return PredictionResponse(
            ticker=request.ticker.upper(),
            date=request.date,
            model_probability_up=round(prob_up, 4),
            sentiment_score_finbert=round(request.sentiment_pos - request.sentiment_neg, 4),            
            combined_probability=round(combined_prob, 4),
            final_decision=decision
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine failure: {str(e)}")