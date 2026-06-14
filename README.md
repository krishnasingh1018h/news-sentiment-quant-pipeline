## 📊 About the Project

**news-sentiment-quant-pipeline** is a production-grade, end-to-end machine learning ecosystem designed to predict short-term stock price movements for high-volatility tech and semiconductor equities (e.g., NVDA, TSM, AMD, GOOGL). 

By bridging the gap between natural language processing (NLP) and quantitative financial engineering, the pipeline transforms raw, unorganized headline streams into clean, actionable alpha signals.



🏗️ Core Architecture & Pipeline Stages

1. ETL & Data Backfilling (`models/backfill.py`)**: Consolidates historical Kaggle financial headline data with real-time price streams from the `yfinance` API. Cleans, standardizes time-series alignments, and structures over 26,000 unified rows into a local SQLite database (`pulsestream.db`).
2. Feature Engineering**: Generates multi-dimensional feature matrices consisting of raw FinBERT sentiment weights (`positive`, `negative`, `neutral`), trailing technical metrics (`price_momentum`), and multi-day rolling sentiment trends (`rolling_3d_pos`, `rolling_3d_neg`).
3. ML Core Classifier (`models/train_classification.py`)**: Trains a highly optimized Gradient Boosted Decision Tree (**XGBoost Classifier**) to map complex interactions between market momentum and public sentiment into a binary direction prediction (`Up` vs `Down/Flat`).
4. Production Inference Engine (`app/fastapi.py`)**: A lightweight, asynchronous **FastAPI** web server that serializes the model weights via `joblib`, performs fast, isolated CPU-bound inference, and outputs structured, schema-validated JSON payloads via Pydantic.



🧰 Tech Stack
* Deep Learning NLP:** FinBERT (HuggingFace Transformers)
* Machine Learning:** XGBoost, Scikit-Learn
* Data Infrastructure:** Pandas, NumPy, SQLite3, YFinance
* API Gateway & Serving:** FastAPI, Pydantic, Uvicorn, Joblib
