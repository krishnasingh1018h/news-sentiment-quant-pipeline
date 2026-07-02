import streamlit as st
import requests
import datetime
import plotly.graph_objects as go
import plotly.express as px

# 1. Page Configuration & Theme Initialization
st.set_page_config(
    page_title="PulseStream • Quant Sentiment Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium UI CSS Overrides (Dark-mode optimized, neon terminal aesthetics)
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@100;300;400;700&family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Code block override */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Premium custom card style */
    .metric-card {
        background: linear-gradient(135deg, #131c31 0%, #0f1626 100%);
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }
    
    .metric-card-positive {
        border-left: 5px solid #10b981;
    }
    
    .metric-card-negative {
        border-left: 5px solid #ef4444;
    }

    .metric-title {
        color: #94a3b8;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
    }

    .metric-delta {
        font-size: 14px;
        font-weight: 500;
        margin-top: 6px;
    }
    
    /* Status Badge styling */
    .status-badge {
        background-color: #1e293b;
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# 2. Header and Dashboard Title
col_title, col_status = st.columns([8, 2])
with col_title:
    st.markdown("<h1 style='margin-bottom: 0px; font-weight: 800; background: linear-gradient(to right, #60a5fa, #3b82f6, #1d4ed8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>PULSESTREAM QUANT ADVANCED TERMINAL</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 16px; margin-top: 5px;'>Continuous Alpha Pipeline Engine • Machine Learning (XGBoost) × NLP Deep-Learning Sentiment (FinBERT)</p>", unsafe_allow_html=True)
with col_status:
    st.markdown("<div style='text-align: right; margin-top: 15px;'><span class='status-badge'>● HYBRID ENGINE ONLINE</span></div>", unsafe_allow_html=True)

st.write("")

# 3. Sidebar Config Controls
st.sidebar.markdown("### 🎛️ Terminal Configuration")
FASTAPI_URL = st.sidebar.text_input("Backend API Gateway URL", value="http://127.0.0.1:8000/predict")
confidence_threshold = st.sidebar.slider("Decision Alpha Threshold", min_value=0.20, max_value=0.80, value=0.50, step=0.05)

# 4. Main Control Dashboard Panel
col_input, col_info = st.columns([4, 6])

with col_input:
    st.markdown("""
    <div class='metric-card' style='height: 100%; margin-bottom: 0px;'>
        <h3 style='margin-top: 0px; color: #60a5fa; font-weight: 700;'>🎯 Execute Hybrid Inference</h3>
        <p style='color: #94a3b8; font-size: 14px;'>Run live web scrapes or backtest against pre-computed SQL features.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        ticker = st.text_input("Target Equity Ticker (e.g., NVDA, INTC, AAPL):", value="NVDA", max_chars=8).strip().upper()
        
        # Mode selector to change behavior
        pipeline_mode = st.radio(
            "Select Pipeline Execution Mode:",
            options=["🟢 Live Real-Time Forecast (Today)", "📁 Historical Database Backtest (Past)"],
            horizontal=True
        )
        
        if "Live" in pipeline_mode:
            target_date = datetime.date.today()
            st.markdown(f"<div style='background-color: rgba(56, 189, 248, 0.1); border: 1px solid #38bdf8; border-radius: 8px; padding: 10px; color: #38bdf8; font-weight: 600; font-family: \"JetBrains Mono\", monospace;'>📡 INGESTION TIME: {target_date.strftime('%Y-%m-%d')} (TODAY)</div>", unsafe_allow_html=True)
            st.markdown("<span style='color: #64748b; font-size: 12px;'>Scrapes current headlines and live momentum values in real-time.</span>", unsafe_allow_html=True)
        else:
            target_date = st.date_input("Inference Evaluation Horizon Date:", value=datetime.date.today() - datetime.timedelta(days=1))
            st.markdown(f"<div style='background-color: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 8px; padding: 10px; color: #f59e0b; font-weight: 600; font-family: \"JetBrains Mono\", monospace;'>📂 DATABASE QUERY: {target_date.strftime('%Y-%m-%d')}</div>", unsafe_allow_html=True)
            st.markdown("<span style='color: #64748b; font-size: 12px;'>Queries the historical SQL database to retrieve exact features from this date.</span>", unsafe_allow_html=True)
        
        st.write("")
        # Action Trigger
        trigger_prediction = st.button("RUN PIPELINE EXECUTION", type="primary", use_container_width=True)

with col_info:
    st.markdown(f"""
    <div class='metric-card' style='height: 100%;'>
        <h3 style='margin-top: 0px; color: #94a3b8; font-weight: 700;'>⚙️ Automated Operations Chain</h3>
        <p style='color: #64748b; font-size: 13px; line-height: 1.6; margin-bottom: 5px;'>
            1. <b>Routing Assessment:</b> Evaluates target date. Live inputs trigger scraper pipelines; historical target dates route directly to SQL Feature Tables.<br>
            2. <b>NLP Sentiment Extraction:</b> Runs raw strings through the active FinBERT pipeline to generate positive, negative, and neutral probability vectors.<br>
            3. <b>Market Momentum Indexing:</b> Evaluates 5-day rolling price trends via yfinance.<br>
            4. <b>XGBoost Inference Classification:</b> Evaluates compiled feature vectors against historical models.<br>
            5. <b>Consolidated Decisioning:</b> Triggers the final movement direction outcome.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# 5. Pipeline Run Event Handlers & Output Visualization
if trigger_prediction:
    if not ticker:
        st.warning("Please specify a valid stock ticker before executing prediction.")
    else:
        # Beautiful progress indicators to keep user visually engaged
        status_container = st.container()
        with status_container:
            st.markdown("<h4 style='color: #60a5fa;'>⛓️ Executing Stage Sequences...</h4>", unsafe_allow_html=True)
            progress_bar = st.progress(0)
            
            # Step-by-step progress simulation to align with network calls
            import time
            if "Live" in pipeline_mode:
                progress_bar.progress(20, text="Scraping live Yahoo Finance news & market metrics...")
                time.sleep(0.5)
                progress_bar.progress(50, text="Analyzing headlines via FinBERT Transformer Pipeline...")
            else:
                progress_bar.progress(30, text="Searching SQLite Feature Store...")
                time.sleep(0.3)
                progress_bar.progress(70, text="Retrieving historical alignment metrics...")
            
            try:
                # API Call to active FastAPI backend
                payload = {"ticker": ticker, "date": str(target_date)}
                response = requests.post(FASTAPI_URL, json=payload, timeout=45)
                
                if response.status_code == 200:
                    progress_bar.progress(85, text="Compiling feature matrix & pushing to XGBoost Model...")
                    time.sleep(0.3)
                    progress_bar.progress(100, text="Inference Complete!")
                    
                    data = response.json()
                    status_container.empty() # Clear progress logs once successful
                    
                    # Store variables for easy plotting
                    final_decision = data["final_decision"]
                    model_prob_up = data["model_probability_up"]
                    sentiment_score = data["sentiment_score_finbert"]
                    combined_prob = data["combined_probability"]
                    features = data["live_features_extracted"]
                    
                    # Add decision coloring logic
                    is_bullish = final_decision.upper() == "UP"
                    card_color = "metric-card-positive" if is_bullish else "metric-card-negative"
                    accent_color = "#10b981" if is_bullish else "#ef4444"
                    
                    # Display Final Outcome Main Panel
                    st.markdown(f"""
                    <div class='metric-card {card_color}'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <div class='metric-title'>Pipeline Directional Outlook</div>
                                <div style='font-size: 48px; font-weight: 800; color: {accent_color}; margin: 5px 0;'>
                                    {final_decision.upper()}
                                </div>
                                <div style='color: #94a3b8; font-size: 14px;'>
                                    Model predicts a <b>{"BULLISH" if is_bullish else "BEARISH/FLAT"}</b> trend trajectory for <b>{ticker}</b> evaluated on <b>{target_date}</b>.
                                </div>
                            </div>
                            <div style='text-align: right;'>
                                <div class='metric-title'>Alpha Confidence index</div>
                                <div style='font-size: 54px; font-weight: 800; color: #ffffff;'>
                                    {round(combined_prob * 100, 1)}%
                                </div>
                                <div style='color: #64748b; font-size: 12px;'>Threshold for Up: {confidence_threshold * 100}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 6. High-End Visualization Layout (Two Column Charts Grid)
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        # Donut Chart for Sentiment Class Breakdown
                        labels = ['Positive Sentiment', 'Negative Sentiment', 'Neutral Sentiment']
                        values = [features['sentiment_positive'], features['sentiment_negative'], features['sentiment_neutral']]
                        colors = ['#10b981', '#ef4444', '#64748b']
                        
                        fig_sentiment = go.Figure(data=[go.Pie(
                            labels=labels, 
                            values=values, 
                            hole=.5,
                            marker=dict(colors=colors, line=dict(color='#0b0f19', width=2)),
                            textinfo='percent+label'
                        )])
                        
                        fig_sentiment.update_layout(
                            title=dict(text="FinBERT Sentiment Weight Distribution", font=dict(color='#ffffff', size=16)),
                            paper_bgcolor='rgba(15, 22, 38, 1)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#94a3b8'),
                            showlegend=False,
                            margin=dict(t=50, b=20, l=20, r=20),
                            height=320
                        )
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                    with col_chart2:
                        # Radial Gauge Chart for XGBoost Confidence Probability Output
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = model_prob_up * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "XGBoost Historical Classifier Up-Prob", 'font': {'color': '#ffffff', 'size': 16}},
                            gauge = {
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                                'bar': {'color': "#3b82f6"},
                                'bgcolor': "#1e293b",
                                'borderwidth': 2,
                                'bordercolor': "#334155",
                                'steps': [
                                    {'range': [0, 40], 'color': '#131c31'},
                                    {'range': [40, 60], 'color': '#1e293b'},
                                    {'range': [60, 100], 'color': '#1e3a8a'}
                                ],
                                'threshold': {
                                    'line': {'color': "#10b981", 'width': 4},
                                    'thickness': 0.75,
                                    'value': confidence_threshold * 100
                                }
                            }
                        ))
                        
                        fig_gauge.update_layout(
                            paper_bgcolor='rgba(15, 22, 38, 1)',
                            font=dict(color='#94a3b8'),
                            margin=dict(t=50, b=20, l=20, r=20),
                            height=320
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                    # 7. Secondary Metrics Grid Rows
                    st.subheader("📋 Feature Engineering Breakdown Vector")
                    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                    
                    with col_f1:
                        st.markdown(f"""
                        <div class='metric-card' style='padding: 16px; margin-bottom: 0px;'>
                            <div class='metric-title'>Price Momentum (5d)</div>
                            <div class='metric-value' style='font-size: 24px; color: {"#10b981" if features["price_momentum"] >= 0 else "#ef4444"};'>
                                {round(features["price_momentum"] * 100, 3)}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_f2:
                        st.markdown(f"""
                        <div class='metric-card' style='padding: 16px; margin-bottom: 0px;'>
                            <div class='metric-title'>FinBERT Net Score</div>
                            <div class='metric-value' style='font-size: 24px; color: #38bdf8;'>
                                {sentiment_score}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_f3:
                        st.markdown(f"""
                        <div class='metric-card' style='padding: 16px; margin-bottom: 0px;'>
                            <div class='metric-title'>Rolling 3d Pos Mean</div>
                            <div class='metric-value' style='font-size: 24px;'>
                                {features["rolling_3d_pos"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_f4:
                        st.markdown(f"""
                        <div class='metric-card' style='padding: 16px; margin-bottom: 0px;'>
                            <div class='metric-title'>Rolling 3d Neg Mean</div>
                            <div class='metric-value' style='font-size: 24px;'>
                                {features["rolling_3d_neg"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                elif response.status_code == 404:
                    status_container.empty()
                    st.error(f"❌ Target Data Mismatch: {response.json().get('detail', 'No record matches this criteria')}")
                else:
                    status_container.empty()
                    st.error(f"❌ API Core Failure: Server responded with status code {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                status_container.empty()
                st.error("🚨 Connection Refused: Ensure your FastAPI backend server is actively hosted on uvicorn (`http://127.0.0.1:8000`) before running inference predictions.")
            except Exception as e:
                status_container.empty()
                st.error(f"⚠️ App-level runtime exception: {str(e)}")

# Footer Branding
st.markdown("<hr style='border: 1px solid #1e293b;' />", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569; font-size: 12px; font-family: \"JetBrains Mono\", monospace;'>PULSESTREAM QUANT STACK v3.1.0 • CONTAINER-READY SECURED ENVIRONMENT</p>", unsafe_allow_html=True)