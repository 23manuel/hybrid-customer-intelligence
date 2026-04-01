from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os
import uvicorn

# Global variables to hold models in memory
scaler, kmeans, agents, persona_map = None, None, None, None

# --- 1. LIFESPAN MANAGER (The Modern FastAPI Way) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything BEFORE the 'yield' happens on startup
    global scaler, kmeans, agents, persona_map
    BASE_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
    
    try:
        scaler = joblib.load(os.path.join(BASE_DIR, 'feature_scaler.pkl'))
        kmeans = joblib.load(os.path.join(BASE_DIR, 'kmeans_segmenter.pkl'))
        agents = {i: joblib.load(os.path.join(BASE_DIR, f'clv_agent_{i}.pkl')) for i in range(4)}
        
        with open(os.path.join(BASE_DIR, 'persona_map.json'), 'r') as f:
            persona_map = {int(k): v for k, v in json.load(f).items()}
        print("✅ Models loaded successfully into memory.")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        
    yield # The API is now running and accepting requests
    
    # Everything AFTER the 'yield' happens on shutdown
    print("🛑 Shutting down API and clearing memory...")
    scaler, kmeans, agents, persona_map = None, None, None, None

# --- 2. INITIALIZE API ---
app = FastAPI(
    title="Customer Intelligence & LTV API",
    description="Production API for routing customers and forecasting Lifetime Value.",
    version="1.0.0",
    lifespan=lifespan  # <-- We attach the lifespan manager here
)

# --- 3. DEFINE DATA CONTRACT (Pydantic) ---
class CustomerTelemetry(BaseModel):
    yearly_income: float
    total_debt: float
    credit_score: int
    total_credit_limit: float
    current_age: int
    total_spent: float
    num_transactions: int
    avg_days_between_txns: float
    account_lifespan_days: int

# --- 4. THE INFERENCE ENDPOINT ---
@app.post("/predict_ltv")
def predict_customer_value(data: CustomerTelemetry):
    try:
        # Step A: Routing (K-Means)
        segment_features = pd.DataFrame([[
            data.total_spent, data.num_transactions, 
            data.avg_days_between_txns, data.total_credit_limit
        ]], columns=["total_spent", "num_transactions", "avg_days_between_txns", "total_credit_limit"])

        scaled_features = scaler.transform(segment_features)
        cluster_id = int(kmeans.predict(scaled_features))
        persona = persona_map.get(cluster_id, "Unknown Segment")

        # Step B: Specialist Agent Inference (XGBoost)
        clv_features = pd.DataFrame([[
            data.yearly_income, data.total_debt, data.credit_score, data.total_credit_limit, 
            data.current_age, data.num_transactions, data.avg_days_between_txns, data.account_lifespan_days
        ]], columns=[
            "yearly_income", "total_debt", "credit_score", "total_credit_limit", "current_age",
            "num_transactions", "avg_days_between_txns", "account_lifespan_days"
        ])

        predicted_ltv = float(agents[cluster_id].predict(clv_features))

        # Step C: Return JSON Response
        return {
            "status": "success",
            "routing_details": {
                "assigned_cluster_id": cluster_id,
                "persona_label": persona
            },
            "forecast": {
                "predicted_lifetime_value_usd": round(predicted_ltv, 2)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
