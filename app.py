import streamlit as st
import pandas as pd
import joblib
import json
import os
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Intelligence OS", page_icon="⚡", layout="wide")

st.title("⚡ Customer Intelligence & LTV Radar")
st.markdown("A production-grade ML pipeline that segments users and routes them to specialist XGBoost agents.")

# --- LOAD MODELS & METADATA ---
@st.cache_resource
def load_models_and_metadata():
    # Point directly to the NEW artifacts folder
    BASE_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
    
    scaler = joblib.load(os.path.join(BASE_DIR, 'feature_scaler.pkl'))
    kmeans = joblib.load(os.path.join(BASE_DIR, 'kmeans_segmenter.pkl'))
    agents = {i: joblib.load(os.path.join(BASE_DIR, f'clv_agent_{i}.pkl')) for i in range(4)}
    
    with open(os.path.join(BASE_DIR, 'persona_map.json'), 'r') as f:
        persona_map = {int(k): v for k, v in json.load(f).items()}
        
    return scaler, kmeans, agents, persona_map

try:
    scaler, kmeans, agents, persona_map = load_models_and_metadata()
except Exception as e:
    st.error(f"Failed to load artifacts from Drive. Error: {e}")
    st.stop()

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("📡 Customer Telemetry")

with st.sidebar.expander("Demographics & Financials", expanded=True):
    yearly_income = st.number_input("Yearly Income ($)", min_value=0, value=45000, step=1000)
    total_debt = st.number_input("Total Debt ($)", min_value=0, value=15000, step=1000)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=498)
    total_credit_limit = st.number_input("Total Credit Limit ($)", min_value=0, value=500, step=1000)
    current_age = st.slider("Current Age", min_value=18, max_value=100, value=34)

with st.sidebar.expander("Behavioral Intensity", expanded=True):
    current_total_spent = st.number_input("Total Spent to Date ($)", min_value=0, value=80, step=10)
    num_transactions = st.number_input("Number of Transactions", min_value=1, value=12)
    avg_days_between_txns = st.number_input("Avg Days Between Swipes", min_value=1.0, value=6.1)
    account_lifespan_days = st.number_input("Account Lifespan (Days)", min_value=1, value=365)

# --- PREDICTION LOGIC ---
if st.sidebar.button("Run Intelligence Engine 🚀"):
    with st.spinner("Processing Telemetry..."):

        # 1. Routing Engine (K-Means)
        segment_features = pd.DataFrame([[
            current_total_spent, num_transactions, avg_days_between_txns, total_credit_limit
        ]], columns=["total_spent", "num_transactions", "avg_days_between_txns", "total_credit_limit"])

        scaled_features = scaler.transform(segment_features)
        cluster_id = int(kmeans.predict(scaled_features)[0])
        persona = persona_map.get(cluster_id, "Unknown Segment")

        # 2. Specialist Agent (XGBoost)
        clv_features = pd.DataFrame([[
            yearly_income, total_debt, credit_score, total_credit_limit, current_age,
            num_transactions, avg_days_between_txns, account_lifespan_days
        ]], columns=[
            "yearly_income", "total_debt", "credit_score", "total_credit_limit", "current_age",
            "num_transactions", "avg_days_between_txns", "account_lifespan_days"
        ])

        specialist_agent = agents.get(cluster_id)
        predicted_ltv = float(specialist_agent.predict(clv_features)[0])

        # --- DISPLAY RESULTS ---
        st.success("Analysis Complete.")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Predicted Behavioral Segment", value=f"Cluster {cluster_id}")
            st.info(f"**Persona:** {persona}")

        with col2:
            st.metric(label="Forecasted Lifetime Value (LTV)", value=f"${predicted_ltv:,.2f}")
            st.warning(f"Routed via Agent {cluster_id} for segment-specific accuracy.")

        # --- OBSERVABILITY LAYER ---
        with st.expander("🛠️ SYSTEM AUDIT: REVEAL K-MEANS MATH", expanded=False):
            st.write("**1. Raw DataFrame Input:**", segment_features)
            st.write("**2. Scaled Array (What the AI actually sees):**", scaled_features)
            distances = kmeans.transform(scaled_features)
            st.write("**3. Euclidean Distance to Clusters:**", distances)
