
import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Intelligence OS", page_icon="⚡", layout="wide")

st.title("⚡ Customer Intelligence & LTV Radar")
st.markdown("A production-grade ML pipeline that segments users and routes them to specialist XGBoost agents for accurate Lifetime Value (LTV) forecasting.")

# --- LOAD MODELS (Cached for speed) ---
@st.cache_resource
def load_models():
    # Ensure your .pkl files are in a folder named 'models' or in the same directory
    base_path = "" # Change this if your models are in a subfolder, e.g., "models/"
    
    scaler = joblib.load(base_path + 'feature_scaler.pkl')
    kmeans = joblib.load(base_path + 'kmeans_segmenter.pkl')
    
    agents = {
        0: joblib.load(base_path + 'clv_agent_0.pkl'),
        1: joblib.load(base_path + 'clv_agent_1.pkl'),
        2: joblib.load(base_path + 'clv_agent_2.pkl'),
        3: joblib.load(base_path + 'clv_agent_3.pkl')
    }
    return scaler, kmeans, agents

try:
    scaler, kmeans, agents = load_models()
except Exception as e:
    st.error(f"Error loading models. Ensure all .pkl files are in the repository. Details: {e}")
    st.stop()

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("Customer Telemetry Inputs")

st.sidebar.subheader("Demographics & Financials")
yearly_income = st.sidebar.number_input("Yearly Income ($)", min_value=0, value=65000, step=1000)
total_debt = st.sidebar.number_input("Total Debt ($)", min_value=0, value=15000, step=1000)
credit_score = st.sidebar.slider("Credit Score", min_value=300, max_value=850, value=720)
total_credit_limit = st.sidebar.number_input("Total Credit Limit ($)", min_value=0, value=25000, step=1000)
current_age = st.sidebar.slider("Current Age", min_value=18, max_value=100, value=34)

st.sidebar.subheader("Behavioral Intensity")
current_total_spent = st.sidebar.number_input("Total Spent to Date ($)", min_value=0, value=5000, step=100)
num_transactions = st.sidebar.number_input("Number of Transactions", min_value=1, value=45)
avg_days_between_txns = st.sidebar.number_input("Avg Days Between Swipes", min_value=1.0, value=7.5)
account_lifespan_days = st.sidebar.number_input("Account Lifespan (Days)", min_value=1, value=365)

# --- PREDICTION LOGIC ---
if st.sidebar.button("Run Intelligence Engine 🚀"):
    with st.spinner("Processing through ML Pipeline..."):
        
        # 1. Prepare data for Segmentation (Epic 4 features)
        segment_features = pd.DataFrame([[
            current_total_spent, num_transactions, avg_days_between_txns, total_credit_limit
        ]], columns=['total_spent', 'num_transactions', 'avg_days_between_txns', 'total_credit_limit'])
        
        # Scale and Predict Segment (FIXED: added inside the int)
        scaled_features = scaler.transform(segment_features)
        cluster_id = int(kmeans.predict(scaled_features))
        
        # Define Persona Names based on your Epic 4 findings
        personas = {
            0: "Casual User (Low Spend/Low Frequency)",
            1: "High Credit User (Premium Target)",
            2: "Regular User (Core Base)",
            3: "Volume Whale (High Spend/High Frequency)"
        }
        
        # 2. Prepare data for the Specialist Agent (Epic 6 features)
        clv_features = pd.DataFrame([[
            yearly_income, total_debt, credit_score, total_credit_limit, current_age,
            num_transactions, avg_days_between_txns, account_lifespan_days
        ]], columns=[
            'yearly_income', 'total_debt', 'credit_score', 'total_credit_limit', 'current_age',
            'num_transactions', 'avg_days_between_txns', 'account_lifespan_days'
        ])
        
        # Route to the correct XGBoost Agent (FIXED: added at the end)
        specialist_agent = agents[cluster_id]
        predicted_ltv = specialist_agent.predict(clv_features)
        
        # --- DISPLAY RESULTS ---
        st.success("Analysis Complete.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Predicted Behavioral Segment", value=f"Cluster {cluster_id}")
            st.info(f"**Persona:** {personas[cluster_id]}")
            
        with col2:
            st.metric(label="Forecasted Lifetime Value (LTV)", value=f"${predicted_ltv:,.2f}")
            st.warning("Routed via Agent " + str(cluster_id) + " for segment-specific accuracy.")
            
        st.divider()
        st.markdown("""
        ### How this works under the hood:
        1. **Data Ingestion:** Captures raw financial and behavioral telemetry.
        2. **K-Means Routing:** Evaluates spending velocity and routes the user to 1 of 4 specialized profiles.
        3. **XGBoost Specialist Agents:** Instead of a general, inaccurate model, a dedicated XGBoost model trained *only* on similar users forecasts the final LTV.
        """)
