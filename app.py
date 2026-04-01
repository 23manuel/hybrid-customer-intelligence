import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Intelligence OS", page_icon="⚡", layout="wide")

st.title("⚡ Customer Intelligence & LTV Radar")
st.markdown(
    "A production-grade ML pipeline that segments users and routes them to specialist XGBoost agents for accurate Lifetime Value (LTV) forecasting."
)

# --- LOAD MODELS (Cached for speed) ---
@st.cache_resource
def load_models_and_metadata():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Load Models
    scaler = joblib.load(os.path.join(BASE_DIR, 'feature_scaler.pkl'))
    kmeans = joblib.load(os.path.join(BASE_DIR, 'kmeans_segmenter.pkl'))
    agents = {i: joblib.load(os.path.join(BASE_DIR, f'clv_agent_{i}.pkl')) for i in range(4)}
    
    # Load the Semantic Map (The FIX)
    with open(os.path.join(BASE_DIR, 'persona_map.json'), 'r') as f:
        # JSON keys are strings, we cast them to ints for the model
        persona_map = {int(k): v for k, v in json.load(f).items()}
        
    return scaler, kmeans, agents, persona_map

scaler, kmeans, agents, persona_map = load_models_and_metadata()

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

        # 1. Prepare data for Segmentation
        segment_features = pd.DataFrame([[
            current_total_spent, num_transactions, avg_days_between_txns, total_credit_limit
        ]], columns=["total_spent", "num_transactions", "avg_days_between_txns", "total_credit_limit"])

        scaled_features = scaler.transform(segment_features)
        cluster_id = int(kmeans.predict(scaled_features)[0])

        personas = {
            0: "Volume Whale (High Spend/High Frequency)", # The math proves this is the Whale
            1: "High Credit User (Premium Target)", 
            2: "Casual User (Low Spend/Low Frequency)", 
            3: "Regular User (Core Base)"
        }

        persona = personas.get(cluster_id, "Unknown Segment")

        # 2. Prepare data for the Specialist Agent
        clv_features = pd.DataFrame([[
            yearly_income, total_debt, credit_score, total_credit_limit, current_age,
            num_transactions, avg_days_between_txns, account_lifespan_days
        ]], columns=[
            "yearly_income", "total_debt", "credit_score", "total_credit_limit", "current_age",
            "num_transactions", "avg_days_between_txns", "account_lifespan_days"
        ])

        specialist_agent = agents.get(cluster_id)
        if specialist_agent is None:
            st.error(f"No specialist agent found for cluster {cluster_id}.")
            st.stop()

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

        st.divider()
        st.markdown("""
        ### How this works under the hood:
        1. **Data Ingestion:** Captures raw financial and behavioral telemetry.
        2. **K-Means Routing:** Evaluates spending velocity and routes the user to 1 of 4 specialized profiles.
        3. **XGBoost Specialist Agents:** A dedicated model trained on similar users forecasts the final LTV.
        """)
