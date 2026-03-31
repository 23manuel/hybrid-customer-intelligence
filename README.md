**Customer Intelligence ML Platform**
_(Formerly hybrid-customer-intelligence)_

A production-grade machine learning system that processes massive transaction logs to deliver customer segmentation, fraud detection, and segmented Lifetime Value (CLV) forecasting.

**TABLE**
   *  Problem Statement
   *  System Architecture
   *  Pipeline Stages
   *  Model Performance
   *  Business Applications
   *  Quick Start (Running the Pipeline)

_____________________________________________________________________________________________________________________________
**1. Problem Statement:**
Standard demographic models fail to accurately predict customer value, and basic churn models are useless in high-retention (99%) environments. This system shifts the focus from predicting churn to identifying revenue drivers, flagging anomalies, and deploying specialized ML agents to forecast LTV based on behavioral intensity.

**2. System Architecture:**
The pipeline is designed for memory efficiency and high-velocity processing, utilizing Polars to handle 15M+ rows under a strict 1.4GB memory limit before feeding structured features into the modeling layers.

                    Raw Transaction Logs (15M+ rows)
                               │
                               ▼
        Data Processing & Feature Engineering (Polars)
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
           Fraud Detection        Behavioral Segmentation
           (Isolation Forest)        (K-Means)
                    │                     │
                    ▼                     ▼
           Suspicious Alerts       Segmented CLV Forecasting
                                   (4× XGBoost Agents)
                    │                     │
                    └──────────┬──────────┘
                               ▼
                   Business Intelligence Outputs


                         
**3. Pipeline Stages:**
  *  Data Ingestion & Memory Optimization: Polars lazy evaluation is used to process 15M+ transaction records. Engineered features include avg_days_between_txns and account_lifespan_days.
  *  Anomaly & Fraud Detection: Analyzes a 5M row sample to flag the top 1% of extreme transaction anomalies (fraud/chargebacks).
  *  Customer Segmentation: K-Means clustering creates 4 distinct profiles based on spending volume and engagement.
  *  Hybrid Engine Setup: Merges demographic data with behavioral intensity metrics.
  *  Specialist Agent Routing: Instead of one global model, the system routes data to 4 distinct XGBoost models, each tuned to a specific customer segment for higher forecasting accuracy.

                            Raw Data 
                               │
                               ▼
                       Data Processing 
                               │
                               ▼
                         Feature Store         
                               │
                               ▼
                             Models         

**4. Model Performance**
  *  Churn Prediction: Deprecated. Exploratory Data Analysis revealed 99% retention, making churn prediction obsolete. System pivoted to LTV maximization.
  *  Fraud Detection (Isolation Forest): Successfully flagged top 1% isolated anomalies (Contamination: 0.01) across a 5M transaction sample.
  *  Customer Segmentation (K-Means): Identified 4 actionable cohorts (Casuals, High Credit, Regulars, Volume Whales).
  * CLV Forecasting (4x XGBoost Agents): * Baseline Model Error: $111,011 MAE
  *  Specialist Agent System Error: $108,000 MAE (Error reduced drastically for the 80% 'Everyday User' segment to ~$49k MAE).

**5. Business Applications**
  *  Targeted Marketing: Allocating heavy marketing spend only to 'Volume Whales' and 'High Credit' users.
  *  Risk Mitigation: Real-time flagging of compromised accounts and massive chargeback risks.
  *  Revenue Forecasting: Providing finance teams with realistic CLV estimations based on actual user behavior, not just static demographics.

**6. Quick Start (Running the Pipeline)**
All models and transformers are serialized via joblib for direct FastAPI deployment.

**Install dependencies**
pip install polars pyarrow pandas scikit-learn xgboost

**Run the pipeline**
python Segmented_CLV_Forecasting(1).py
