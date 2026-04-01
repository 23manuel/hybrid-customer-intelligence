# Customer Intelligence ML Platform

*(Formerly hybrid-customer-intelligence)*

A production-grade machine learning system that processes massive transaction logs to deliver **customer segmentation, fraud detection, and segmented Lifetime Value (CLV) forecasting**.

| Capability                  | Technology                   | Outcome                                           |
| --------------------------- | ---------------------------- | ------------------------------------------------- |
| Large-Scale Data Processing | Polars Lazy Engine           | Processes **15M+ transactions** within ~1.4GB RAM |
| Behavioral Segmentation     | Scikit-Learn K-Means         | Identifies **4 distinct customer personas**       |
| Fraud Detection             | Isolation Forest             | Flags **top 1% anomalous transactions**           |
| CLV Forecasting             | Segmented XGBoost Models     | Specialized predictions per customer cohort       |
| Memory Optimization         | Polars + Feature Downcasting | Handles large datasets efficiently                |
| Deployment Ready            | Joblib Serialization         | Models exportable for **FastAPI / Streamlit**     |

---

# Table of Contents

* Problem Statement
* System Architecture
* Pipeline Stages
* Model Performance
* Production Post-Mortem: Persona Labeling Bug
* Business Applications
* Quick Start (Running the Pipeline)

---

# 1. Problem Statement

Standard demographic models fail to accurately predict customer value, and basic churn models are ineffective in **high-retention environments (~99%)**.

Instead of predicting churn, this system focuses on:

* identifying **revenue-driving customer behaviors**
* detecting **anomalous transactions**
* forecasting **Customer Lifetime Value using segmented specialist models**

The platform shifts from traditional churn analytics to **behavior-driven intelligence**.

---

# 2. System Architecture

The pipeline is designed for **memory efficiency and high-throughput processing**, using **Polars lazy execution** to process **15M+ transaction records** under a strict **~1.4GB RAM constraint** before feeding structured features into modeling layers.

```
Raw Transaction Logs (15M+ rows)
           │
           ▼
Data Processing & Feature Engineering (Polars)
           │
   ┌───────┴────────┐
   ▼                ▼
Fraud Detection   Behavioral Segmentation
(Isolation Forest)      (K-Means)
   │                │
   ▼                ▼
Suspicious Alerts  Segmented CLV Forecasting
                   (4× XGBoost Agents)
   │                │
   └───────┬────────┘
           ▼
Business Intelligence Outputs
```

---

# 3. Pipeline Stages

### Data Ingestion & Memory Optimization

Polars lazy execution processes **15M+ transaction records** while maintaining a low memory footprint.

Key engineered features:

* `avg_days_between_txns`
* `account_lifespan_days`
* `days_since_last`
* `num_transactions`
* `total_spent`

---

### Anomaly & Fraud Detection

Isolation Forest analyzes a **5M transaction sample** to detect the most extreme behaviors.

Configuration:

* `n_estimators = 100`
* `contamination = 0.01`

The model flags the **top 1% of isolated transactions**, typically representing:

* fraud attempts
* extreme purchases
* large chargebacks

---

### Customer Segmentation

K-Means clustering groups customers into **4 behavioral personas** based on engagement and spending intensity.

Segmentation features:

* total spending
* transaction frequency
* transaction cadence
* credit capacity

---

### Hybrid Feature Engine

The system merges:

**Demographic attributes**

* yearly income
* debt
* credit score
* credit limits
* age

with

**Behavioral intensity metrics**

* number of transactions
* account lifespan
* transaction cadence

This hybrid feature space dramatically improves predictive signal strength.

---

### Specialist Agent Routing

Instead of using a single global model, the system deploys **four specialized XGBoost agents**, each trained on a specific customer segment.

```
Raw Data
   │
   ▼
Data Processing
   │
   ▼
Feature Store
   │
   ▼
Segment Router
   │
   ▼
4 Specialist XGBoost Models
```

Segment specialization reduces model confusion and improves prediction stability.

---

# 4. Model Performance

### Churn Prediction

Exploratory analysis revealed **~99% retention**, making churn modeling ineffective.

The system pivoted toward **LTV maximization and behavioral intelligence**.

---

### Fraud Detection (Isolation Forest)

* Dataset: 5M transactions
* Contamination: **1%**

The model successfully flags extreme anomalous behavior across large transaction streams.

---

### Customer Segmentation (K-Means)

Four behavioral cohorts were identified:

| Segment           | Description                                   |
| ----------------- | --------------------------------------------- |
| Casual Users      | Low spending and engagement                   |
| Regular Users     | Moderate spending patterns                    |
| High Credit Users | High credit limits and spending potential     |
| Volume Whales     | Extremely high spend and transaction activity |

---

### CLV Forecasting (XGBoost Specialist Agents)

| Model                   | Error (MAE) |
| ----------------------- | ----------- |
| Baseline Single Model   | $111,011    |
| Specialist Agent System | ~$108,000   |

For the **largest user segment (~80% of customers)**, prediction error improved significantly to:

**~$49k MAE**

This allows reliable forecasting for the majority of users while flagging high-value outliers for human oversight.

---

# 5. Production Post-Mortem

## Persona Labeling Bug

During early deployment of the **Segmented CLV system**, the UI occasionally displayed incorrect persona labels.

Example:

A customer predicted to generate **$150k+ lifetime value** appeared in the interface as **"Casual User."**

The model predictions were correct — but the **persona interpretation layer was flawed.**

---

## Root Cause

The system assumed that **K-Means cluster IDs had fixed meanings.**

Example implementation:

```python
personas = {
0: "Casual",
1: "Regular",
2: "High Credit",
3: "Whale"
}
```

This assumption is incorrect.

K-Means assigns **arbitrary cluster IDs**, which can change after:

* retraining
* dataset changes
* environment differences

Thus the same behavioral cluster might appear under a different integer label.

---

## Impact

This created a **semantic mismatch between predictions and personas**, reducing trust in the system despite correct model outputs.

---

## Production Fix: Dynamic Persona Mapping

Instead of trusting cluster IDs, the system now **derives personas from centroid characteristics**.

Clusters are ranked by **behavioral value metrics** such as `total_spent`.

Example logic:

```python
centroids = kmeans.cluster_centers_
cluster_rank = np.argsort(centroids[:, spend_feature_index])

persona_map = {
cluster_rank[0]: "Casual",
cluster_rank[1]: "Regular",
cluster_rank[2]: "High Credit",
cluster_rank[3]: "Whale"
}
```

The UI now references this dynamic mapping instead of raw cluster IDs.

---

## Additional Fixes

During debugging several secondary issues were resolved:

**NumPy Prediction Handling**

```python
cluster_id = int(kmeans.predict(features)[0])
```

**Feature Order Enforcement**

Training and inference pipelines now enforce identical feature schemas.

**Behavioral Metric Adjustment**

Transaction cadence estimation improved by using **transaction gaps instead of transaction count**.

---

## Result

After implementing the fix:

* Persona labels remain stable across retraining cycles
* UI semantics match model behavior
* the segmented CLV pipeline became **production-safe**

---

# 6. Business Applications

### Targeted Marketing

Marketing spend can focus on **Volume Whales** and **High Credit Users**, while minimizing cost on low-value segments.

---

### Risk Monitoring

The anomaly engine flags extreme transactions, helping detect:

* compromised accounts
* fraud attempts
* chargeback patterns

---

### Revenue Forecasting

Finance teams gain realistic **CLV projections based on behavioral data**, not static demographics.

---

# 7. Quick Start

All trained models and transformers are serialized with **joblib** for deployment.

API Deployment (FastAPI):
To run the backend API locally for system integration:
uvicorn api:app --reload
Navigate to http://localhost:8000/docs to interact with the auto-generated Swagger UI and test the /predict_ltv endpoint.

---

## Install Dependencies

```
pip install polars pyarrow pandas scikit-learn xgboost
```

---

## Run the Pipeline

```
python Segmented_CLV_Forecasting(1).py
```


