## Segmented Customer Intelligence Pipeline
A high-performance ML pipeline for behavioral segmentation, anomaly detection, and hybrid CLV forecasting.

Project Strategy
This project moves beyond standard "toy" datasets to solve a real-world financial services problem: How do we predict value when demographics are a weak signal? By pivoting from a failed demographic-only model to a Behavioral Hybrid Model, this pipeline achieved a significant reduction in prediction error ($MAE$) and identified the 20% of high-value users requiring human oversight.

Technical Innovation
  *   High-Performance ETL: Utilized Polars for memory-safe, lightning-fast data processing of large-scale transaction logs, significantly outperforming standard Pandas for initial data cleaning.
  *   Hybrid Modeling Strategy: Engineered a dual-input XGBoost architecture that merges static demographics with dynamic transaction intensity vectors.
  *   Specialist Agent Architecture: Rather than a "one-size-fits-all" model, I implemented four segment-specific XGBoost models. This "Specialist Agent" approach reduced prediction error for mid-tier users by nearly 10%.
  *   Anomaly Detection: Deployed Isolation Forest to flag the top 1% of transactions, identifying potential fraud and high-value outliers for manual audit.

Business Outcomes
  *   Data-Driven Pivot: Identified that churn was only 1%, allowing the business to reallocate resources from "Retention" to "LTV Optimization."
  *   Improved Accuracy: Reduced system-wide forecasting error from $111k to $108k, with localized error for regular users dropping to $49k.
  *   Risk Mitigation: Automated flagging of large refunds and chargebacks, protecting bottom-line revenue.

The Tech Stack
  *   Processing: Polars (Big Data), Pandas (ML compatibility)
  *   Machine Learning: Scikit-Learn (K-Means, Isolation Forest), XGBoost (Gradient Boosting)
  *   Visualization: Seaborn, Matplotlib
  *   Deployment Ready: Serialized models using Joblib for API integration.

Project Structure (The 6 Epics)
  *   Foundation: Polars-based ETL and Schema alignment.
  *   Feature Engineering: Transaction intensity and frequency mapping.
  *   Segmentation: K-Means clustering into 4 behavioral archetypes.
  *   Security: Anomaly detection via Isolation Forest.
  *   Forecasting: Hybrid XGBoost CLV modeling.
  *   Optimization: Segmented "Specialist Agent" deployment.


Intellectual Property Notice
Copyright © 2026 Emmanuel Okon.
This repository contains proprietary business logic and architectural designs. Access is strictly for auditing and recruitment review. Unauthorized distribution or replication of the "Washh Engine" is prohibited.
