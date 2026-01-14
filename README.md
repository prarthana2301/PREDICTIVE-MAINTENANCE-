 FactoryGuard AI – IoT Predictive Maintenance Engine

 📌 Project Overview

"FactoryGuard AI" is an end-to-end **IoT Predictive Maintenance system** designed to predict catastrophic equipment failures **24 hours in advance** using time-series sensor data.

This project was developed as my **first project during my internship at Zaalima Development Pvt. Ltd.** and focuses on building a **production-ready machine learning pipeline** rather than just model experimentation.

The system targets a real-world manufacturing scenario involving **500 robotic arms** equipped with vibration, temperature, and pressure sensors, where unexpected downtime can result in massive financial losses.

Use Case

A critical manufacturing plant floor continuously streams sensor data from robotic arms. Failures are rare (<1%) but extremely costly.

Objective:

* Predict catastrophic failures "24 hours before occurrence"
* Enable scheduled preventive maintenance
* Reduce false alarms while maintaining high precision


 🧠 Key Features

🔹 Time-Series Feature Engineering

* Rolling Mean, Exponential Moving Average (EMA)
* Rolling Standard Deviation
* Lag Features (t-1, t-2)
* Time windows: **1 hour, 6 hours, 12 hours**
* Efficient feature generation using **Pandas**

 🔹 Modeling Strategy

* **Baseline Models:** Logistic Regression, Random Forest
* **Production Models:** XGBoost, LightGBM
* Hyperparameter tuning using **GridSearchCV / Optuna**

 🔹 Extreme Class Imbalance Handling

* Failure events < **1% of data**
* Accuracy explicitly avoided
* Evaluation Metric: **Precision-Recall AUC (PR-AUC)**
* Techniques used:
* Class weight adjustment (preferred)
* SMOTE (imbalanced-learn)

 🔹 Model Explainability

* SHAP (SHapley Additive Explanations)
* Local explanations for each failure prediction
* Helps maintenance engineers understand *why* a failure is predicted

🔹 Deployment & Inference

* End-to-end pipeline serialized using **joblib**
* **Flask REST API** for real-time inference
* Accepts JSON sensor payloads
* Response latency < **50ms**


📂 Project Structure

FactoryGuard-AI/
│
├── WEEK 1/
│   ├── FINAL_PREDICTION.ipynb     # Initial modeling & baseline experiments
│   ├── Documentation.docx         # Project requirement & design notes
│   └── machinery_data.xlsx        # Raw sensor dataset
│
├── WEEK 2/
│   └── factoryguard_scaler.pkl    # Saved preprocessing pipeline (scaler)
│
├── WEEK 3/
│   └── factoryguard_xgb_model.pkl # Trained XGBoost production model
│
├── WEEK 4/
│   ├── app.py                     # Flask API for real-time inference
│   └── style.css                  # Frontend styling
│
├── README.md                      # Project documentation
├── requirements.txt
├── venv/                          # Virtual environment (should be gitignored)
└── .ipynb_checkpoints/            # Jupyter checkpoints (should be gitignored)

 
📊 Evaluation Metrics

* **Primary Metric:** Precision-Recall AUC (PR-AUC)
* Precision prioritized to avoid unnecessary maintenance alerts
* Model validated for stability and latency



📈 SHAP Explainability Example

> "Failure predicted due to high rolling mean of temperature (>80°C) and sudden spike in vibration variance"

SHAP plots are included to provide **transparent, actionable insights** for maintenance engineers.

 🏢 Internship Context

* **Organization:** Zaalima Development Pvt. Ltd.
* **Project Type:** Internship – First  ML Project

 Follow the on-screen instructions to interact with the application, submit input data, and view predictions.
 <img width="1917" height="1007" alt="f1" src="https://github.com/user-attachments/assets/726b2505-844b-4a5c-a5bb-8323123d53f2" />

