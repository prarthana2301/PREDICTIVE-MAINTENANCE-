import streamlit as st
import numpy as np
import joblib

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="FactoryGuard AI",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ FactoryGuard AI")
st.subheader("IoT Predictive Maintenance Engine")

# =========================
# Load Model & Scaler
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("factoryguard_xgb_model.pkl")
    scaler = joblib.load("factoryguard_scaler.pkl")
    return model, scaler

model, scaler = load_model()

# =========================
# Prediction Function (FIXED)
# =========================
def predict_failure(sensor_1, sensor_2, sensor_3):
    """
    Model was trained with 46 features.
    UI collects 3 key sensors.
    Remaining features are initialized with baseline values (0).
    """

    # Create input with SAME number of features used in training
    X = np.zeros((1, 46))

    # Assign user inputs to first 3 sensors
    X[0, 0] = sensor_1
    X[0, 1] = sensor_2
    X[0, 2] = sensor_3

    # Scale using trained scaler
    X_scaled = scaler.transform(X)

    # Predict failure probability
    prob = model.predict_proba(X_scaled)[0][1]

    return prob

# =========================
# Sidebar
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Input Data", "Results"]
)

# =========================
# Home Page
# =========================
if page == "Home":
    st.markdown("""
    ### 📌 About FactoryGuard AI

    FactoryGuard AI predicts **catastrophic machine failure** in advance using
    sensor data from industrial equipment.

    **Model Highlights**
    - XGBoost (Production Model)
    - Optimized using PR-AUC
    - Handles extreme class imbalance
    - SHAP-based explainability (offline)

    **Deployment**
    - Trained offline in Jupyter
    - Deployed using Streamlit
    """)

# =========================
# Input Page
# =========================
elif page == "Input Data":
    st.header("🔧 Enter Sensor Readings")

    sensor_1 = st.number_input("Sensor 1 (Vibration)", value=0.0, step=0.01)
    sensor_2 = st.number_input("Sensor 2 (Temperature)", value=0.0, step=0.01)
    sensor_3 = st.number_input("Sensor 3 (Pressure)", value=0.0, step=0.01)

    if st.button("Submit"):
        st.session_state["input_features"] = [sensor_1, sensor_2, sensor_3]
        st.success("Sensor values submitted successfully!")

# =========================
# Results Page
# =========================
elif page == "Results":
    st.header("📊 Prediction Results")

    if "input_features" not in st.session_state:
        st.warning("Please enter sensor values in the Input Data page.")
    else:
        s1, s2, s3 = st.session_state["input_features"]

        prob = predict_failure(s1, s2, s3)

        st.metric(
            label="Failure Probability",
            value=f"{prob * 100:.2f}%"
        )

        # Risk interpretation
        if prob >= 0.7:
            st.error("🔴 High Risk – Immediate maintenance required")
        elif prob >= 0.4:
            st.warning("🟠 Medium Risk – Monitor closely")
        else:
            st.success("🟢 Low Risk – Machine healthy")

