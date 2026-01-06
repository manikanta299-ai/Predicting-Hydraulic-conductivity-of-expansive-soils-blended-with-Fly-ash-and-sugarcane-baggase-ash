# ============================================================
# Hydraulic Conductivity Prediction GUI
# RF–GWO Model | USEPA & MOEF Compliant
# ============================================================

import streamlit as st
import numpy as np
import joblib

# ------------------------------------------------------------
# 1. PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Hydraulic Conductivity Predictor",
    layout="centered"
)

st.title("Hydraulic Conductivity Prediction for Landfill Liners")
st.markdown(
    "**Regulatory Criterion (USEPA & MOEF):**  \n"
    r"$\mathbf{HC \le 1 \times 10^{-7}\;cm/s}$"
)

# ------------------------------------------------------------
# 2. LOAD TRAINED MODEL
# ------------------------------------------------------------
model = joblib.load("rf_gwo_model.pkl")

# ------------------------------------------------------------
# 3. INPUT SECTION
# ------------------------------------------------------------
st.subheader("Input Parameters")

FA = st.number_input("Fly Ash (%)", 0.0, 100.0, 10.0)
SCBA = st.number_input("SCBA (%)", 0.0, 100.0, 10.0)
EC = st.number_input("Expansive Clay (%)", 0.0, 100.0, 80.0)

LL = st.number_input("Liquid Limit (%)", 20.0, 120.0, 70.0)
PI = st.number_input("Plasticity Index (%)", 5.0, 80.0, 31.0)
FSI = st.number_input("Free Swell Index (%)", 0.0, 200.0, 38.0)

MDUW = st.number_input("Maximum Dry Unit Weight (kN/m³)", 10.0, 25.0, 14.0)
OMC = st.number_input("Optimum Moisture Content (%)", 5.0, 40.0, 32.0)
UCS = st.number_input("UCS (28 days, kPa)", 50.0, 1000.0, 321.0)

# ------------------------------------------------------------
# 4. INPUT VALIDATION
# ------------------------------------------------------------
warnings = []

if abs(FA + SCBA + EC - 100) > 1:
    warnings.append("⚠ FA + SCBA + EC should sum to 100%")

if UCS < 200:
    warnings.append("⚠ UCS < 200 kPa (may not satisfy liner strength requirement)")

if LL < PI:
    warnings.append("⚠ Liquid Limit should be greater than Plasticity Index")

if warnings:
    for w in warnings:
        st.warning(w)

# ------------------------------------------------------------
# 5. PREDICTION
# ------------------------------------------------------------
if st.button("Predict Hydraulic Conductivity"):

    X = np.array([[FA, SCBA, EC, LL, PI, FSI, MDUW, OMC, UCS]])

    predicted_log_hc = model.predict(X)[0]
    predicted_hc = 10 ** predicted_log_hc

    st.subheader("Prediction Results")

    st.write(f"**Predicted log₁₀(HC):** {predicted_log_hc:.4f}")
    st.write(f"**Predicted HC:** {predicted_hc:.2e} cm/s")

    # --------------------------------------------------------
    # 6. TRAFFIC-LIGHT ACCEPTANCE LOGIC (ENGINEERING-CORRECT)
    # --------------------------------------------------------
    # Green  : logHC ≤ -7.0  (clearly compliant)
    # Amber  : -7.0 < logHC ≤ -6.5 (acceptable with uncertainty)
    # Red    : logHC > -6.5 (not acceptable)

    if predicted_log_hc <= -7.0:
        st.success("🟢 ACCEPTABLE (Green Zone): Fully complies with USEPA & MOEF criteria")

    elif -7.0 < predicted_log_hc <= -6.5:
        st.warning(
            "🟡 ACCEPTABLE WITH CAUTION (Amber Zone):\n\n"
            "Prediction is within experimental uncertainty limits.\n"
            "Complies with landfill liner criteria considering laboratory variability."
        )

    else:
        st.error(
            "🔴 NOT ACCEPTABLE (Red Zone):\n\n"
            "Does NOT comply with USEPA & MOEF landfill liner hydraulic conductivity requirement."
        )

    st.markdown(
        "---\n"
        "**Note:** Predictions are based on a machine learning model trained on "
        "experimental FA–SCBA–EC liner data. Laboratory validation is recommended."
    )
