"""
Employee Attrition Predictor (self-contained, no preprocessor.pkl needed)
---------------------------------------------------------------------------
Only file needed alongside this: best_catboost_model.pkl

Before running: paste the FEATURE_ORDER, CATEGORY_ORDERS, SCALER_MEAN and
SCALER_SCALE printed by generate_constants.py into the CONFIG section below.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "best_catboost_model.pkl"

# =========================== CONFIG (paste output of generate_constants.py here) ===========================
FEATURE_ORDER = [
    "Age", "Gender", "Years at Company", "Job Role", "Monthly Income", "Work-Life Balance",
    "Job Satisfaction", "Performance Rating", "Number of Promotions", "Overtime",
    "Distance from Home", "Education Level", "Marital Status", "Number of Dependents",
    "Job Level", "Company Size", "Company Tenure", "Remote Work", "Leadership Opportunities",
    "Innovation Opportunities", "Company Reputation", "Employee Recognition", "Age_group",
]

# PLACEHOLDER VALUES - replace with the real ones from generate_constants.py before deploying!
CATEGORY_ORDERS = {
    "Age_group": ["adult", "old", "teen", "too old"],
    "Gender": ["Female", "Male"],
    "Job Role": ["Education", "Finance", "Healthcare", "Media", "Technology"],
    "Work-Life Balance": ["Excellent", "Fair", "Good", "Poor"],
    "Job Satisfaction": ["High", "Low", "Medium", "Very High"],
    "Performance Rating": ["Average", "High", "Low", "Very High"],
    "Overtime": ["No", "Yes"],
    "Education Level": ["Associate Degree", "Bachelor’s Degree", "High School", "Master’s Degree", "PhD"],
    "Marital Status": ["Divorced", "Married", "Single"],
    "Job Level": ["Entry", "Mid", "Senior"],
    "Company Size": ["Large", "Medium", "Small"],
    "Remote Work": ["No", "Yes"],
    "Leadership Opportunities": ["No", "Yes"],
    "Innovation Opportunities": ["No", "Yes"],
    "Company Reputation": ["Excellent", "Fair", "Good", "Poor"],
    "Employee Recognition": ["High", "Low", "Medium", "Very High"],
}

SCALER_MEAN = [0.0] * len(FEATURE_ORDER)   # PLACEHOLDER - paste real SCALER_MEAN
SCALER_SCALE = [1.0] * len(FEATURE_ORDER)  # PLACEHOLDER - paste real SCALER_SCALE

AGE_BINS = [0, 19, 30, 60, 100]
AGE_LABELS = ["teen", "adult", "old", "too old"]

NUMERIC_COLS = [
    "Age", "Years at Company", "Monthly Income", "Number of Promotions",
    "Distance from Home", "Number of Dependents", "Company Tenure",
]
NUMERIC_RANGES = {
    "Age": (18, 60, 30),
    "Years at Company": (0, 50, 5),
    "Monthly Income": (1000, 20000, 5000),
    "Number of Promotions": (0, 10, 0),
    "Distance from Home": (1, 100, 10),
    "Number of Dependents": (0, 10, 0),
    "Company Tenure": (0, 100, 20),
}
# ============================================================================================================


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def encode(col: str, value):
    return CATEGORY_ORDERS[col].index(value)


def build_input_row(user_values: dict) -> pd.DataFrame:
    age_group = pd.cut([user_values["Age"]], bins=AGE_BINS, labels=AGE_LABELS)[0]
    user_values["Age_group"] = encode("Age_group", age_group)

    for col in CATEGORY_ORDERS:
        if col == "Age_group":
            continue
        user_values[col] = encode(col, user_values[col])

    row = np.array([[user_values[col] for col in FEATURE_ORDER]], dtype=float)
    scaled = (row - np.array(SCALER_MEAN)) / np.array(SCALER_SCALE)
    return pd.DataFrame(scaled, columns=FEATURE_ORDER)


def main():
    st.set_page_config(page_title="Employee Attrition Predictor", page_icon="📊", layout="centered")
    st.title("📊 Employee Attrition Predictor")
    st.caption("Fill in the employee details and get a real-time attrition risk prediction (CatBoost).")

    try:
        model = load_model()
    except FileNotFoundError:
        st.error(f"Missing `{MODEL_PATH}`. Put it in the repo root next to app.py.")
        st.stop()

    categorical_cols = [c for c in FEATURE_ORDER if c not in NUMERIC_COLS and c != "Age_group"]
    user_values = {}
    col1, col2 = st.columns(2)
    fields = categorical_cols + [c for c in NUMERIC_COLS if c in FEATURE_ORDER]

    for i, col in enumerate(fields):
        target = col1 if i % 2 == 0 else col2
        if col in categorical_cols:
            user_values[col] = target.selectbox(col, CATEGORY_ORDERS[col])
        else:
            lo, hi, default = NUMERIC_RANGES[col]
            user_values[col] = target.number_input(col, min_value=lo, max_value=hi, value=default)

    if st.button("Predict Attrition Risk", type="primary", use_container_width=True):
        X = build_input_row(user_values)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        attrition_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

        st.divider()
        if int(pred) == 1:
            st.error(f"⚠️ High attrition risk — {attrition_prob:.1%} probability of leaving")
        else:
            st.success(f"✅ Low attrition risk — {attrition_prob:.1%} probability of leaving")
        st.progress(min(max(attrition_prob, 0.0), 1.0))


if __name__ == "__main__":
    main()
