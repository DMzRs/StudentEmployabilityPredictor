import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ------------------------------
# Load Model, Scaler, and Columns
# ------------------------------
loaded = joblib.load("Model/RandomForest_model.pkl")
model = loaded["model"]
scaler = loaded["scaler"]
columns = loaded["columns"]  # exact feature names used during training

# ------------------------------
# Title and Disclaimer
# ------------------------------
st.title("Student's Employability Prediction Model")
st.markdown(
    """
    **Disclaimer:**  
    This tool provides a prediction based on the given inputs and the trained machine learning model.  
    Results are for **educational and experimental purposes only** and should not be treated as an absolute measure of employability.
    """
)

# ------------------------------
# Custom CSS for styling
# ------------------------------
st.markdown(
    """
    <style>
    .criteria-title {
        font-size: 20px; 
        font-weight: 600; 
        margin-bottom: -10px;
    }
    .criteria-sub {
        font-size: 14px; 
        color: gray;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Criteria Definitions
# ------------------------------
criteria = {
    "General Appearance": "general_appearance",
    "Manner Of Speaking": "speaking_manner",
    "Physical Condition": "physical_condition",
    "Mental Alertness": "mental_alertness",
    "Self Confidence": "self_confidence",
    "Ability to Present Ideas": "ability_to_present_ideas",
    "Communication Skills": "communication_skills",
    "Student Performance Rating": "spr"
}

# ------------------------------
# Collect User Inputs
# ------------------------------
user_inputs = {}
for display_name, key_name in criteria.items():
    st.markdown(f"<p class='criteria-title'>{display_name}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='criteria-sub'>Rate from 5 (Excellent) to 1 (Poor)</p>", unsafe_allow_html=True)

    choice = st.radio(
        f"Select rating for {display_name}", 
        options=[5, 4, 3, 2, 1], 
        index=None,
        horizontal=True,
        key=key_name
    )
    user_inputs[key_name] = choice

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict"):
    if None in user_inputs.values():
        st.warning("⚠️ Please rate all criteria before predicting.")
    else:
        # Convert inputs into DataFrame with correct column names & order
        X_new_df = pd.DataFrame([user_inputs], columns=columns)

        # Apply scaling
        X_new_scaled = scaler.transform(X_new_df)

        # Predict
        prediction = model.predict(X_new_scaled)[0]

        # Map prediction to label
        label_map = {0: "Less Employable", 1: "Employable"}
        prediction_label = label_map.get(prediction, "Unknown")

        st.success(f"🎯 Prediction: {prediction} → **{prediction_label}**")
