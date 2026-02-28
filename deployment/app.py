import os

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Predictor", layout="wide")
st.title("Wellness Tourism Package Prediction")

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "nalamrc/tourism-wellness-model")

model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_model.joblib", repo_type="model")
model = joblib.load(model_path)

# Collect input fields and convert to dataframe.
input_payload = {
    "Age": st.number_input("Age", min_value=18, max_value=90, value=35),
    "TypeofContact": st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"]),
    "CityTier": st.selectbox("CityTier", [1, 2, 3]),
    "DurationOfPitch": st.number_input("DurationOfPitch", min_value=1, max_value=120, value=15),
    "Occupation": st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]),
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "NumberOfPersonVisiting": st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=2),
    "NumberOfFollowups": st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=3),
    "ProductPitched": st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]),
    "PreferredPropertyStar": st.number_input("PreferredPropertyStar", min_value=1, max_value=5, value=3),
    "MaritalStatus": st.selectbox("MaritalStatus", ["Single", "Married", "Divorced", "Unmarried"]),
    "NumberOfTrips": st.number_input("NumberOfTrips", min_value=0, max_value=20, value=2),
    "Passport": st.selectbox("Passport", [0, 1]),
    "PitchSatisfactionScore": st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=3),
    "OwnCar": st.selectbox("OwnCar", [0, 1]),
    "NumberOfChildrenVisiting": st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=5, value=0),
    "Designation": st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]),
    "MonthlyIncome": st.number_input("MonthlyIncome", min_value=1000, max_value=300000, value=35000),
}

input_df = pd.DataFrame([input_payload])
st.dataframe(input_df)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else float(prediction)
    st.write({"will_purchase": int(prediction), "purchase_probability": float(probability)})
