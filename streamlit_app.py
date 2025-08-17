import streamlit as st
import joblib
import pandas as pd
import json
from datetime import datetime
import os

# Load model
MODEL_PATH = "xgb_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Place xgb_model.pkl in the working directory.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Load zip multipliers
ZIP_MULTIPLIERS_PATH = "zip3_multipliers_corrected.json"
if not os.path.exists(ZIP_MULTIPLIERS_PATH):
    st.error(f"Zip multipliers file not found at {ZIP_MULTIPLIERS_PATH}.")
    st.stop()

with open(ZIP_MULTIPLIERS_PATH, "r") as f:
    zip3_multipliers = json.load(f)

# Streamlit app UI
st.title("House Price Prediction")

beds = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=None)
baths = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=None)
sqft = st.number_input("House Size (sqft)", min_value=100, max_value=10000, value=None)
acre_lot = st.number_input("Lot Size (acre)", min_value=0.0, max_value=10.0, value=None)
zip_code = st.text_input("ZIP Code", value=None)
date_input = st.date_input("Date (approximate listing date)")

if st.button("Predict Price"):
    try:
        year = date_input.year
        month = date_input.month

        # Build DataFrame
        input_df = pd.DataFrame(
            [[beds, baths, acre_lot, zip_code, sqft, year, month]],
            columns=['bed', 'bath', 'acre_lot', 'zip_code', 'house_size', 'year', 'month']
        )
        input_df['zip_code'] = input_df['zip_code'].astype('category')

        # Predict
        price_val = float(model.predict(input_df)[0])

        # Inflation adjustment
        today = datetime.today()
        total_months = (today.year - year) * 12 + (today.month - month)
        inflation_rate = -0.002
        price_val *= (1 + inflation_rate * total_months)

        # Zip code adjustment
        zip_str = str(zip_code).zfill(5)
        zip3 = zip_str[:3]
        multiplier = zip3_multipliers.get(zip3, 1.0) * 1.7
        price_val *= multiplier

        # Lot size adjustment
        price_val *= (1.2) ** acre_lot

        st.success(f"Predicted House Price: ${price_val:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
