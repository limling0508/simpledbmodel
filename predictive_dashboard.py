import os
import streamlit as st
import pandas as pd
import joblib

st.title("Revenue Prediction Dashboard")

# --- Paths (always relative to this script) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "sales.csv")
MODEL_PATH = os.path.join(BASE_DIR, "revenue_model.pkl")

# --- Check files exist (prevents silent blank page) ---
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Make sure sales.csv is in the same folder as this app.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f" Model not found: {MODEL_PATH}")
    st.info("Run: python train_model.py  to create revenue_model.pkl")
    st.stop()

# --- Load dataset (proof of connection) ---
df = pd.read_csv(DATA_PATH)
st.success("Dataset loaded successfully!")
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
st.dataframe(df.head(5))

# --- Load trained model ---
model = joblib.load(MODEL_PATH)

# --- Use dataset values for dropdown options (strong proof) ---
regions = sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else ["North", "South", "East", "West"]
products = sorted(df["product"].dropna().unique().tolist()) if "product" in df.columns else ["Widget", "Gadget", "Tool", "Device"]

# --- Inputs ---
units = st.slider("Units Sold", 10, 100)
region = st.selectbox("Region", regions)
product = st.selectbox("Product", products)

# --- Prepare input data ---
input_df = pd.DataFrame({
    "units_sold": [units],
    "region": [region],
    "product": [product]
})

# --- Predict revenue ---
predicted_revenue = model.predict(input_df)[0]
st.metric("Predicted Revenue", f"${predicted_revenue:,.2f}")

