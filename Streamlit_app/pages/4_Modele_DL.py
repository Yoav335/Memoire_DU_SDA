import streamlit as st
from utils.data_loader import load_data
from utils.dl_models import run_xgboost

st.title("🧠 Modèle XGBoost")

df = load_data()

target = st.selectbox("Variable cible", df.columns.tolist())
features = st.multiselect("Variables explicatives", df.columns.tolist())

if st.button("Lancer XGBoost"):
    if features and target:
        run_xgboost(df, target, features)
    else:
        st.warning("Veuillez choisir des variables.")
