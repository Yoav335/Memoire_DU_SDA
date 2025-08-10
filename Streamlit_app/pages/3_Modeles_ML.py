import streamlit as st
from utils.data_loader import load_data
from utils.ml_models import run_ridge, run_random_forest

st.title("🤖 Modèles Machine Learning")

df = load_data()

target = st.selectbox("Variable cible", df.columns.tolist())
features = st.multiselect("Variables explicatives", df.columns.tolist())

model_type = st.radio("Choisissez un modèle", ["Ridge", "Random Forest"])

if st.button("Entraîner le modèle"):
    if features and target:
        if model_type == "Ridge":
            run_ridge(df, target, features)
        else:
            run_random_forest(df, target, features)
    else:
        st.warning("Veuillez choisir des variables.")
