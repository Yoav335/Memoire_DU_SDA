import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS
from utils.data_loader import load_data
import statsmodels.api as sm

st.set_page_config(layout="wide")
st.title("📊 Panel Regression – Effet des variables sur le GDP")

# Charger les données
df = load_data()
df_panel = df.set_index(["country", "year"])
target_vars = ['gdp', 'gdp_per_capita', 'co2', 'co2_per_capita']

# Sélection target et features
target = st.selectbox("Variable cible (target)", target_vars)
from utils.ml_causal import default_features
features = st.multiselect(
    "Variables explicatives",
    options=[f for f in default_features if f in df.columns],
    default=[f for f in default_features if f in df.columns]
)

if st.button("Lancer l'estimation PanelOLS"):
    y = df_panel[target]
    X = sm.add_constant(df_panel[features])

    # Modèle avec effets fixes
    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    results = model.fit(cov_type='robust')

    # Résumé général
    st.subheader("Résumé PanelOLS")
    st.text(results.summary)

    # Coefficients
    st.subheader("Coefficients estimés")
    coeffs_df = pd.DataFrame({'Variable': results.params.index, 'Coefficient': results.params.values})
    st.dataframe(coeffs_df)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x="Coefficient", y="Variable", data=coeffs_df, palette="viridis", ax=ax)
    ax.set_title("Coefficients des variables explicatives")
    st.pyplot(fig)

    # Récupérer les effets fixes séparés
    effects_df = results.estimated_effects.reset_index()
    
    if 'entity' in effects_df.columns:
        st.subheader("Effets fixes par pays (Entity Effects)")
        entity_effects_df = effects_df[['entity', 'effect']].dropna().rename(columns={'entity':'Pays','effect':'Effet'})
        st.dataframe(entity_effects_df)
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x="Effet", y="Pays", data=entity_effects_df, palette="coolwarm", ax=ax)
        ax.set_title("Effets fixes par pays")
        st.pyplot(fig)

    if 'time' in effects_df.columns:
        st.subheader("Effets fixes par année (Time Effects)")
        time_effects_df = effects_df[['time', 'effect']].dropna().rename(columns={'time':'Année','effect':'Effet'})
        st.dataframe(time_effects_df)
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x="Année", y="Effet", data=time_effects_df, palette="coolwarm", ax=ax)
        plt.xticks(rotation=45)
        ax.set_title("Effets fixes par année")
        st.pyplot(fig)
