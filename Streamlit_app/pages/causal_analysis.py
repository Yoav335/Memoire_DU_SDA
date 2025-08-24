import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_data
from linearmodels.panel import PanelOLS
import statsmodels.api as sm


from utils.data_loader import load_data
from utils.ml_causal import default_features

st.set_page_config(layout="wide")
st.title("🧩 Analyse Causale du PIB")

# --- Charger les données ---
df = load_data()
df = df.set_index(['country', 'year'])  # Panel index requis pour PanelOLS

# --- Sélection target et features ---
target_vars = ['gdp', 'gdp_per_capita', 'co2', 'co2_per_capita']
target = st.selectbox("Variable cible (target)", target_vars)

features = st.multiselect(
    "Variables explicatives (features)",
    options=[f for f in default_features if f in df.columns],
    default=[f for f in default_features if f in df.columns]
)

if not features:
    st.warning("Sélectionnez au moins une variable explicative.")
    st.stop()

# --- PanelOLS global ---
st.subheader("1️⃣ Effets fixes globaux (PanelOLS)")
try:
    exog = sm.add_constant(df[features])
    mod = PanelOLS(df[target], exog, entity_effects=True, time_effects=True)
    res = mod.fit()
    st.text(res.summary)

    st.subheader("Coefficients globaux")
    coeffs = res.params[1:]  # retirer la constante
    st.dataframe(coeffs.to_frame(name="Coefficient"))

    # Graphique coefficients
    plt.figure(figsize=(10,6))
    sns.barplot(x=coeffs.values, y=coeffs.index, palette="viridis")
    plt.title("Coefficients globaux PanelOLS")
    plt.xlabel("Coefficient")
    plt.ylabel("Variables")
    st.pyplot(plt)
    plt.clf()
except Exception as e:
    st.error(f"Erreur PanelOLS : {e}")

# --- Coefficients par année ---
st.subheader("2️⃣ Évolution des coefficients par année")
coeffs_year_list = []
years = df.index.get_level_values('year').unique()

for y in years:
    df_y = df.xs(y, level='year')
    try:
        exog_y = sm.add_constant(df_y[features])
        mod_y = PanelOLS(df_y[target], exog_y)
        res_y = mod_y.fit()
        coeffs_year_list.append(res_y.params[1:].rename(y))
    except:
        continue

if coeffs_year_list:
    df_coeffs_year = pd.concat(coeffs_year_list, axis=1)
    st.dataframe(df_coeffs_year)

    # Heatmap
    plt.figure(figsize=(12,8))
    sns.heatmap(df_coeffs_year, annot=True, fmt=".2e", cmap="coolwarm", center=0)
    plt.title("Coefficients par variable et année")
    plt.xlabel("Année")
    plt.ylabel("Variable")
    st.pyplot(plt)
    plt.clf()
else:
    st.info("Impossible de calculer les coefficients par année : pas assez de données ou erreur OLS.")

# --- Evolution d’une variable ---
st.subheader("3️⃣ Visualisation interactive d’une variable")
var_selected = st.selectbox("Choisissez la variable à visualiser", features)
if 'df_coeffs_year' in locals():
    plt.figure(figsize=(12,6))
    plt.plot(df_coeffs_year.loc[var_selected], marker='o')
    plt.title(f"Évolution du coefficient de {var_selected} sur {target}")
    plt.xlabel("Année")
    plt.ylabel("Coefficient")
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()
