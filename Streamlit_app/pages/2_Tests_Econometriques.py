import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.econometrics import run_ols_with_results  
import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.ml_models import train_ridge, train_random_forest, train_ridge_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Tests Économétriques – Impact des émissions sur la croissance par zone géographique")

# Chargement des données
df = load_data()

if "gdp_per_capita" not in df.columns:
    if {"gdp", "population"}.issubset(df.columns):
        df["gdp_per_capita"] = df["gdp"] / df["population"]
    else:
        st.error("Les colonnes 'gdp' et/ou 'population' manquent dans les données.")
        st.stop()

selected_vars = [
    'co2_per_capita', 'energy_per_capita', 'cement_co2_per_capita', 'gas_co2_per_capita',
    'oil_co2_per_capita', 'other_co2_per_capita', 'methane_per_capita', 'nitrous_oxide_per_capita',
    'ghg_per_capita', 'land_use_change_co2_per_capita', 'co2_per_gdp',
    'co2_including_luc_per_gdp', 'total_ghg', 'cement_co2', 'coal_co2', 'gas_co2', 'oil_co2',
    'other_industry_co2', 'primary_energy_consumption', 'temperature_change_from_ghg', 'trade_co2',
    'energy_per_gdp', 'co2_growth_abs', 'share_global_cumulative_co2','trade_co2_share'
]
target_vars = ['gdp', 'gdp_per_capita', 'co2', 'co2_per_capita']

default_models = {
    "Régression 1": {
        "target": "gdp",
        "features": [    "cement_co2",
    "oil_co2",
    "coal_co2",
    "methane",
    "nitrous_oxide",
    ],
        "description": "Modèle avec PIB total expliqué par CO2 total et consommation d'énergie par habitant."
    },
    "Régression 2": {
        "target": "gdp_per_capita",
        "features": ["total_ghg", "oil_co2", "gas_co2"],
        "description": "Modèle avec PIB par habitant expliqué par émissions totales de GES, CO2 pétrole et CO2 gaz."
    }
}

pays_disponibles = sorted(df['country'].dropna().unique())
zones = ["Monde"] + pays_disponibles

tabs = st.tabs(["Régression 1", "Régression 2", "Personnalisé", "Comparaison"])

# Initialisation session_state pour stocker résultats modèles
if "model_results" not in st.session_state:
    st.session_state.model_results = {}

def store_results(key, results):
    st.session_state.model_results[key] = results

def run_and_store(name, data, target, features, zone):
    results = run_ols_with_results(data, target, features)
    # results = dict avec les infos (r2, aic, bic, rmse, résidus, coefficients, ...)
    store_results(f"{name} ({zone})", results)

# Onglet Régression 1
with tabs[0]:
    st.header("Régression 1")
    zone = st.selectbox("Choisissez la zone géographique (Régression 1)", zones, key="zone_reg1")

    df_zone = df.copy() if zone == "Monde" else df[df["country"] == zone]
    if df_zone.empty:
        st.warning(f"Aucune donnée pour {zone}.")
        st.stop()

    model_info = default_models["Régression 1"]
    st.write(model_info["description"])
    st.write(f"Variable expliquée : **{model_info['target']}**")
    st.write(f"Variables explicatives : {model_info['features']}")

    if st.button(f"Lancer Régression 1 pour {zone}", key=f"run_reg1_{zone}"):
        try:
            run_and_store("Régression 1", df_zone, model_info["target"], model_info["features"], zone)
            st.success("Modèle lancé et résultats sauvegardés.")
        except Exception as e:
            st.error(f"Erreur : {e}")

# Onglet Régression 2
with tabs[1]:
    st.header("Régression 2")
    zone = st.selectbox("Choisissez la zone géographique (Régression 2)", zones, key="zone_reg2")

    df_zone = df.copy() if zone == "Monde" else df[df["country"] == zone]
    if df_zone.empty:
        st.warning(f"Aucune donnée pour {zone}.")
        st.stop()

    model_info = default_models["Régression 2"]
    st.write(model_info["description"])
    st.write(f"Variable expliquée : **{model_info['target']}**")
    st.write(f"Variables explicatives : {model_info['features']}")

    if st.button(f"Lancer Régression 2 pour {zone}", key=f"run_reg2_{zone}"):
        try:
            run_and_store("Régression 2", df_zone, model_info["target"], model_info["features"], zone)
            st.success("Modèle lancé et résultats sauvegardés.")
        except Exception as e:
            st.error(f"Erreur : {e}")

# Onglet Personnalisé
with tabs[2]:
    st.header("Modèle personnalisé")
    zone = st.selectbox("Choisissez la zone géographique (Personnalisé)", zones, key="zone_custom")

    df_zone = df.copy() if zone == "Monde" else df[df["country"] == zone]
    if df_zone.empty:
        st.warning(f"Aucune donnée pour {zone}.")
        st.stop()

    target = st.selectbox("Variable expliquée", target_vars, key="custom_target")
    available_vars = [v for v in selected_vars if v in df_zone.columns and v != target]
    variables = st.multiselect("Variables explicatives", options=available_vars, default=available_vars[:3], key="custom_vars")

    if st.button(f"Lancer modèle personnalisé pour {zone}", key="custom_run"):
        if not variables:
            st.warning("Sélectionnez au moins une variable.")
        else:
            try:
                run_and_store("Modèle personnalisé", df_zone, target, variables, zone)
                st.success("Modèle lancé et résultats sauvegardés.")
            except Exception as e:
                st.error(f"Erreur : {e}")
# --- Onglet Comparaison Econométrie ---
with tabs[3]:
    st.header("Comparaison des modèles économétriques")

    if not st.session_state.model_results:
        st.info("Lancez au moins un modèle dans les autres onglets pour comparer ici.")
        st.stop()

    modèles_lancés = list(st.session_state.model_results.keys())
    modèles_choisis = st.multiselect("Sélectionnez les modèles à comparer", modèles_lancés, default=modèles_lancés)

    if not modèles_choisis:
        st.warning("Sélectionnez au moins un modèle pour comparer.")
    else:
        import numpy as np

        # ---------------- Tableau résumé ----------------
        résumé = []
        for mod in modèles_choisis:
            res = st.session_state.model_results[mod]
            résumé.append({
                "Modèle": mod,
                "R²": res.get("r2", np.nan),
                "RMSE": res.get("rmse", np.nan),
            })
        df_résumé = pd.DataFrame(résumé).set_index("Modèle")
        st.dataframe(df_résumé.style.format({
            "R²": "{:.3f}",
            "RMSE": "{:.3f}"
        }))

        # ---------------- Graphiques métriques ----------------
        fig, ax = plt.subplots(figsize=(10,5))
        df_résumé.plot(kind="bar", ax=ax)
        plt.title("Comparaison des métriques")
        plt.ylabel("Score / Erreur")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        # ---------------- Graph valeurs réelles vs prédites ----------------
        plt.figure(figsize=(8,6))
        for mod in modèles_choisis:
            res = st.session_state.model_results[mod]
            actual = res.get("y_train", res.get("actual"))
            pred = res.get("y_pred_train", res.get("predictions"))
            if actual is not None and pred is not None:
                plt.scatter(actual, pred, alpha=0.6, label=mod)
        plt.plot([plt.xlim()[0], plt.xlim()[1]], [plt.xlim()[0], plt.xlim()[1]], 'k--', lw=2)
        plt.xlabel("Valeurs réelles")
        plt.ylabel("Valeurs prédites")
        plt.title("Valeurs réelles vs Prédictions")
        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()

        # ---------------- Graph résidus vs valeurs ajustées ----------------
        plt.figure(figsize=(8,6))
        for mod in modèles_choisis:
            res = st.session_state.model_results[mod]
            resid = res.get("resid_train", res.get("resid"))
            fitted = res.get("y_pred_train", res.get("fittedvalues"))
            if resid is not None and fitted is not None:
                plt.scatter(fitted, resid, alpha=0.6, label=mod)
        plt.axhline(0, color='black', lw=1, linestyle='--')
        plt.xlabel("Valeurs ajustées")
        plt.ylabel("Résidus")
        plt.title("Résidus vs Valeurs ajustées")
        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()

        # ---------------- Coefficients normalisés ----------------
        coeffs_list = []
        for mod in modèles_choisis:
            res = st.session_state.model_results[mod]
            if "coefficients" in res:
                coeff_series = pd.Series(res["coefficients"], name=mod)
                coeffs_list.append(coeff_series)
        if coeffs_list:
            df_coeffs = pd.concat(coeffs_list, axis=1).fillna(0)
            st.subheader("📋 Coefficients normalisés")
            st.dataframe(df_coeffs.style.format("{:.3e}"))

            st.subheader("📊 Comparaison graphique des coefficients")
            fig, ax = plt.subplots(figsize=(12,6))
            df_coeffs.plot(kind="bar", ax=ax)
            plt.ylabel("Coefficient normalisé")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
