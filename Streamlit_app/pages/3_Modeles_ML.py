import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.ml_models import train_ridge, train_random_forest, train_ridge_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🤖 Machine Learning – Analyse des émissions et croissance")

# Charger les données
df = load_data()

# Définir les colonnes disponibles
# Cibles : PIB + CO2
# Créer les colonnes
df['gdp_per_capita'] = df['gdp'] / df['population']
df['co2_per_capita'] = df['co2'] / df['population']

# Définir les cibles par nom de colonne
target_vars = ['gdp', 'gdp_per_capita', 'co2', 'co2_per_capita']


# Supprimer colonnes cibles, pays, année
feature_vars = [c for c in df.columns if c not in target_vars + ['country', 'year']]

# Variables explicatives par défaut
#default_features = [
   # "cement_co2",
   # "cumulative_cement_co2",
   # "oil_co2",
   # "co2_growth_prct",
   # "co2_including_luc_growth_abs", -0,2
    #"coal_co2",
   # "co2_including_luc_per_unit_energy",
   # "methane",
    #"co2_growth_abs",
   # "nitrous_oxide",
   # "co2_including_luc",
   # "co2_including_luc_per_gdp",
    #"co2_including_luc_growth_prct",
    # "energy_per_gdp"

    # Pour la belegique, on test d'autres variables
default_features = [
    "cement_co2",
    "oil_co2",
    "coal_co2",
    "methane",
    "nitrous_oxide",
    
    # Intensité carbone / énergie
   # "co2_including_luc_per_unit_energy",  # gCO2 par unité d'énergie
   # "energy_per_capita",                  # énergie par habitant
    #"energy_per_gdp",                     # énergie par unité de PIB
    
    # Variation et tendance des émissions
    #"co2_growth_abs",
    #"co2_growth_prct",
    #"co2_including_luc_growth_abs",
    #"co2_including_luc_growth_prct",
    
    # Autres indicateurs économiques propres
   # "gdp_per_capita",                     # pour capturer effet taille/population
]

# Sélection de la target et des features
target = st.selectbox("Choisissez la variable cible (target)", target_vars)
features = st.multiselect(
    "Sélectionnez les variables explicatives",
    feature_vars,
    default=[f for f in default_features if f in feature_vars]
)

# Sélection de la zone
pays_disponibles = sorted(df['country'].dropna().unique())
zones = ["Monde"] + pays_disponibles
zone = st.selectbox("Choisissez la zone géographique (Personnalisé)", zones, key="zone_custom")

# Choix du modèle ML
model_choice = st.radio(
    "Choisissez le modèle de Machine Learning",
    ["Ridge Regression (train only)", "Ridge Regression (train/test)", "Random Forest"]
)

# Tabs pour organisation
ml_tabs = st.tabs(["📚 Entraînement", "📊 Graphiques", "📈 Comparaison"])

# Stockage des résultats
if "ml_results" not in st.session_state:
    st.session_state.ml_results = {}

# --- Onglet Entraînement ---
with ml_tabs[0]:
    st.subheader("Entraînement du modèle")
    if st.button("Lancer l'entraînement"):
        if not features:
            st.error("Veuillez sélectionner au moins une variable explicative.")
        else:
            # Filtrer le dataframe selon la zone sélectionnée
            df_zone = df.copy() if zone == "Monde" else df[df["country"] == zone].copy()

            if model_choice == "Ridge Regression (train only)":
                results = train_ridge(df_zone, target, features)
            elif model_choice == "Ridge Regression (train/test)":
                results = train_ridge_split(df_zone, target, features)
            elif model_choice == "Random Forest":
                results = train_random_forest(df_zone, target, features)

            model_key = f"{model_choice} - {target} - {zone}"
            st.session_state.ml_results[model_key] = results
            st.success(f"Modèle {model_choice} entraîné avec succès pour la zone {zone} !")

# --- Onglet Graphiques ---
with ml_tabs[1]:
    st.subheader("Visualisation des résultats")
    if st.session_state.ml_results:
        model_selected = st.selectbox("Choisissez un modèle entraîné", list(st.session_state.ml_results.keys()))
        res = st.session_state.ml_results[model_selected]

        # Pour Ridge train/test, on utilise y_test et y_pred_test
        if "y_test" in res and "y_pred_test" in res:
            from utils.ml_models import plot_real_vs_pred, plot_feature_importance
            y = res["y_test"]
            y_pred = res["y_pred_test"]

            # Graph valeurs réelles vs prédictions
            plot_real_vs_pred(y, y_pred, f"{model_selected} : Valeurs réelles vs Prédictions")

            # Cas Ridge (coefficients dispo)
            if "coefficients" in res:
                coeffs = pd.Series(res["coefficients"]).sort_values(ascending=False)
                plot_feature_importance(coeffs, f"{model_selected} : Coefficients")

                # ✅ Tableau des coefficients
                st.markdown("### 📋 Coefficients des variables")
                st.dataframe(coeffs.to_frame("Coefficient"))

            # Cas Random Forest (feature importances dispo)
            elif "feature_importances" in res:
                importances = pd.Series(res["feature_importances"]).sort_values(ascending=False)
                plot_feature_importance(importances, f"{model_selected} : Importance des variables")

                # ✅ Tableau des importances
                st.markdown("### 📋 Importance des variables")
                st.dataframe(importances.to_frame("Importance"))

        else:
            st.warning("Pas de données de prédiction pour ce modèle. Assurez-vous d'utiliser un modèle train/test.")

        # Arbre Random Forest
        if "model" in res and model_selected.startswith("Random Forest"):
            rf_model = res["model"]
            st.markdown("### Visualisation d'un arbre de la forêt")

            n_trees = len(rf_model.estimators_)
            tree_index = st.slider("Choisissez l'arbre à afficher", 0, n_trees - 1, 0)

            fig, ax = plt.subplots(figsize=(20, 12))
            plot_tree(
                rf_model.estimators_[tree_index],
                feature_names=res.get("features", None),
                filled=True,
                rounded=True,
                fontsize=10,
                ax=ax
            )
            st.pyplot(fig)

            depths = [estimator.tree_.max_depth for estimator in rf_model.estimators_]
            fig, ax = plt.subplots()
            sns.histplot(depths, bins=range(max(depths) + 2), ax=ax)
            ax.set_title("Distribution des profondeurs des arbres dans la forêt")
            ax.set_xlabel("Profondeur de l'arbre")
            ax.set_ylabel("Nombre d'arbres")
            st.pyplot(fig)
    else:
        st.info("Entraînez d'abord un modèle pour voir les graphiques.")

# --- Onglet Comparaison ---
with ml_tabs[2]:
    st.subheader("Comparaison des modèles")

    if not st.session_state.ml_results:
        st.info("Entraînez d'abord au moins un modèle pour comparer.")
    else:
        modèles_dispo = list(st.session_state.ml_results.keys())
        modèles_choisis = st.multiselect(
            "Sélectionnez les modèles à comparer", 
            modèles_dispo, 
            default=modèles_dispo
        )

        if not modèles_choisis:
            st.warning("Sélectionnez au moins un modèle pour comparer.")
        else:
            import numpy as np

            # ---------------- Résumé général (R², RMSE) ----------------
            résumé = []
            for m in modèles_choisis:
                res = st.session_state.ml_results[m]
                résumé.append({
                    "Modèle": m,
                    "R² train": res.get("r2_train", res.get("r2", np.nan)),
                    "R² test": res.get("r2_test", np.nan),
                    "RMSE train": res.get("rmse_train", res.get("rmse", np.nan)),
                    "RMSE test": res.get("rmse_test", np.nan),
                })

            df_resume = pd.DataFrame(résumé).set_index("Modèle")

            # Tableau résumé
            st.dataframe(df_resume.style.format({
                "R² train": "{:.3f}",
                "R² test": "{:.3f}",
                "RMSE train": "{:.3e}",
                "RMSE test": "{:.3e}"
            }))

            # ---------------- Graphiques des métriques ----------------
            numeric_cols = df_resume.select_dtypes(include="number").columns
            if not numeric_cols.empty:
                fig, ax = plt.subplots(figsize=(12,6))
                df_resume[numeric_cols].plot(kind="bar", ax=ax)
                plt.ylabel("Score / Erreur")
                plt.title("Comparaison des métriques train/test")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

            # ---------------- Coefficients des modèles (Ridge uniquement) ----------------
            coeffs_list = []
            for m in modèles_choisis:
                res = st.session_state.ml_results[m]
                if "coefficients" in res:  # dispo seulement pour Ridge
                    coef_series = pd.Series(res["coefficients"], name=m)
                    coeffs_list.append(coef_series)

            if coeffs_list:
                df_coeffs = pd.concat(coeffs_list, axis=1).fillna(0)

                st.subheader("📋 Coefficients normalisés des variables (Ridge)")
                st.dataframe(df_coeffs.style.format("{:.3e}"))

                st.subheader("📊 Comparaison visuelle des coefficients normalisés")
                fig, ax = plt.subplots(figsize=(12,6))
                df_coeffs.plot(kind="bar", ax=ax)
                plt.title("Comparaison des coefficients Ridge par variable (normalisés)")
                plt.ylabel("Coefficient normalisé")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
