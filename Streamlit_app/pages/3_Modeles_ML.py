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
target_vars = ['gdp', 'gdp_per_capita', 'co2', 'co2_per_capita']
feature_vars = [c for c in df.columns if c not in target_vars + ['country', 'year']]

default_features = [
    "cement_co2",
    "cumulative_cement_co2",
    "oil_co2",
    "co2_growth_prct",
    "co2_including_luc_growth_abs",
    "coal_co2",
    "co2_including_luc_per_unit_energy",
    "methane",
    "co2_growth_abs",
    "nitrous_oxide",
    "co2_including_luc",
    "co2_including_luc_per_gdp",
    "co2_including_luc_growth_prct",
    "energy_per_gdp"
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

        if "fittedvalues" in res and "resid" in res:
            from utils.ml_models import plot_real_vs_pred, plot_feature_importance
            y = res["fittedvalues"] + res["resid"]
            y_pred = res["fittedvalues"]

            plot_real_vs_pred(y, y_pred, f"{model_selected} : Valeurs réelles vs Prédictions")

            if "coefficients" in res:
                plot_feature_importance(res["coefficients"], f"{model_selected} : Coefficients")
            elif "feature_importances" in res:
                plot_feature_importance(res["feature_importances"], f"{model_selected} : Importance des variables")
        else:
            st.warning("Pas de données de prédiction pour ce modèle.")

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
    if st.session_state.ml_results:
        modèles_dispo = list(st.session_state.ml_results.keys())
        modèles_choisis = st.multiselect("Sélectionnez les modèles à comparer", modèles_dispo, default=modèles_dispo)

        if modèles_choisis:
            import numpy as np

            # Construire le tableau résumé
            résumé = []
            for m in modèles_choisis:
                res = st.session_state.ml_results[m]
                résumé.append({
                    "Modèle": m,
                    "R²": res.get("r2", np.nan),
                    "RMSE": res.get("rmse", np.nan),
                    "Loss": res.get("loss", np.nan)  # np.nan si pas dispo
                })

            df_resume = pd.DataFrame(résumé).set_index("Modèle")

            # Affichage du tableau avec format conditionnel
            st.dataframe(df_resume.style.format({
                "R²": "{:.3f}",
                "RMSE": "{:.3f}",
                "Loss": lambda x: "{:.3f}".format(x) if not pd.isna(x) else "-"
            }))

            # --- Graphique comparatif des métriques (R², RMSE, Loss) ---
            numeric_cols = df_resume.select_dtypes(include="number").columns
            if not numeric_cols.empty:
                fig, ax = plt.subplots()
                df_resume[numeric_cols].plot(kind="bar", ax=ax)
                plt.ylabel("Score / Erreur / Perte")
                plt.title("Comparaison des métriques")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            # --- Graphique dédié à la fonction de perte ---
            if "Loss" in df_resume.columns and df_resume["Loss"].notna().any():
                fig, ax = plt.subplots()
                df_resume["Loss"].dropna().plot(kind="bar", color="salmon", ax=ax)
                plt.ylabel("Loss")
                plt.title("Comparaison de la fonction de perte (Loss)")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("Sélectionnez au moins un modèle pour comparer.")
    else:
        st.info("Entraînez d'abord au moins un modèle pour comparer.")
