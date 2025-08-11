import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.ml_models import train_ridge, train_random_forest
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🤖 Machine Learning – Analyse des émissions et croissance")

# Charger les données
df = load_data()

# Définir les colonnes disponibles
target_vars = ['gdp', 'gdp_per_capita', 'co2', 'co2_per_capita']
feature_vars = [c for c in df.columns if c not in target_vars + ['country', 'year']]

# Sélection de la target et des features
target = st.selectbox("Choisissez la variable cible (target)", target_vars)
features = st.multiselect("Sélectionnez les variables explicatives", feature_vars, default=feature_vars[:3])

pays_disponibles = sorted(df['country'].dropna().unique())
zones = ["Monde"] + pays_disponibles

zone = st.selectbox("Choisissez la zone géographique (Personnalisé)", zones, key="zone_custom")

# Choix du modèle ML
model_choice = st.radio("Choisissez le modèle de Machine Learning", ["Ridge Regression", "Random Forest"])

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
            if zone == "Monde":
                df_zone = df.copy()
            else:
                df_zone = df[df["country"] == zone].copy()

            if model_choice == "Ridge Regression":
                results = train_ridge(df_zone, target, features)
            elif model_choice == "Random Forest":
                results = train_random_forest(df_zone, target, features)

            model_key = f"{model_choice} - {target} - {zone}"
            st.session_state.ml_results[model_key] = results
            st.success(f"Modèle {model_choice} entraîné avec succès pour la zone {zone} !")

# --- Onglet Graphiques ---


# --- Onglet Graphiques ---
with ml_tabs[1]:
    st.subheader("Visualisation des résultats")
    if st.session_state.ml_results:
        model_selected = st.selectbox("Choisissez un modèle entraîné", list(st.session_state.ml_results.keys()))
        res = st.session_state.ml_results[model_selected]

        # Affichage valeurs réelles vs prédictions (comme avant)
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

        # --- Affichage arbre pour Random Forest ---
        if "model" in res and model_selected.startswith("Random Forest"):
            rf_model = res["model"]  # modèle RandomForestRegressor entraîné
            st.markdown("### Visualisation d'un arbre de la forêt")

            # Choix de l'arbre à afficher
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
            sns.histplot(depths, bins=range(max(depths)+2), ax=ax)
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
            résumé = []
            for m in modèles_choisis:
                res = st.session_state.ml_results[m]
                résumé.append({
                    "Modèle": m,
                    "R²": res.get("r2"),
                    "RMSE": res.get("rmse")
                })
            df_resume = pd.DataFrame(résumé).set_index("Modèle")
            st.dataframe(df_resume.style.format({"R²": "{:.3f}", "RMSE": "{:.3f}"}))

            # Graphique comparatif
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            df_resume.plot(kind="bar", ax=ax)
            plt.ylabel("Score / Erreur")
            st.pyplot(fig)
        else:
            st.warning("Sélectionnez au moins un modèle pour comparer.")
    else:
        st.info("Entraînez d'abord au moins un modèle pour comparer.")
