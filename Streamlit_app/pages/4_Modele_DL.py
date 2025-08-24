import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.ml_causal import default_features
from utils.xgboost_model import (
    train_xgboost, plot_real_vs_pred, plot_feature_importance, plot_residuals,
    plot_pred_vs_true_hist, plot_partial_dependence, plot_shap_summary,
    plot_predictions_vs_feature, plot_residuals_vs_feature, plot_pred_distribution_by_feature_bin,    
    plot_correlation_matrix, plot_prediction_histogram,
    plot_learning_curve_interactive, plot_sorted_predictions, plot_shap_waterfall,
)

st.title("🚀 Modélisation avec XGBoost")
st.markdown("""
Cette interface permet d'entraîner, d'évaluer et d'expliquer un modèle **XGBoost** 
sur les variables économiques et environnementales sélectionnées.
""")

# =====================
# CHARGEMENT DONNÉES
# =====================
df = load_data()
target_vars = ['gdp', 'gdp_per_capita', 'co2', 'co2_per_capita']

# Par défaut, utiliser les mêmes features que les autres modèles
default_feats = [f for f in default_features if f in df.columns]

target = st.selectbox("🎯 Variable cible (target)", target_vars, key="select_target")
features = st.multiselect(
    "🧩 Variables explicatives",
    options=default_feats,
    default=default_feats,
    key="select_features"
)

pays_disponibles = sorted(df['country'].dropna().unique())
zones = ["Monde"] + pays_disponibles
zone = st.selectbox("🌍 Zone géographique", zones, key="select_zone")

# =====================
# CRÉATION ONGLETS
# =====================
tabs = st.tabs([
    "📚 Entraînement",
    "📊 Visualisations",
    "🔎 Analyse avancée",
    "📉 Prédictions"
])

if "xgb_results" not in st.session_state:
    st.session_state.xgb_results = {}

# =====================
# ONGLET 1 : ENTRAÎNEMENT
# =====================
with tabs[0]:
    st.header("📚 Entraînement du modèle")

    if st.button("⚡ Lancer l'entraînement XGBoost", key="btn_train"):
        if not features:
            st.error("❌ Sélectionnez au moins une variable explicative.")
        else:
            df_zone = df if zone == "Monde" else df[df["country"] == zone]
            results = train_xgboost(df_zone, target, features)
            model_key = f"XGBoost - {target} - {zone}"
            st.session_state.xgb_results[model_key] = results

            st.success("✅ Modèle entraîné avec succès !")
            st.metric("R²", f"{results['r2']:.3f}")
            st.metric("RMSE", f"{results['rmse']:.3f}")
            st.metric("MAE", f"{results['mae']:.3f}")
            st.metric("MAPE (%)", f"{results['mape']:.2f}")


# =====================
# ONGLET 2 : VISUALISATIONS
# =====================
with tabs[1]:
    st.header("📊 Visualisations du modèle")

    if st.session_state.xgb_results:
        model_selected = st.selectbox("📌 Sélectionnez un modèle", list(st.session_state.xgb_results.keys()), key="select_model_visualisations")
        res = st.session_state.xgb_results[model_selected]

        st.subheader("📍 Comparaison Réel vs Prédit")
        plot_real_vs_pred(res["y_test"], res["y_pred"])
        st.caption("➡ Permet de vérifier si les prédictions suivent la tendance des valeurs réelles.")

        st.subheader("⭐ Importance des variables")
        plot_feature_importance(res["feature_importances"], res["features"])
        st.caption("➡ Met en évidence les variables qui influencent le plus le modèle.")

        st.subheader("📈 Résidus")
        plot_residuals(res["y_test"], res["y_pred"])
        st.caption("➡ Vérifie si les erreurs sont centrées autour de zéro.")

        st.subheader("📊 Histogrammes des valeurs")
        plot_pred_vs_true_hist(res["y_test"], res["y_pred"])

        st.subheader("📉 Matrice de corrélation")
        plot_correlation_matrix(res["X_test"])

        feature_for_resid = st.selectbox("Choisissez une feature pour analyser les résidus", res["features"])
        plot_residuals_vs_feature(res["X_test"], res["y_test"], res["y_pred"], feature_for_resid)

        st.subheader("📊 Distribution des prédictions")
        plot_prediction_histogram(res["y_pred"])

        st.subheader("📈 Courbe d'apprentissage")
        plot_learning_curve_interactive(res["model"], res["X_test"], res["y_test"])

        st.subheader("📏 Comparaison des valeurs triées")
        plot_sorted_predictions(res["y_test"], res["y_pred"])

    else:
        st.info("ℹ️ Entraînez d'abord un modèle.")

# =====================
# ONGLET 3 : ANALYSE AVANCÉE
# =====================
with tabs[2]:
    st.header("🔎 Analyse avancée")

    if st.session_state.xgb_results:
        model_selected = st.selectbox("📌 Modèle", list(st.session_state.xgb_results.keys()), key="select_model_analyse")
        res = st.session_state.xgb_results[model_selected]

        st.subheader("🧩 Partial Dependence Plot")
        top_feature = res["features"][res["feature_importances"].argmax()]
        plot_partial_dependence(res["model"], res["X_test"], top_feature, res["features"])

        st.subheader("🌟 SHAP Summary Plot")
        if st.button("Afficher le SHAP summary", key="btn_shap"):
            plot_shap_summary(res["model"], res["X_test"])

        st.subheader("💡 SHAP Waterfall (explication locale)")
        index_obs = st.slider("Observation à expliquer", 0, len(res["X_test"]) - 1, 0)
        if st.button("Afficher le SHAP waterfall", key="btn_shap_waterfall"):
            plot_shap_waterfall(res["model"], res["X_test"], index=index_obs)



    else:
        st.info("ℹ️ Entraînez d'abord un modèle.")

# =====================
# ONGLET 4 : PRÉDICTIONS
# =====================
with tabs[3]:
    st.header("📉 Tableau des prédictions")

    if st.session_state.xgb_results:
        model_selected = st.selectbox("📌 Modèle", list(st.session_state.xgb_results.keys()), key="select_model_predictions")
        res = st.session_state.xgb_results[model_selected]

        df_pred = pd.DataFrame({
            "Valeurs réelles": res["y_test"],
            "Prédictions": res["y_pred"]
        })
        st.dataframe(df_pred)

        csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Télécharger les prédictions CSV", data=csv, file_name="predictions.csv")

        feature_for_pred = st.selectbox("Choisissez une feature pour explorer les prédictions", res["features"], key="select_feature_for_pred")

        st.subheader("📍 Prédictions vs Feature")
        plot_predictions_vs_feature(res["X_test"], res["y_test"], res["y_pred"], feature_for_pred)

        st.subheader("📍 Résidus vs Feature")
        plot_residuals_vs_feature(res["X_test"], res["y_test"], res["y_pred"], feature_for_pred)

        st.subheader("📊 Distribution des prédictions par bins")
        plot_pred_distribution_by_feature_bin(res["X_test"], res["y_pred"], feature_for_pred)

    else:
        st.info("ℹ️ Entraînez d'abord un modèle.")
