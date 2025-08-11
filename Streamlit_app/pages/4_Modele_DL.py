import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.xgboost_model import (
    train_xgboost, plot_real_vs_pred, plot_feature_importance, plot_residuals,
    plot_pred_vs_true_hist, plot_partial_dependence, plot_shap_summary,
    plot_predictions_vs_feature, plot_residuals_vs_feature, plot_pred_distribution_by_feature_bin,    
    plot_correlation_matrix, plot_residuals_vs_feature, plot_prediction_histogram
)


st.title("🚀 XGBoost – Modélisation")

df = load_data()
target_vars = ['gdp', 'gdp_per_capita', 'co2', 'co2_per_capita']
feature_vars = [c for c in df.columns if c not in target_vars + ['country', 'year']]

target = st.selectbox("Variable cible (target)", target_vars, key="select_target")
features = st.multiselect("Variables explicatives", feature_vars, default=feature_vars[:3], key="select_features")

pays_disponibles = sorted(df['country'].dropna().unique())
zones = ["Monde"] + pays_disponibles
zone = st.selectbox("Zone géographique", zones, key="select_zone")

tabs = st.tabs(["📚 Entraînement", "📊 Graphiques", "📈 Analyse avancée", "📉 Prédictions"])

if "xgb_results" not in st.session_state:
    st.session_state.xgb_results = {}

with tabs[0]:
    st.subheader("Entraînement")
    if st.button("Lancer l'entraînement XGBoost", key="btn_train"):
        if not features:
            st.error("Sélectionnez au moins une variable explicative.")
        else:
            df_zone = df if zone == "Monde" else df[df["country"] == zone]
            results = train_xgboost(df_zone, target, features)
            model_key = f"XGBoost - {target} - {zone}"
            st.session_state.xgb_results[model_key] = results
            st.success(f"Modèle entraîné : R²={results['r2']:.3f}, RMSE={results['rmse']:.3f}")

with tabs[1]:
    st.subheader("Visualisations")
    if st.session_state.xgb_results:
        model_selected = st.selectbox(
            "Modèle", 
            list(st.session_state.xgb_results.keys()), 
            key="select_model_visualisations"
        )
        res = st.session_state.xgb_results[model_selected]

        plot_real_vs_pred(res["y_test"], res["y_pred"], f"{model_selected} : Réel vs Prédit")
        plot_feature_importance(res["feature_importances"], res["features"], f"{model_selected} : Importance des variables")
        plot_residuals(res["y_test"], res["y_pred"])
        plot_pred_vs_true_hist(res["y_test"], res["y_pred"])
        plot_correlation_matrix(res["X_test"])

        feature_for_resid = st.selectbox("Choisissez une feature pour visualiser les résidus", res["features"])
        plot_residuals_vs_feature(res["X_test"], res["y_test"], res["y_pred"], feature_for_resid)
        plot_prediction_histogram(res["y_pred"])

    else:
        st.info("Entraînez un modèle d'abord.")

with tabs[2]:
    st.subheader("Analyse avancée")
    if st.session_state.xgb_results:
        model_selected = st.selectbox(
            "Modèle pour analyse avancée", 
            list(st.session_state.xgb_results.keys()), 
            key="select_model_analyse"
        )
        res = st.session_state.xgb_results[model_selected]

        top_feature = res["features"][res["feature_importances"].argmax()]
        with st.spinner("Calcul du Partial Dependence Plot..."):
            plot_partial_dependence(res["model"], res["X_test"], top_feature, res["features"])

        if st.button("Afficher SHAP summary plot", key="btn_shap"):
            st.write("Calcul en cours, patience...")
            plot_shap_summary(res["model"], res["X_test"])
    else:
        st.info("Entraînez un modèle d'abord.")

with tabs[3]:
    st.subheader("Tableau des prédictions et graphiques")
    if st.session_state.xgb_results:
        model_selected = st.selectbox(
            "Modèle pour prédictions", 
            list(st.session_state.xgb_results.keys()), 
            key="select_model_predictions"
        )
        res = st.session_state.xgb_results[model_selected]

        df_pred = pd.DataFrame({
            "Valeurs réelles": res["y_test"],
            "Prédictions": res["y_pred"]
        })
        st.dataframe(df_pred)

        csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger les prédictions CSV", data=csv, file_name="predictions.csv")

        feature_for_pred = st.selectbox(
            "Choisissez une feature pour visualiser les prédictions", 
            res["features"], 
            key="select_feature_for_pred"
        )

        st.markdown("### Prédictions vs Feature")
        plot_predictions_vs_feature(res["X_test"], res["y_test"], res["y_pred"], feature_for_pred)

        st.markdown("### Résidus vs Feature")
        plot_residuals_vs_feature(res["X_test"], res["y_test"], res["y_pred"], feature_for_pred)

        st.markdown("### Distribution des prédictions par bins de la feature")
        plot_pred_distribution_by_feature_bin(res["X_test"], res["y_pred"], feature_for_pred)
    else:
        st.info("Entraînez un modèle d'abord.")
