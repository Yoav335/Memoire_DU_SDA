import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.model_selection import learning_curve

def train_xgboost(data, target, features, test_size=0.2, random_state=42):
    df_model = data[features + [target]].dropna()

    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "model": model,
        "r2": r2,
        "rmse": rmse,
        "X_test": X_test,
        "y_test": y_test.values,
        "y_pred": y_pred,
        "features": features,
        "feature_importances": model.feature_importances_
    }

def plot_real_vs_pred(y_true, y_pred, title="Réelles vs Prédictions"):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importances, features, title="Importance des variables"):
    importance_df = pd.DataFrame({
        "Variable": features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Variable")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred, title="Distribution des résidus"):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color='purple')
    plt.title(title)
    plt.xlabel("Résidus (réel - prédit)")
    plt.tight_layout()
    plt.show()

def plot_pred_vs_true_hist(y_true, y_pred, title="Distribution des valeurs réelles vs prédites"):
    plt.figure(figsize=(8, 5))
    sns.histplot(y_true, bins=30, color='blue', label='Réel', alpha=0.6)
    sns.histplot(y_pred, bins=30, color='orange', label='Prédit', alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Partial Dependence Plot simple pour une variable
def plot_partial_dependence(model, X, feature_name, features, n_grid=30):

    # Échantillonner les données pour accélérer
    X_sample = X.sample(n=2000, random_state=42) if len(X) > 2000 else X.copy()
    grid = np.linspace(X_sample[feature_name].min(), X_sample[feature_name].max(), n_grid)
    preds = []

    X_temp = X_sample.copy()

    for val in grid:
        X_temp[feature_name] = val
        pred = model.predict(X_temp[features])
        preds.append(np.mean(pred))

    preds = np.array(preds)

    plt.figure(figsize=(8, 5))
    plt.plot(grid, preds)
    plt.xlabel(feature_name)
    plt.ylabel("Prédiction moyenne")
    plt.title(f"Partial Dependence : {feature_name}")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()


def plot_shap_summary(model, X, max_display=10):
    import shap

    try:
        # Crée l'explainer SHAP (attention à la compatibilité selon modèle)
        explainer = shap.Explainer(model)
        
        # Calcul des valeurs SHAP
        shap_values = explainer(X)

        # Création d'une figure matplotlib
        plt.figure(figsize=(10, 6))
        
        # Trace le summary plot sans afficher directement (show=False)
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)

        # Affiche la figure dans Streamlit
        st.pyplot(plt.gcf())
        plt.clf()

    except Exception as e:
        st.error(f"Erreur lors du calcul SHAP : {e}")

def plot_predictions_vs_feature(X_test, y_true, y_pred, feature_name):
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=X_test[feature_name], y=y_true, label="Valeurs réelles", alpha=0.6)
    sns.scatterplot(x=X_test[feature_name], y=y_pred, label="Prédictions", alpha=0.6)
    plt.xlabel(feature_name)
    plt.ylabel("Valeur cible")
    plt.title(f"Prédictions vs Réel selon '{feature_name}'")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_residuals_vs_feature(X_test, y_true, y_pred, feature_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=X_test[feature_name], y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel(feature_name)
    plt.ylabel("Résidus (réel - prédit)")
    plt.title(f"Résidus vs {feature_name}")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_pred_distribution_by_feature_bin(X_test, y_pred, feature_name, bins=10):
    df = X_test.copy()
    df['prediction'] = y_pred
    df['bin'] = pd.qcut(df[feature_name], q=bins, duplicates='drop')

    plt.figure(figsize=(10,6))
    sns.boxplot(x='bin', y='prediction', data=df)
    plt.xticks(rotation=45)
    plt.xlabel(feature_name)
    plt.ylabel("Prédictions")
    plt.title(f"Distribution des prédictions selon bins de '{feature_name}'")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()


def plot_correlation_matrix(X):
    corr = X.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Corrélation entre variables explicatives")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_residuals_vs_feature(X, y_true, y_pred, feature_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=X[feature_name], y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel(feature_name)
    plt.ylabel("Résidus (réel - prédit)")
    plt.title(f"Résidus vs {feature_name}")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_prediction_histogram(y_pred):
    plt.figure(figsize=(8,5))
    sns.histplot(y_pred, bins=30, color='green', kde=True)
    plt.title("Distribution des prédictions")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()