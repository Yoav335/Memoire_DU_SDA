import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd


def plot_real_vs_pred(y_true, y_pred, title="Valeurs réelles vs Prédictions"):
    """
    Affiche un scatter plot des valeurs réelles vs prédites avec la diagonale y=x.

    Args:
        y_true (array-like): valeurs réelles
        y_pred (array-like): valeurs prédites
        title (str): titre du graphique
    """
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    plt.title(title)
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.clf()


def plot_feature_importance(series_importance, title="Importance des variables"):
    """
    Affiche un barplot des importances ou coefficients de variables.

    Args:
        series_importance (pd.Series): index = noms variables, valeurs = importances ou coefficients
        title (str): titre du graphique
    """
    series_sorted = series_importance.sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x=series_sorted.values, y=series_sorted.index, palette="viridis")
    plt.title(title)
    plt.xlabel("Importance / Coefficient")
    plt.ylabel("Variables")
    st.pyplot(plt.gcf())
    plt.clf()

def train_ridge(df, target, features, alpha=60, normalize=True):
    df_model = df[features + [target]].dropna()

    X = df_model[features].values
    y = df_model[target].values.reshape(-1, 1)

    # --- Standardisation ---
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    if normalize:  # standardiser aussi y pour des coefficients comparables
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y).ravel()
    else:
        y_scaled = y.ravel()

    # --- Modèle ---
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y_scaled)
    y_pred_scaled = model.predict(X_scaled)

    # Repasser les prédictions dans l’échelle originale si y était normalisé
    if normalize:
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
        resid = y.ravel() - y_pred
    else:
        y_pred = y_pred_scaled
        resid = y.ravel() - y_pred

    # --- Coefficients standardisés ---
    if normalize:
        coeffs_norm = pd.Series(model.coef_, index=features)  # coefficients comparables entre features
    else:
        coeffs_norm = pd.Series(model.coef_ / scaler_X.scale_, index=features)

    results = {
        "r2_train": r2_score(y, y_pred),
        "rmse_train": np.sqrt(mean_squared_error(y, y_pred)),
        "coefficients": coeffs_norm,
        # ✅ clés compatibles avec l’onglet graphique
        "y_train": y.ravel(),
        "y_pred_train": y_pred,
        "resid_train": resid,
    }
    return results



def train_random_forest(df, target, features, n_estimators=100, test_size=0.2, random_state=42):
    """
    Entraîne un RandomForestRegressor avec standardisation des features et retourne 
    métriques, fitted/test values et "coefficients" normalisés pour comparaison.
    """
    # Préparer les données
    df_model = df[features + [target]].dropna()
    if df_model.shape[0] < 10:
        raise ValueError("Pas assez de données pour entraîner le modèle.")

    X = df_model[features].values
    y = df_model[target].values

    # Standardiser les features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    # Entraîner le modèle
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Métriques
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Normalisation des importances pour comparaison avec Ridge
    importances = pd.Series(model.feature_importances_, index=features)
    # On multiplie par l'écart-type pour revenir à l'échelle originale approximative
    coeffs_norm = importances * scaler.scale_

    results = {
        "model": model,
        "features": features,
        "r2": r2,
        "rmse": rmse,
        "coefficients": coeffs_norm,  # utilisable comme les coefficients Ridge
        "fittedvalues": y_pred,
        "resid": y_test - y_pred,
        "target": y_test,
    }
    return results

def train_ridge_split(df, target, features, alpha=60, test_size=0.2, random_state=42):
    df_model = df[features + [target]].dropna()
    X = df_model[features].values
    y = df_model[target].values

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Standardisation
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()

    # Ridge
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train_scaled)

    # Prédictions
    y_train_pred = scaler_y.inverse_transform(model.predict(X_train_scaled).reshape(-1,1)).ravel()
    y_test_pred = scaler_y.inverse_transform(model.predict(X_test_scaled).reshape(-1,1)).ravel()

    results = {
        "r2_train": r2_score(y_train, y_train_pred),
        "r2_test": r2_score(y_test, y_test_pred),
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "coefficients": pd.Series(model.coef_, index=features),
        # ✅ clés compatibles avec l’onglet graphique
        "y_train": y_train,
        "y_pred_train": y_train_pred,
        "y_test": y_test,
        "y_pred_test": y_test_pred,
        "resid_train": y_train - y_train_pred,
        "resid_test": y_test - y_test_pred,
    }

    return results
