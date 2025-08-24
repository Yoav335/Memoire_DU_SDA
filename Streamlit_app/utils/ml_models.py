import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
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


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

def train_ridge(df, target, features, alpha=1.0, normalize=True):
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
        # si y pas normalisé : remettre à l’échelle originale
        coeffs_norm = pd.Series(model.coef_ / scaler_X.scale_, index=features)

    results = {
        "r2": r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "coefficients": coeffs_norm,
        "fittedvalues": y_pred,
        "resid": resid,
        "target": y.ravel(),
    }
    return results


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

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

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def train_ridge_split(df, target, features, alpha=1.0, test_size=0.2, random_state=42):
    """
    Entraîne une régression Ridge avec split train/test et calcule la perte.

    Args:
        df (pd.DataFrame): Données contenant features + target
        target (str): Nom de la variable cible
        features (list): Liste des variables explicatives
        alpha (float): Paramètre de régularisation Ridge
        test_size (float): Proportion des données pour le test
        random_state (int): Graine aléatoire

    Returns:
        dict: Résultats avec métriques, coefficients, fitted/test values et perte
    """
    df_model = df[features + [target]].dropna()
    X = df_model[features].values
    y = df_model[target].values

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = Ridge(alpha=alpha, random_state=random_state)
    model.fit(X_train, y_train)

    # Prédictions test
    y_pred = model.predict(X_test)

    # Calcul des métriques
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Fonction de perte Ridge sur le test
    mse_test = mean_squared_error(y_test, y_pred)
    ridge_penalty = alpha * np.sum(model.coef_ ** 2)
    loss = mse_test + ridge_penalty

    results = {
        "model": model,
        "features": features,
        "r2": r2,
        "rmse": rmse,
        "coefficients": pd.Series(model.coef_, index=features),
        "fittedvalues": y_pred,
        "resid": y_test - y_pred,
        "target": y_test,
        "loss": loss,
    }
    return results
