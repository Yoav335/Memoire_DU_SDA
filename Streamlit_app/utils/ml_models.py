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


def train_ridge(df, target, features, alpha=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np

    df_model = df[features + [target]].dropna()

    X = df_model[features].values
    y = df_model[target].values

    model = Ridge(alpha=alpha)
    model.fit(X, y)
    y_pred = model.predict(X)

    results = {
        "r2": r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "coefficients": pd.Series(model.coef_, index=features),
        "fittedvalues": y_pred,
        "resid": y - y_pred,
        "target": y,
    }
    return results

def train_random_forest(df, target, features, n_estimators=100, test_size=0.2, random_state=42):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd

    # Préparer les données et retirer les lignes avec NaN dans features ou target
    df_model = df[features + [target]].dropna()

    if df_model.shape[0] < 10:
        raise ValueError("Pas assez de données pour entraîner le modèle.")

    X = df_model[features].values
    y = df_model[target].values

    # Séparer train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Initialiser et entraîner le modèle
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Prédictions sur test set
    y_pred = model.predict(X_test)

    # Calcul des métriques sur test set
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Résultats dans un dict
    results = {
        "model": model,  # le modèle entraîné complet pour visualisation
        "features": features,
        "r2": r2,
        "rmse": rmse,
        "feature_importances": pd.Series(model.feature_importances_, index=features).sort_values(ascending=False),
        "fittedvalues": y_pred,
        "resid": y_test - y_pred,
        "target": y_test,
    }
    return results
