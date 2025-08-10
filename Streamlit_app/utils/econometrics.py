import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st

def run_ols_with_results(data, target, features):
    """
    Effectue une régression OLS, affiche les résultats avec Streamlit,
    puis retourne un dictionnaire avec les indicateurs et résidus.

    Args:
        data (pd.DataFrame): dataframe avec les données
        target (str): nom de la variable expliquée
        features (list of str): liste des variables explicatives

    Returns:
        dict: résultats du modèle contenant notamment :
            - r2 : coefficient de détermination
            - aic : critère AIC
            - bic : critère BIC
            - rmse : erreur quadratique moyenne
            - coefficients : pandas.Series des coefficients
            - resid : résidus (np.array) ou None si indisponible
            - fittedvalues : valeurs ajustées (np.array)
            - predictions : valeurs prédites (np.array), identique à fittedvalues
            - actual : valeurs réelles (np.array)
    """

    df_model = data[features + [target]].dropna()

    X = df_model[features]
    X = sm.add_constant(X)
    y = df_model[target]
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    model = sm.OLS(y, X).fit()

    st.text(model.summary())

    # Graphique résidus vs valeurs ajustées
    plt.figure(figsize=(8,4))
    sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, line_kws={'color': 'red', 'lw': 1})
    plt.xlabel("Valeurs ajustées")
    plt.ylabel("Résidus")
    plt.title("Graphique des résidus")
    st.pyplot(plt.gcf())
    plt.clf()

    # Histogramme des résidus
    plt.figure(figsize=(8,4))
    sns.histplot(model.resid, kde=True)
    plt.xlabel("Résidus")
    plt.title("Distribution des résidus")
    st.pyplot(plt.gcf())
    plt.clf()

    rmse = np.sqrt(np.mean(model.resid**2))

    resid_values = model.resid.values if hasattr(model, 'resid') else None

    results = {
        "r2": model.rsquared,
        "aic": model.aic,
        "bic": model.bic,
        "rmse": rmse,
        "coefficients": model.params,
        "resid": resid_values,
        "fittedvalues": model.fittedvalues.values,
        "predictions": model.fittedvalues.values,
        "actual": y.values
    }

    return results
