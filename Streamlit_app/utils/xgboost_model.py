import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import shap
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, learning_curve

# =====================
# 1. Entraînement XGBoost
# =====================
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

    # Métriques
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return {
        "model": model,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "X_test": X_test,
        "y_test": y_test.values,
        "y_pred": y_pred,
        "features": features,
        "feature_importances": model.feature_importances_
    }

# =====================
# 2. Diagnostics de performance
# =====================
def plot_real_vs_pred(y_true, y_pred, title="Réelles vs Prédictions"):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title(title)
    st.pyplot(plt.gcf())
    plt.clf()

def plot_real_vs_pred_sorted(y_true, y_pred, title="Valeurs réelles vs prédictions triées"):
    df_plot = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).sort_values("y_true")
    plt.figure(figsize=(10,5))
    plt.plot(df_plot["y_true"].values, label="Réelles", marker="o")
    plt.plot(df_plot["y_pred"].values, label="Prédictions", marker="x")
    plt.legend()
    plt.title(title)
    st.pyplot(plt.gcf())
    plt.clf()

def plot_pred_vs_true_hist(y_true, y_pred, title="Distribution des valeurs réelles vs prédites"):
    plt.figure(figsize=(8, 5))
    sns.histplot(y_true, bins=30, color='blue', label='Réel', alpha=0.6)
    sns.histplot(y_pred, bins=30, color='orange', label='Prédit', alpha=0.6)
    plt.legend()
    plt.title(title)
    st.pyplot(plt.gcf())
    plt.clf()


def plot_learning_curve_interactive(model, X, y):
    st.markdown("### Courbe d'apprentissage")
    
    # Slider pour ajuster la taille du test
    test_size = st.slider(
        "Proportion du jeu de test", min_value=0.1, max_value=0.5, value=0.2, step=0.05
    )

    # Calcul du learning curve
    from sklearn.model_selection import learning_curve
    
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, scoring="r2", 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        # Calcul des moyennes
        train_scores_mean = train_scores.mean(axis=1)
        test_scores_mean = test_scores.mean(axis=1)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(train_sizes, train_scores_mean, label="Train R²", marker='o')
        ax.plot(train_sizes, test_scores_mean, label="Test R²", marker='x')
        ax.set_xlabel("Taille du jeu d'entraînement")
        ax.set_ylabel("R²")
        ax.set_title("Courbe d'apprentissage XGBoost")
        ax.legend()
        st.pyplot(fig)

        st.markdown(
            "**Note :** Le R² du train est très proche de 1 car XGBoost est un modèle flexible qui s'ajuste très bien aux données d'entraînement. "
            "Pour détecter un surapprentissage, regardez plutôt le R² du test."
        )
    except Exception as e:
        st.error(f"Erreur lors du calcul de la courbe d'apprentissage : {e}")


def plot_sorted_predictions(y_true, y_pred, title="Comparaison valeurs triées"):

    sorted_idx = np.argsort(y_true)
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(len(y_true)), np.array(y_true)[sorted_idx], label="Réel", marker="o", alpha=0.7)
    plt.plot(np.arange(len(y_pred)), np.array(y_pred)[sorted_idx], label="Prédit", marker="x", alpha=0.7)
    plt.xlabel("Observations triées")
    plt.ylabel("Valeur cible")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

# =====================
# 3. Interprétabilité
# =====================
def plot_feature_importance(importances, features, title="Importance des variables"):
    importance_df = pd.DataFrame({
        "Variable": features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Variable")
    plt.title(title)
    st.pyplot(plt.gcf())
    plt.clf()

def plot_partial_dependence(model, X, feature_name, features, n_grid=30):
    X_sample = X.sample(n=2000, random_state=42) if len(X) > 2000 else X.copy()
    grid = np.linspace(X_sample[feature_name].min(), X_sample[feature_name].max(), n_grid)
    preds = []

    X_temp = X_sample.copy()
    for val in grid:
        X_temp[feature_name] = val
        pred = model.predict(X_temp[features])
        preds.append(np.mean(pred))

    plt.figure(figsize=(8, 5))
    plt.plot(grid, preds)
    plt.xlabel(feature_name)
    plt.ylabel("Prédiction moyenne")
    plt.title(f"Partial Dependence : {feature_name}")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_shap_summary(model, X, max_display=10):
    import shap
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.error(f"Erreur SHAP : {e}")

def plot_shap_local(model, X, index=0):
    import shap
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        shap.plots.waterfall(shap_values[index], max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.error(f"Erreur SHAP locale : {e}")

def plot_shap_waterfall(model, X, index=0):
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        st.write(f"Explication de l’observation n° {index}")

        # Crée une figure Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[index], max_display=10, show=False)
        
        # Récupère la figure globale générée par SHAP et l'affiche
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)  # ferme la figure pour libérer la mémoire
    except Exception as e:
        st.error(f"Erreur lors du SHAP waterfall plot : {e}")

# =====================
# 4. Résidus & distributions
# =====================
def plot_residuals(y_true, y_pred, title="Distribution des résidus"):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color='purple')
    plt.title(title)
    plt.xlabel("Résidus")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_residuals_vs_feature(X, y_true, y_pred, feature_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=X[feature_name], y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel(feature_name)
    plt.ylabel("Résidus")
    plt.title(f"Résidus vs {feature_name}")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_predictions_vs_feature(X_test, y_true, y_pred, feature_name):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X_test[feature_name], y=y_true, label="Valeurs réelles", alpha=0.6)
    sns.scatterplot(x=X_test[feature_name], y=y_pred, label="Prédictions", alpha=0.6)
    plt.xlabel(feature_name)
    plt.ylabel("Valeur cible")
    plt.title(f"Prédictions vs Réel selon '{feature_name}'")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_pred_distribution_by_feature_bin(X_test, y_pred, feature_name, bins=10):
    df = X_test.copy()
    df['prediction'] = y_pred
    df['bin'] = pd.qcut(df[feature_name], q=bins, duplicates='drop')

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='bin', y='prediction', data=df)
    plt.xticks(rotation=45)
    plt.xlabel(feature_name)
    plt.ylabel("Prédictions")
    plt.title(f"Distribution des prédictions selon bins de '{feature_name}'")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_prediction_histogram(y_pred):
    plt.figure(figsize=(8, 5))
    sns.histplot(y_pred, bins=30, color='green', kde=True)
    plt.title("Distribution des prédictions")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_correlation_matrix(X):
    corr = X.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Corrélation entre variables explicatives")
    st.pyplot(plt.gcf())
    plt.clf()

# =====================
# 5. Comparaison par groupe (ex: pays)
# =====================
def eval_by_group(X_test, y_true, y_pred, group_values, group_name="country"):
    df_eval = X_test.copy()
    df_eval["y_true"] = y_true
    df_eval["y_pred"] = y_pred
    df_eval[group_name] = group_values

    eval_df = df_eval.groupby(group_name).apply(
        lambda d: pd.Series({
            "R²": r2_score(d["y_true"], d["y_pred"]),
            "RMSE": np.sqrt(mean_squared_error(d["y_true"], d["y_pred"]))
        })
    )
    st.dataframe(eval_df.sort_values("R²", ascending=False))
    return eval_df

