import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression

# Variables par défaut
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

def list_confounders(df, target, features):
    """
    Identifie les variables confondantes potentielles pour la variable cible.

    Seules les colonnes numériques sont considérées.
    """
    df_model = df[[target] + features].dropna()
    numeric_cols = df_model.select_dtypes(include="number").columns.tolist()
    if target not in numeric_cols:
        raise ValueError(f"La target '{target}' doit être numérique.")
    
    features_numeric = [f for f in features if f in numeric_cols]
    if not features_numeric:
        return []

    # On utilise le test de f_regression pour détecter les variables corrélées au target
    X = df_model[features_numeric].values
    y = df_model[target].values
    _, p_values = f_regression(X, y)

    # Les variables avec p-value < 0.05 sont considérées comme confounders potentiels
    confounders = [features_numeric[i] for i, p in enumerate(p_values) if p < 0.05]
    return confounders

def plot_causal_dag(df, target, features, confounders=None):
    """
    Affiche un DAG simplifié pour la target et les features.
    Les colonnes non numériques sont automatiquement ignorées.
    """
    if confounders is None:
        confounders = []

    # On garde uniquement les colonnes numériques
    df_model = df[[target] + features].dropna()
    numeric_cols = df_model.select_dtypes(include="number").columns.tolist()
    features_numeric = [f for f in features if f in numeric_cols]

    # Création du graphe
    G = nx.DiGraph()
    G.add_node(target, color='red')
    
    # Ajout des features
    for f in features_numeric:
        G.add_node(f, color='skyblue')
        G.add_edge(f, target)

    # Ajout des confounders
    for c in confounders:
        if c in numeric_cols:
            G.add_node(c, color='lightgreen')
            for f in features_numeric:
                G.add_edge(c, f)

    # Dessin du graphe
    colors = [G.nodes[n].get('color', 'grey') for n in G.nodes()]
    plt.figure(figsize=(12,6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=2500, arrowsize=20)
    plt.title("DAG simplifié")
    plt.show()
