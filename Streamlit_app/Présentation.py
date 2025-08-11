import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from utils.data_loader import load_data

st.set_page_config(
    page_title="Mémoire de recherche :  Data science",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Container centré verticalement et horizontalement */
.header-container {
    text-align: center;
    margin-bottom: 60px;
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}

/* Logo très grand, largeur fixe */
.header-logo {
    width: 280px;
    height: auto;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    margin-bottom: 25px;
    transition: box-shadow 0.3s ease;
}
.header-logo:hover {
    box-shadow: 0 8px 25px rgba(0,114,255,0.5);
    cursor: pointer;
}

/* Titre principal - serif, sérieux, bien lisible */
.header-title {
    font-size: 3.8rem;
    font-weight: 700;
    color: #000;
    margin: 0;
    font-family: 'Georgia', serif;
    user-select: none;
}

/* Sous-titre discret, italique */
.header-subtitle {
    font-size: 1.4rem;
    color: #444444;
    margin-top: 6px;
    font-style: italic;
    font-weight: 500;
    font-family: 'Georgia', serif;
}

/* Description sous le header, centrée, bleu hypertexte */
.description {
    max-width: 900px;
    margin: 40px auto 50px auto;
    font-size: 1.15rem;
    color: #0072ff;
    font-family: 'Georgia', serif;
    line-height: 1.5;
    user-select: none;
}
</style>

<div class="header-container">
    <img class="header-logo" src="https://upload.wikimedia.org/wikipedia/commons/9/99/Logo_of_the_Pantheon-Sorbonne_University_in_Paris.png" alt="Logo La Sorbonne">
    <h1 class="header-title"> Mémoire de recherche :  Data science</h1>
    <p class="header-subtitle">Mémoire DU SDA – Yoav Cohen et Salma Lahbati</p>
    <p class="header- subtitle">Explorez les impacts des émissions de CO2 sur la croissance économique grâce à plusieurs approches statistiques et d'apprentissage automatique.</p>

</div>

""", unsafe_allow_html=True)
st.markdown("---")
# --- Description + instructions ---
st.markdown("""
Utilisez le menu latéral pour naviguer entre :  
- Présentation des modèles  
- Tests économétriques  
- Modèles Machine Learning (Ridge, Random Forest, XGBoost)  
- Modèles Deep Learning (en préparation)  
""")

# Chargement des données
df = load_data()
# --- Aperçu des données clés ---
st.markdown("---")
st.subheader(" Aperçu des données CO2 & Économiques")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Pays analysés", df['country'].nunique())
col2.metric("Années couvertes", df['year'].nunique())
col3.metric("Total CO2 (Mt)", f"{df['co2'].sum():,.0f}")
col4.metric("Moyenne CO2 / hab.", f"{df['co2_per_capita'].mean():.2f}")
col5.metric("PIB total (Md $)", f"{df['gdp'].sum() / 1e9:,.1f}")

# Calcul du PIB par habitant si nécessaire
if "gdp_per_capita" not in df.columns:
    if {"gdp", "population"}.issubset(df.columns):
        df["gdp_per_capita"] = df["gdp"] / df["population"]
    else:
        st.error("Les colonnes 'gdp' et/ou 'population' manquent dans les données.")
        st.stop()

# --- Statistiques descriptives ---
st.subheader("📊 Statistiques descriptives globales")

desc = df[['co2', 'co2_per_capita', 'gdp', 'gdp_per_capita']].describe().T
desc = desc[['mean', 'std', 'min', 'max']].round(2)
st.dataframe(desc, use_container_width=True)

st.subheader("📊 Comparaison PIB et Émissions de CO2 par habitant : France, Inde & Monde")

# Préparer les données filtrées pour ces pays + monde
countries = ['France', 'India']
df_latest_year = df['year'].max()

# Calculer la moyenne mondiale par année
df_world = df.groupby('year').agg({
    'co2_per_capita': 'mean',
    'gdp_per_capita': 'mean'
}).reset_index()
df_world['country'] = 'Monde'

# Filtrer les pays choisis
df_selected = df[df['country'].isin(countries)][['year', 'country', 'co2_per_capita', 'gdp_per_capita']]

# Concaténer pour inclure le monde
df_compare = pd.concat([df_selected, df_world], ignore_index=True)

# Tracer avec matplotlib et double axe Y
fig, ax1 = plt.subplots(figsize=(12,6))

colors = {'France':'#1f77b4', 'India':'#ff7f0e', 'Monde':'#2ca02c'}

for country in df_compare['country'].unique():
    subset = df_compare[df_compare['country'] == country]
    ax1.plot(subset['year'], subset['gdp_per_capita'], label=f"{country} - PIB/hab", color=colors[country], linestyle='-')

ax1.set_xlabel("Année")
ax1.set_ylabel("PIB par habitant (USD)", color='black')
ax1.tick_params(axis='y', labelcolor='black')

# Second axe y pour CO2 par habitant
ax2 = ax1.twinx()

for country in df_compare['country'].unique():
    subset = df_compare[df_compare['country'] == country]
    ax2.plot(subset['year'], subset['co2_per_capita'], label=f"{country} - CO2/hab", color=colors[country], linestyle='--')

ax2.set_ylabel("Émissions CO2 par habitant (tonnes)", color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Légende combinée
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=9)

ax1.set_title("Évolution du PIB et des émissions de CO2 par habitant (France, Inde et Monde)")

ax1.grid(True, linestyle='--', alpha=0.3)

st.pyplot(fig)

# --- Carte interactive : émissions totales par pays ---
if st.checkbox("Afficher la carte des émissions totales par pays"):
    df_latest = df[df['year'] == df['year'].max()]
    fig_map = px.choropleth(df_latest, locations="iso_code",
                            color="co2",
                            hover_name="country",
                            color_continuous_scale="reds",
                            title=f"Émissions totales de CO2 par pays en {df['year'].max()}",
                            labels={"co2": "CO2 (Mt)"})
    st.plotly_chart(fig_map, use_container_width=True)

# --- Carte interactive : émissions CO2 par habitant ---
if st.checkbox("Afficher la carte des émissions de CO2 par habitant"):
    df_latest = df[df['year'] == df['year'].max()]
    fig_map_pc = px.choropleth(df_latest, locations="iso_code",
                               color="co2_per_capita",
                               hover_name="country",
                               color_continuous_scale="viridis",
                               title=f"Émissions de CO2 par habitant par pays en {df['year'].max()}",
                               labels={"co2_per_capita": "CO2 par habitant (tonnes)"})
    st.plotly_chart(fig_map_pc, use_container_width=True)

# --- Nouvelle carte interactive : PIB total par pays ---
if st.checkbox("Afficher la carte du PIB total par pays"):
    df_latest = df[df['year'] == df['year'].max()]
    fig_gdp = px.choropleth(df_latest, locations="iso_code",
                            color="gdp",
                            hover_name="country",
                            color_continuous_scale="blues",
                            title=f"PIB total par pays en {df['year'].max()}",
                            labels={"gdp": "PIB total (USD)"})
    st.plotly_chart(fig_gdp, use_container_width=True)

# --- Graphique croissance moyenne annuelle du PIB par habitant ---
st.subheader("📈 Croissance moyenne annuelle du PIB par habitant (2000-2020)")

growth_df = df[df['year'].between(2000, 2020)].copy()
growth_df.sort_values(['country', 'year'], inplace=True)

growth_rate = growth_df.groupby('country').apply(
    lambda x: (np.log(x.loc[x['year'] == x['year'].max(), 'gdp_per_capita'].values[0]) -
               np.log(x.loc[x['year'] == x['year'].min(), 'gdp_per_capita'].values[0]))
              / (x['year'].max() - x['year'].min())
).reset_index(name='growth_rate')

growth_rate['growth_rate_pct'] = growth_rate['growth_rate'] * 100

fig2, ax2 = plt.subplots(figsize=(12,6))
growth_rate_sorted = growth_rate.sort_values('growth_rate_pct', ascending=False).head(15)
sns.barplot(data=growth_rate_sorted, y='country', x='growth_rate_pct', ax=ax2, palette='mako')
ax2.set_xlabel("Croissance moyenne annuelle du PIB par habitant (%)")
ax2.set_ylabel("")
ax2.set_title("Top 15 pays par croissance moyenne annuelle du PIB par habitant (2000-2020)")
st.pyplot(fig2)

# --- Footer ---
st.markdown("""
---
Créé par Yoav Cohen et Salma Lahbati– Projet Mémoire DU SDA  
Données source : Our World in Data (owid-co2-data.csv)  
""")
