# Projet Analyse Économétrique et Modélisation des Émissions

Ce projet propose une application interactive développée avec Streamlit pour analyser l’impact des émissions de gaz à effet de serre sur la croissance économique par zone géographique.  
Il combine plusieurs approches statistiques et de machine learning, incluant :

- Régressions économétriques classiques (OLS)  
- Modèles de machine learning (ex. XGBoost)  
- Modèles de deep learning (à venir)  

L’objectif est de comparer les performances des différents modèles, visualiser les résultats, et faciliter l’interprétation des relations entre variables économiques et environnementales.

---

## Installation

```bash
git clone https://github.com/Yoav335/Memoire_DU_SDA.git

cd Memoire_DU_SDA


```bash
python3 -m venv MemoireM1
source MemoireM1/bin/activate


pip install -r requirements.txt
streamlit run app.py


### Requirements.txt

streamlit
pandas
statsmodels
scikit-learn
xgboost
matplotlib
seaborn
