import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import streamlit as st

def run_xgboost(df, target, features):
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    st.write("R² :", r2_score(y_test, preds))
    st.write("RMSE :", mean_squared_error(y_test, preds, squared=False))
