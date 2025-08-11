import pandas as pd


def load_data():
    df = pd.read_csv("Streamlit_app/utils/owid-co2-data.csv")
    df = df.dropna()
    return df
