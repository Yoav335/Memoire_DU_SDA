import pandas as pd


def load_data():
    df = pd.read_csv("/Users/yoavcohen/Desktop/Memoire_DU_SDA/Mémoire/Data/owid-co2-data.csv")
    df = df.dropna()
    return df
