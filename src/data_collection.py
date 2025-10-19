import pandas as pd

def load_listings(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def load_reviews(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def load_neighbourhoods(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df
