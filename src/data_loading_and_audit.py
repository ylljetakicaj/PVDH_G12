import pandas as pd
import numpy as np

df = pd.read_csv("unprocessed dataset/listings.csv")

print("Forma e datasetit:", df.shape)
print("Kolonat:", df.columns.tolist())

print("\n Llojet e kolonave:")
print(df.dtypes)

print("\n 5 rreshtat e parë:")
print(df.head())

print("\n Mungesat (% për kolonë):")
print((df.isna().mean() * 100).round(2).sort_values(ascending=False))

dupes = df.duplicated().sum()
print(f"\n Numri i rreshtave të dublikuar: {dupes}")
