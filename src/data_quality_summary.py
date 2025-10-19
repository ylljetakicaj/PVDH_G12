import pandas as pd

df = pd.read_csv("../unprocessed dataset/listings.csv")

summary = pd.DataFrame({
    "Kolona": df.columns,
    "Lloji": df.dtypes.astype(str),
    "Mungesa (%)": (df.isna().mean() * 100).round(2),
    "Vlera Unike": [df[c].nunique(dropna=True) for c in df.columns],
    "Shembull 1": [df[c].dropna().unique()[0] if df[c].dropna().nunique() > 0 else None for c in df.columns],
    "Shembull 2": [df[c].dropna().unique()[1] if df[c].dropna().nunique() > 1 else None for c in df.columns],
})

summary = summary.sort_values("Mungesa (%)", ascending=False)

print(summary.to_string(index=False))
