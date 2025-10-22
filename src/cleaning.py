import pandas as pd
from sklearn.impute import SimpleImputer

def clean_data(df, dataset_name):
    """Clean dataset by removing duplicates and fixing inconsistencies."""
    print(f"Cleaning {dataset_name}...")
    if dataset_name == "listings":
        if 'id' in df.columns:
            initial_rows = len(df)
            df = df.drop_duplicates(subset=['id'])
            print(f"Removed {initial_rows - len(df)} duplicate IDs in listings")
        if 'price' in df.columns:
            df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float, errors='ignore')
    elif dataset_name == "reviews":
        if 'id' in df.columns:
            initial_rows = len(df)
            df = df.drop_duplicates(subset=['id'])
            print(f"Removed {initial_rows - len(df)} duplicate IDs in reviews")
    return df
