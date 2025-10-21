import pandas as pd
import re

def clean_price_column(df):
    """Remove currency symbols, commas, and convert price to numeric."""
    if "price" in df.columns:
        df["price"] = (
            df["price"]
            .astype(str)
            .apply(lambda x: re.sub(r"[^\d.]", "", x))
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    return df