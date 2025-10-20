import pandas as pd

def check_quality(df, dataset_name):
    """Assess data quality."""
    report = {
        'dataset': dataset_name,
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'numeric_stats': df.select_dtypes(include='number').describe().to_dict()
    }
    if dataset_name == "listings":
        if 'id' in df.columns:
            report['duplicate_ids'] = df['id'].duplicated().sum()
        if 'price' in df.columns:
            report['price_outliers'] = len(df[df['price'] > df['price'].quantile(0.99)])
    return report