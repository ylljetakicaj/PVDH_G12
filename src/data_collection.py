import pandas as pd
from pathlib import Path

def load_datasets(directory_path):
    """Load all CSV files from the specified directory path."""
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return {}
    
    datasets = {}
    expected_files = ['listings.csv', 'reviews.csv', 'neighbourhoods.csv']
    csv_files = list(directory.glob("*.csv"))
    
    print(f"Found {len(csv_files)} CSV files in {directory}")
    
    if not csv_files:
        print(f"Error: No CSV files found in {directory}")
        return {}
    
    for file in csv_files:
        print(f"Processing: {file.name}")
        if file.name in expected_files:
            try:
                df = pd.read_csv(file)
                datasets[file.stem] = df
                print(f"✓ Loaded {file.name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"✗ Error loading {file.name}: {e}")
        else:
            print(f"⚠ Skipping unexpected file: {file.name}")
    
    return datasets

def define_data_types(df, dataset_name):
    """Define data types for each dataset."""
    print(f"Defining data types for {dataset_name} ({len(df)} rows)")
    
    if dataset_name == "listings":
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str)
        if 'host_id' in df.columns:
            df['host_id'] = df['host_id'].astype(str)
        if 'price' in df.columns:
            df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float, errors='ignore')
        for col in ['host_response_rate', 'host_acceptance_rate']:
            if col in df.columns:
                df[col] = df[col].replace('%', '', regex=True).astype(float, errors='ignore') / 100
        for col in ['last_scraped', 'first_review', 'last_review']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        for col in ['neighbourhood_cleansed', 'property_type', 'room_type', 'host_is_superhost']:
            if col in df.columns:
                df[col] = df[col].astype('category')
    elif dataset_name == "reviews":
        if 'listing_id' in df.columns:
            df['listing_id'] = df['listing_id'].astype(str)
        if 'reviewer_id' in df.columns:
            df['reviewer_id'] = df['reviewer_id'].astype(str)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
    elif dataset_name == "neighbourhoods":
        for col in ['neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group']:
            if col in df.columns:
                df[col] = df[col].astype('category')
    
    return df