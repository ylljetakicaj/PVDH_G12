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

def identify_missing(df):
    """Identify missing values in the dataset."""
    missing = df.isnull().sum().to_dict()
    print("Missing values per column:")
    for col, count in missing.items():
        if count > 0:
            print(f"  {col}: {count}")
    return missing

def handle_missing_values(df, dataset_name):
    """Handle missing values with appropriate strategies."""
    print(f"Handling missing values for {dataset_name}...")
    if dataset_name == "listings":
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['category']).columns
        
        # Filter out columns that are entirely NaN
        valid_numeric_cols = [col for col in numeric_cols if df[col].notna().any()]
        all_nan_numeric_cols = [col for col in numeric_cols if df[col].isna().all()]
        
        print(f"Valid numeric columns for imputation: {valid_numeric_cols}")
        print(f"All-NaN numeric columns (filled with 0): {all_nan_numeric_cols}")
        
        # Impute valid numeric columns
        if valid_numeric_cols:
            imputer = SimpleImputer(strategy='median')
            imputed_data = imputer.fit_transform(df[valid_numeric_cols])
            df[valid_numeric_cols] = pd.DataFrame(imputed_data, columns=valid_numeric_cols, index=df.index)
        
        # Fill all-NaN numeric columns with 0
        for col in all_nan_numeric_cols:
            df[col] = df[col].fillna(0)
        
        # Impute categorical columns
        for col in cat_cols:
            if not df[col].isna().all():
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                # Add 'Unknown' to categories if needed
                if 'Unknown' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(['Unknown'])
                df[col] = df[col].fillna(mode_value)
            else:
                print(f"Warning: {col} is entirely NaN, filling with 'Unknown'")
                if 'Unknown' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(['Unknown'])
                df[col] = df[col].fillna('Unknown')
        
        # Special handling for review-related columns
        if 'number_of_reviews' in df.columns:
            df['number_of_reviews'] = df['number_of_reviews'].fillna(0)
        if 'avg_review_length' in df.columns:
            df['avg_review_length'] = df['avg_review_length'].fillna(0)
        # Handle number_of_reviews_y (from merge)
        if 'number_of_reviews_y' in df.columns:
            df['number_of_reviews_y'] = df['number_of_reviews_y'].fillna(0)
            
    elif dataset_name == "reviews":
        if 'comments' in df.columns:
            initial_rows = len(df)
            df = df.dropna(subset=['comments'])
            print(f"Dropped {initial_rows - len(df)} rows with missing comments in reviews")
    
    return df

