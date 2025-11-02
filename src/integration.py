import pandas as pd
from pathlib import Path

class DataIntegrator:
    def __init__(self):
        self.data = None
        self.files_processed = 0
    
    def find_files(self, directory, pattern="*.csv"):
        directory = Path(directory)
        files = list(directory.glob(pattern))
        files.sort()
        print(f"Found {len(files)} files")
        return files
    
    def merge_files(self, file_paths, sample_size=None):
        all_data = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                df['source_file'] = file_path.stem
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                all_data.append(df)
                self.files_processed += 1
                print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                continue
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"Combined data: {len(self.data)} rows")
        return self.data
    
    def aggregate_by_time(self, date_col, freq='M'):
        if self.data is None:
            return None
        df = self.data.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if freq == 'M':
            df['period'] = df[date_col].dt.to_period('M')
        elif freq == 'Y':
            df['period'] = df[date_col].dt.to_period('Y')
        elif freq == 'D':
            df['period'] = df[date_col].dt.date
        time_agg = df.groupby('period', observed=True).size().reset_index(name='count')
        return time_agg
    
    def aggregate_by_category(self, category_col):
        if self.data is None:
            return None
        cat_agg = self.data.groupby(category_col, observed=True).size().reset_index(name='count')
        return cat_agg.sort_values('count', ascending=False)
    
    def get_summary_stats(self):
        if self.data is None:
            return None
        stats = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'files_processed': self.files_processed,
            'missing_values': self.data.isnull().sum().sum(),
            'duplicates': self.data.duplicated().sum()
        }
        return stats
    
    def save_data(self, output_path):
        if self.data is not None:
            self.data.to_csv(output_path, index=False)
            print(f"Data saved to: {output_path}")
            return True
        return False

def merge_neighbourhoods(listings, neighbourhoods):
    """Merge listings with neighbourhoods on neighbourhood_cleansed."""
    print("\n=== NEIGHBOURHOOD MERGE DEBUG ===")
    print(f"Listings columns: {list(listings.columns)}")
    print(f"Neighbourhoods columns: {list(neighbourhoods.columns)}")
    
    if 'neighbourhood_cleansed' not in listings.columns:
        print("Error: 'neighbourhood_cleansed' not found in listings")
        return listings
    
    possible_neighbourhood_cols = ['neighbourhood', 'neighbourhood_cleansed', 'neighborhood', 'neighborhood_cleansed']
    neighbourhood_key = next((col for col in possible_neighbourhood_cols if col in neighbourhoods.columns), None)
    
    if not neighbourhood_key:
        print(f"Error: No matching neighbourhood column found in neighbourhoods.csv. Available columns: {list(neighbourhoods.columns)}")
        return listings
    
    print(f"Found matching column: '{neighbourhood_key}' in neighbourhoods")
    
    # Normalize values to handle case sensitivity and whitespace
    listings['neighbourhood_cleansed'] = listings['neighbourhood_cleansed'].astype(str).str.strip().str.lower()
    neighbourhoods[neighbourhood_key] = neighbourhoods[neighbourhood_key].astype(str).str.strip().str.lower()
    
    # Debug unique values
    print(f"\nUnique listings.neighbourhood_cleansed ({len(listings['neighbourhood_cleansed'].unique())}):")
    print(sorted(listings['neighbourhood_cleansed'].unique())[:10], "...")
    print(f"\nUnique neighbourhoods.{neighbourhood_key} ({len(neighbourhoods[neighbourhood_key].unique())}):")
    print(sorted(neighbourhoods[neighbourhood_key].unique())[:10], "...")
    
    # Check for matches
    common_values = set(listings['neighbourhood_cleansed']).intersection(set(neighbourhoods[neighbourhood_key]))
    print(f"Common values between listings.neighbourhood_cleansed and neighbourhoods.{neighbourhood_key}: {len(common_values)}")
    
    try:
        merged = listings.merge(neighbourhoods, left_on='neighbourhood_cleansed', right_on=neighbourhood_key, how='left')
        print(f"Merge completed. New shape: {merged.shape}")
        
        if neighbourhood_key in merged.columns:
            non_null_matches = merged[neighbourhood_key].notna().sum()
            print(f"Non-null matches: {non_null_matches}/{len(merged)} ({non_null_matches/len(merged)*100:.1f}%)")
        else:
            print("Warning: Merge key not found in merged dataframe")
        
        if 'neighbourhood_group' in merged.columns:
            non_null_group = merged['neighbourhood_group'].notna().sum()
            print(f"neighbourhood_group non-null values: {non_null_group}/{len(merged)} ({non_null_group/len(merged)*100:.1f}%)")
        else:
            print("Warning: 'neighbourhood_group' not found in merged dataframe")
            print("Available columns after merge:", list(merged.columns))
        
        print("=== END NEIGHBOURHOOD MERGE DEBUG ===\n")
        return merged
        
    except Exception as e:
        print(f"Error during neighbourhood merge: {e}")
        return listings

def merge_reviews(listings, reviews):
    """Merge listings with review counts and average review length."""
    print("\n=== REVIEWS MERGE DEBUG ===")
    print(f"Listings columns: {list(listings.columns)}")
    print(f"Reviews columns: {list(reviews.columns)}")
    
    if 'id' not in listings.columns:
        print("Error: 'id' not found in listings")
        return listings
    
    possible_review_id_cols = ['listing_id', 'id', 'listingID']
    review_id_col = next((col for col in possible_review_id_cols if col in reviews.columns), None)
    
    if not review_id_col:
        print(f"Error: No matching review ID column found in reviews. Available columns: {list(reviews.columns)}")
        return listings
    
    print(f"Found review ID column: '{review_id_col}'")
    
    # Ensure IDs are strings for consistent matching
    listings['id'] = listings['id'].astype(str).str.strip()
    reviews[review_id_col] = reviews[review_id_col].astype(str).str.strip()
    
    # Debug unique values
    print(f"\nUnique listings.id ({len(listings['id'].unique())}):")
    print(listings['id'].head().to_list())
    print(f"\nUnique reviews.{review_id_col} ({len(reviews[review_id_col].unique())}):")
    print(reviews[review_id_col].head().to_list())
    
    # Check for matches
    common_ids = set(listings['id']).intersection(set(reviews[review_id_col]))
    print(f"Common IDs between listings.id and reviews.{review_id_col}: {len(common_ids)}")
    
    try:
        # Count reviews per listing
        review_counts = reviews.groupby(review_id_col)['id'].count().reset_index().rename(columns={'id': 'number_of_reviews'})
        
        # Calculate average review length
        if 'comments' in reviews.columns:
            reviews['comment_length'] = reviews['comments'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
            avg_review_len = reviews.groupby(review_id_col)['comment_length'].mean().reset_index().rename(columns={'comment_length': 'avg_review_length'})
        else:
            print("Warning: 'comments' column not found in reviews, skipping avg_review_length")
            avg_review_len = pd.DataFrame(columns=[review_id_col, 'avg_review_length'])
        
        # Merge review counts
        print(f"Merging listings.id with reviews.{review_id_col}")
        df = listings.merge(review_counts, left_on='id', right_on=review_id_col, how='left')
        
        # Merge average review length if available
        if not avg_review_len.empty:
            df = df.merge(avg_review_len, left_on='id', right_on=review_id_col, how='left')
        
        # Clean up extra columns
        extra_cols = [col for col in [review_id_col + '_x', review_id_col + '_y'] if col in df.columns]
        df.drop(columns=extra_cols, inplace=True, errors='ignore')
        
        # Check merge success
        if 'number_of_reviews' in df.columns:
            non_null_reviews = df['number_of_reviews'].notna().sum()
            print(f"Merged reviews: {non_null_reviews} listings with reviews ({non_null_reviews/len(df)*100:.1f}%)")
        else:
            print("Warning: 'number_of_reviews' not found in merged dataframe")
            print("Available columns after merge:", list(df.columns))
        
        if 'avg_review_length' in df.columns:
            non_null_length = df['avg_review_length'].notna().sum()
            print(f"Avg review length added: {non_null_length} listings")
        
        print(f"Final merged shape: {df.shape}")
        print("=== END REVIEWS MERGE DEBUG ===\n")
        return df
        
    except Exception as e:
        print(f"Error during reviews merge: {e}")
        return listings

def aggregate_listings(df, group_by):
    """Aggregate listings by a column (e.g., neighbourhood_cleansed) with extended statistics."""
    try:
        if group_by not in df.columns:
            print(f"Error: '{group_by}' column not found in dataframe")
            return pd.DataFrame()
        
        # Define aggregation dictionary
        agg_dict = {
            'price': ['mean', 'median', 'min', 'max', 'std', 'count'],
            'review_scores_rating': ['mean', 'median', 'min', 'max', 'std']
        }
        
        # Perform aggregation
        agg_df = df.groupby(group_by, observed=True).agg(agg_dict).reset_index()
        
        # Flatten multi-level column names
        agg_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_df.columns]
        
        # Optional: rename columns for clarity
        agg_df.rename(columns={f'{group_by}_': group_by}, inplace=True)
        
        return agg_df
    except Exception as e:
        print(f"Aggregation error: {e}")
        return pd.DataFrame()


def sample_data(df, n_samples, method='random'):
    """Sample data with random or stratified sampling."""
    try:
        if method == 'random':
            return df.sample(n=min(n_samples, len(df)), random_state=42)
        elif method == 'stratified' and 'neighbourhood_cleansed' in df.columns:
            return df.groupby('neighbourhood_cleansed', observed=True).apply(
                lambda x: x.sample(frac=min(n_samples/len(df), 1.0), random_state=42)
            ).reset_index(drop=True)
        else:
            print("Warning: Stratified sampling requested but neighbourhood_cleansed not found, using random sampling")
            return df.sample(n=min(n_samples, len(df)), random_state=42)
    except Exception as e:
        print(f"Sampling error: {e}")
        return df