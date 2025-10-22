import pandas as pd
import time
import traceback
import os
from data_collection import load_datasets, define_data_types
from data_quality import check_quality
from integration import merge_neighbourhoods, merge_reviews, aggregate_listings, sample_data
from cleaning import clean_data, identify_missing, handle_missing_values

def main():
    try:
        # Get the root directory (parent of src/)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        unprocessed_dir = os.path.join(root_dir, "unprocessed dataset")
        processed_dir = os.path.join(root_dir, "processed dataset")
        
        # Create processed dataset directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)
        
        print(f"Root directory: {root_dir}")
        print(f"Unprocessed dataset directory: {unprocessed_dir}")
        print(f"Processed dataset directory: {processed_dir}")

        # Step 1: Data Collection
        start_time = time.time()
        print("\n=== STEP 1: DATA COLLECTION ===")
        datasets = load_datasets(unprocessed_dir)
        listings = datasets.get('listings')
        reviews = datasets.get('reviews')
        neighbourhoods = datasets.get('neighbourhoods')
        
        if not datasets:
            print("Error: No datasets loaded. Check 'unprocessed dataset' directory.")
            return
            
        print(f"Available datasets: {list(datasets.keys())}")
        print(f"Data loading took {time.time() - start_time:.2f} seconds")

        # Step 2: Define Data Types
        start_time = time.time()
        print("\n=== STEP 2: DEFINE DATA TYPES ===")
        if listings is not None:
            listings = define_data_types(listings, "listings")
            print(f"Listings shape after type definition: {listings.shape}")
        else:
            print("Warning: listings.csv not loaded.")
        if reviews is not None:
            # Optional: Sample reviews to improve performance
            if len(reviews) > 100000:
                reviews = reviews.sample(n=100000, random_state=42)
                print(f"Sampled reviews to {len(reviews)} rows for performance")
            reviews = define_data_types(reviews, "reviews")
            print(f"Reviews shape after type definition: {reviews.shape}")
        else:
            print("Warning: reviews.csv not loaded.")
        if neighbourhoods is not None:
            neighbourhoods = define_data_types(neighbourhoods, "neighbourhoods")
            print(f"Neighbourhoods shape after type definition: {neighbourhoods.shape}")
        else:
            print("Warning: neighbourhoods.csv not loaded.")
        print(f"Type definition took {time.time() - start_time:.2f} seconds")

        # Step 3: Data Quality Assessment
        start_time = time.time()
        print("\n=== STEP 3: DATA QUALITY ASSESSMENT ===")
        if listings is not None:
            print("Assessing data quality for listings...")
            quality_report = check_quality(listings, "listings")
            print("Listings Quality Report Summary:")
            print(f"  - Rows: {quality_report.get('rows', 0)}")
            print(f"  - Columns: {quality_report.get('columns', 0)}")
            print(f"  - Total missing values: {sum(quality_report.get('missing_values', {}).values())}")
            if 'duplicate_ids' in quality_report:
                print(f"  - Duplicate IDs: {quality_report['duplicate_ids']}")
            if 'price_outliers' in quality_report:
                print(f"  - Price outliers: {quality_report['price_outliers']}")
        else:
            print("Skipping quality assessment: listings is None")
        print(f"Quality assessment took {time.time() - start_time:.2f} seconds")

        # Step 4: Integration
        start_time = time.time()
        print("\n=== STEP 4: INTEGRATION ===")
        if listings is not None and neighbourhoods is not None:
            listings = merge_neighbourhoods(listings, neighbourhoods)
            print(f"After neighbourhood merge: {listings.shape}")
        else:
            print("Skipping neighbourhood merge: listings or neighbourhoods is None")
        if listings is not None and reviews is not None:
            listings = merge_reviews(listings, reviews)
            print(f"After reviews merge: {listings.shape}")
        else:
            print("Skipping reviews merge: listings or reviews is None")
        print(f"Integration took {time.time() - start_time:.2f} seconds")

        # Step 5: Aggregation
        start_time = time.time()
        print("\n=== STEP 5: AGGREGATION ===")
        if listings is not None:
            print("Aggregating by neighbourhood...")
            try:
                neighbourhood_agg = aggregate_listings(listings, 'neighbourhood_cleansed')
                if not neighbourhood_agg.empty:
                    print("Aggregated by neighbourhood (top 10):")
                    print(neighbourhood_agg.head(10).to_string())
                else:
                    print("No aggregation data returned")
            except Exception as e:
                print(f"Aggregation error: {e}")
        else:
            print("Skipping aggregation: listings is None")
        print(f"Aggregation took {time.time() - start_time:.2f} seconds")

        # Step 6: Sampling
        start_time = time.time()
        print("\n=== STEP 6: SAMPLING ===")
        if listings is not None and len(listings) > 1000:
            print("Sampling data...")
            try:
                listings = sample_data(listings, n_samples=int(0.1 * len(listings)), method='stratified')
                print(f"Sampled to {len(listings)} rows")
            except Exception as e:
                print(f"Sampling error: {e}")
                print("Falling back to random sampling")
                listings = sample_data(listings, n_samples=int(0.1 * len(listings)), method='random')
                print(f"Sampled to {len(listings)} rows")
        else:
            print(f"Skipping sampling: listings has {len(listings) if listings is not None else 0} rows (<=1000)")
        print(f"Sampling took {time.time() - start_time:.2f} seconds")

        # Step 7: Cleaning
        start_time = time.time()
        print("\n=== STEP 7: CLEANING ===")
        if listings is not None:
            listings = clean_data(listings, "listings")
            print(f"After cleaning: {listings.shape}")
        else:
            print("Skipping listings cleaning: listings is None")
        if reviews is not None:
            reviews = clean_data(reviews, "reviews")
        else:
            print("Skipping reviews cleaning: reviews is None")
        print(f"Cleaning took {time.time() - start_time:.2f} seconds")

        # Step 8: Handle Missing Values
        start_time = time.time()
        print("\n=== STEP 8: HANDLE MISSING VALUES ===")
        if listings is not None:
            print("Handling missing values...")
            missing_report = identify_missing(listings)
            total_missing = sum(missing_report.values())
            print(f"Total missing values before imputation: {total_missing}")
            listings = handle_missing_values(listings, "listings")
            # Verify missing values after imputation
            missing_after = identify_missing(listings)
            total_missing_after = sum(missing_after.values())
            print(f"Total missing values after imputation: {total_missing_after}")
        else:
            print("Skipping missing value handling: listings is None")
        print(f"Missing value handling took {time.time() - start_time:.2f} seconds")

        # Step 9: Save Processed Data
        start_time = time.time()
        print("\n=== STEP 9: SAVE PROCESSED DATA ===")
        if listings is not None:
            output_path = os.path.join(processed_dir, "integrated_listings.csv")
            listings.to_csv(output_path, index=False)
            print(f"Processed data saved to: {output_path}")
            print(f"Final dataset shape: {listings.shape}")
            print("Final dataset columns:", list(listings.columns))
        else:
            print("Skipping save: listings is None")
        print(f"Saving took {time.time() - start_time:.2f} seconds")
        
        print("\n=== PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY ===")

    except Exception as e:
        print("\n=== ERROR OCCURRED ===")
        print(f"Exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()