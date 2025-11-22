import pandas as pd
import time
import traceback
import os
from data_collection import load_datasets, define_data_types
from data_quality import check_quality
from integration import merge_neighbourhoods, merge_reviews, aggregate_listings, sample_data
from cleaning import clean_data, identify_missing, handle_missing_values
from advanced_preprocessing import AdvancedPreprocessor
from outlier_detection import OutlierDetector


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
        else:
            print("Warning: listings.csv not loaded.")

        if reviews is not None:
            if len(reviews) > 100000:
                reviews = reviews.sample(n=100000, random_state=42)
                print(f"Sampled reviews to {len(reviews)} rows for performance")
            reviews = define_data_types(reviews, "reviews")
        else:
            print("Warning: reviews.csv not loaded.")

        if neighbourhoods is not None:
            neighbourhoods = define_data_types(neighbourhoods, "neighbourhoods")
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
            missing_after = identify_missing(listings)
            total_missing_after = sum(missing_after.values())
            print(f"Total missing values after imputation: {total_missing_after}")
        else:
            print("Skipping missing value handling: listings is None")
        print(f"Missing value handling took {time.time() - start_time:.2f} seconds")

        # Step 9: Advanced Preprocessing
        start_time = time.time()
        print("\n=== STEP 9: ADVANCED PREPROCESSING ===")
        if listings is not None:
            preprocessor = AdvancedPreprocessor()
            original_listings = listings.copy()

            print("\n--- 9.1: Creating Derived Properties ---")
            listings = preprocessor.create_derived_properties(listings)

            print("\n--- 9.2: Discretization and Binarization ---")
            try:
                listings = preprocessor.discretize_and_binarize(listings)
                print(f"After discretization and binarization: {listings.shape}")
            except Exception as e:
                print(f"Discretization and binarization failed: {e}")

            print("\n--- 9.3: Property Subset Selection ---")
            listings = preprocessor.select_property_subsets(listings, subset_type='high_value')
            print(f"After property subset selection: {listings.shape}")

            print("\n--- 9.4: Data Transformations ---")
            listings = preprocessor.apply_transformations(listings)

            print("\n--- 9.5: Dimension Reduction ---")
            listings_pca = preprocessor.dimension_reduction(
                listings,
                target_col='price',
                method='pca',
                n_components=0.95,
                feature_types='numeric'
            )
            listings_selected = preprocessor.dimension_reduction(
                listings,
                target_col='price',
                method='univariate',
                n_components=20,
                feature_types='numeric'
            )

            summary = preprocessor.get_preprocessing_summary(original_listings, listings)
        else:
            print("Skipping advanced preprocessing: listings is None")
        print(f"Advanced preprocessing took {time.time() - start_time:.2f} seconds")

        # Step 10: Outlier Detection
        start_time = time.time()
        print("\n=== STEP 10: OUTLIER DETECTION ===")
        if listings is not None:
            print("Detecting outliers using multiple methods...")
            detector = OutlierDetector()
            
            # Identify numeric columns for outlier detection
            numeric_cols = listings.select_dtypes(include=['number']).columns.tolist()
            
            # Remove columns that are not meaningful for outlier detection
            # (like IDs, indices, or columns that are already binary flags)
            exclude_cols = ['id', 'host_id', 'scrape_id', 'listing_id', 'reviewer_id']
            meaningful_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # Also exclude columns that are already outlier flags (if any)
            meaningful_cols = [col for col in meaningful_cols if not col.startswith('outlier_')]
            
            # Select key columns for IQR and Z-Score detection
            # Focus on important features like price, ratings, counts, etc.
            key_cols = []
            priority_keywords = ['price', 'rating', 'score', 'review', 'accommodates', 
                               'bedroom', 'bathroom', 'availability', 'count']
            
            for col in meaningful_cols:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in priority_keywords):
                    key_cols.append(col)
            
            # If we have too many columns, limit to top 15 most important ones
            if len(key_cols) > 15:
                # Prioritize columns with more variation
                key_cols_var = [(col, listings[col].var()) for col in key_cols if listings[col].var() > 0]
                key_cols_var.sort(key=lambda x: x[1], reverse=True)
                key_cols = [col for col, _ in key_cols_var[:15]]
            
            print(f"Using {len(key_cols)} key columns for IQR and Z-Score detection")
            print(f"Key columns: {', '.join(key_cols[:10])}{'...' if len(key_cols) > 10 else ''}")
            
            # 1. IQR Outliers
            print("\n--- 10.1: IQR Outlier Detection ---")
            try:
                listings = detector.detect_iqr(listings, key_cols)
                iqr_summary = {k: v for k, v in detector.get_summary().items() if k.startswith('iqr_')}
                total_iqr = sum(iqr_summary.values())
                print(f"Total IQR outliers detected: {total_iqr}")
                if iqr_summary:
                    print("Top IQR outliers by column:")
                    sorted_iqr = sorted(iqr_summary.items(), key=lambda x: x[1], reverse=True)[:5]
                    for col, count in sorted_iqr:
                        print(f"  {col.replace('iqr_', '')}: {count} outliers")
            except Exception as e:
                print(f"IQR detection error: {e}")
            
            # 2. Z-Score Outliers
            print("\n--- 10.2: Z-Score Outlier Detection ---")
            try:
                listings = detector.detect_zscore(listings, key_cols, threshold=3)
                zscore_summary = {k: v for k, v in detector.get_summary().items() if k.startswith('zscore_')}
                total_zscore = sum(zscore_summary.values())
                print(f"Total Z-Score outliers detected: {total_zscore}")
                if zscore_summary:
                    print("Top Z-Score outliers by column:")
                    sorted_zscore = sorted(zscore_summary.items(), key=lambda x: x[1], reverse=True)[:5]
                    for col, count in sorted_zscore:
                        print(f"  {col.replace('zscore_', '')}: {count} outliers")
            except Exception as e:
                print(f"Z-Score detection error: {e}")
            
            # 3. Isolation Forest
            print("\n--- 10.3: Isolation Forest Outlier Detection ---")
            try:
                # Use all meaningful numeric columns for multivariate methods
                feature_cols = meaningful_cols.copy()
                # Limit to reasonable number of features for performance
                if len(feature_cols) > 50:
                    # Select features with highest variance
                    feature_vars = [(col, listings[col].var()) for col in feature_cols if listings[col].var() > 0]
                    feature_vars.sort(key=lambda x: x[1], reverse=True)
                    feature_cols = [col for col, _ in feature_vars[:50]]
                
                listings = detector.detect_isolation_forest(listings, feature_cols, contamination=0.05)
                iforest_count = detector.get_summary().get('isolation_forest', 0)
                print(f"Isolation Forest outliers detected: {iforest_count}")
            except Exception as e:
                print(f"Isolation Forest detection error: {e}")
            
            # 4. Local Outlier Factor (LOF)
            print("\n--- 10.4: Local Outlier Factor (LOF) Detection ---")
            try:
                listings = detector.detect_lof(listings, feature_cols, contamination=0.05)
                lof_count = detector.get_summary().get('lof', 0)
                print(f"LOF outliers detected: {lof_count}")
            except Exception as e:
                print(f"LOF detection error: {e}")
            
            # 5. Mahalanobis Distance (if PCA components exist)
            print("\n--- 10.5: Mahalanobis Distance Detection ---")
            try:
                # Check if we have PCA components from advanced preprocessing
                pca_cols = [col for col in listings.columns if 'pca' in col.lower() or col.startswith('PC')]
                if len(pca_cols) >= 2:
                    listings = detector.detect_mahalanobis(listings, pca_cols, threshold=3.5)
                    mahalanobis_count = detector.get_summary().get('mahalanobis', 0)
                    print(f"Mahalanobis Distance outliers detected: {mahalanobis_count}")
                else:
                    print("Skipping Mahalanobis: Not enough PCA components found")
            except Exception as e:
                print(f"Mahalanobis detection error: {e}")
            
            # 6. Compute Combined Outlier Score
            print("\n--- 10.6: Computing Combined Outlier Score ---")
            try:
                listings = detector.compute_outlier_score(listings)
                
                # Map outlier types
                if 'outlier_score' in listings.columns:
                    listings['outlier_type'] = listings['outlier_score'].apply(detector.map_outlier_type)
                    
                    # Print summary
                    outlier_type_counts = listings['outlier_type'].value_counts()
                    print("Outlier type distribution:")
                    for outlier_type, count in outlier_type_counts.items():
                        print(f"  {outlier_type}: {count} ({count/len(listings)*100:.1f}%)")
                    
                    # Count total outliers (score > 0)
                    total_outliers = (listings['outlier_score'] > 0).sum()
                    print(f"\nTotal records with at least one outlier flag: {total_outliers} ({total_outliers/len(listings)*100:.1f}%)")
                    
                    # Count extreme outliers (score >= 3)
                    extreme_outliers = (listings['outlier_score'] >= 3).sum()
                    print(f"Extreme outliers (score >= 3): {extreme_outliers} ({extreme_outliers/len(listings)*100:.1f}%)")
            except Exception as e:
                print(f"Outlier score computation error: {e}")
            
            # Print final summary
            print("\n--- Outlier Detection Summary ---")
            final_summary = detector.get_summary()
            print(f"Total outlier detection methods executed: {len([k for k in final_summary.keys() if not k.startswith('iqr_') and not k.startswith('zscore_')])}")
            print(f"Dataset shape after outlier detection: {listings.shape}")
            
        else:
            print("Skipping outlier detection: listings is None")
        print(f"Outlier detection took {time.time() - start_time:.2f} seconds")

        # Step 11: Save Processed Data (all integrated in one CSV)
        start_time = time.time()
        print("\n=== STEP 11: SAVE PROCESSED DATA ===")

        if listings is not None:
            final_df = listings.copy()

            # Merge PCA features if they exist
            if 'listings_pca' in locals() and not listings_pca.empty:
                pca_cols = [c for c in listings_pca.columns if c not in final_df.columns]
                final_df = pd.concat([final_df, listings_pca[pca_cols]], axis=1)

            # Merge univariate-selected features if they exist
            if 'listings_selected' in locals() and not listings_selected.empty:
                selected_cols = [c for c in listings_selected.columns if c not in final_df.columns]
                final_df = pd.concat([final_df, listings_selected[selected_cols]], axis=1)

            # Save integrated file
            integrated_path = os.path.join(processed_dir, "integrated_processed_listings.csv")
            final_df.to_csv(integrated_path, index=False)
            print(f"All processed data saved in one CSV: {integrated_path}")
            print(f"Final dataset shape: {final_df.shape}")
            print(f"Final dataset columns: {len(final_df.columns)} total columns")
            
            # Count outlier detection columns
            outlier_cols = [col for col in final_df.columns if col.startswith('outlier_')]
            if outlier_cols:
                print(f"Outlier detection columns included: {len(outlier_cols)}")
                print(f"  - Outlier flags: {len([c for c in outlier_cols if not c.endswith('_score') and c != 'outlier_type'])}")
                print(f"  - Outlier score: {'outlier_score' in outlier_cols}")
                print(f"  - Outlier type: {'outlier_type' in outlier_cols}")

        else:
            print("Skipping save: listings is None")

        print(f"Saving took {time.time() - start_time:.2f} seconds")
        print("\n=== DATA PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY ===")
        print("Pipeline includes: Data Collection, Quality Assessment, Integration, Cleaning,")
        print("Advanced Preprocessing, and Outlier Detection")

    except Exception as e:
        print("\n=== ERROR OCCURRED ===")
        print(f"Exception: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
