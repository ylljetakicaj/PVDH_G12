import pandas as pd
import numpy as np
import time
import traceback
import os
from data_collection import load_datasets, define_data_types
from data_quality import check_quality
from integration import merge_neighbourhoods, merge_reviews, aggregate_listings, sample_data
from cleaning import clean_data, identify_missing, handle_missing_values
from advanced_preprocessing import AdvancedPreprocessor
from outlier_detection import OutlierDetector
from eda_analyzer import EDAAnalyzer


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

        start_time = time.time()
        print("\n=== STEP 10: EXPLORATORY DATA ANALYSIS (EDA) ===")
        if listings is not None:
            eda_output_dir = os.path.join(root_dir, "eda_plots")
            os.makedirs(eda_output_dir, exist_ok=True)
            
            print("Performing exploratory data analysis...")
            eda = EDAAnalyzer(save_plots=True, output_dir=eda_output_dir)
            
            numeric_cols = listings.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_cols = listings.select_dtypes(include=['object', 'category']).columns.tolist()
            
            numeric_cols = [col for col in numeric_cols if not col.startswith('outlier_')]
            
            print(f"Analyzing {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")
            
            # 1. STATISTIKA PËRMBLEDHËSE 
            print("\n--- 10.1: Summary Statistics ---")
            try:
                if len(numeric_cols) > 0:
                    print("Computing numerical summary statistics...")
                    analysis_numeric_cols = numeric_cols[:30] if len(numeric_cols) > 30 else numeric_cols
                    numerical_summary = eda.numerical_summary(listings, columns=analysis_numeric_cols)
                    print(f"Numerical summary computed for {len(analysis_numeric_cols)} columns")
                    print("\nTop 10 numeric columns summary:")
                    print(numerical_summary.head(10).to_string())
                    
                    summary_path = os.path.join(eda_output_dir, "numerical_summary.csv")
                    numerical_summary.to_csv(summary_path)
                    print(f"Numerical summary saved to: {summary_path}")
                else:
                    print("No numeric columns found for summary")
                
                if len(categorical_cols) > 0:
                    print("\nComputing categorical summary statistics...")
                    analysis_cat_cols = categorical_cols[:20] if len(categorical_cols) > 20 else categorical_cols
                    categorical_summary = eda.categorical_summary(listings, columns=analysis_cat_cols)
                    print(f"Categorical summary computed for {len(analysis_cat_cols)} columns")
                    
                    print("\nTop 5 categorical columns summary:")
                    for i, (col, counts) in enumerate(list(categorical_summary.items())[:5]):
                        print(f"\n{col}:")
                        print(counts.head(10).to_string())
                else:
                    print("No categorical columns found for summary")
                    
            except Exception as e:
                print(f"Summary statistics error: {e}")
                import traceback
                traceback.print_exc()
            
            # 2. ANALIZA MULTIVARIANTE

            print("\n--- 10.2: Multivariate Analysis ---")
            try:
                if len(numeric_cols) > 1:
                    print("Computing correlation matrix...")
                    corr_cols = numeric_cols[:25] if len(numeric_cols) > 25 else numeric_cols
                    
                    correlation_matrix = eda.correlation_matrix(listings, columns=corr_cols, figsize=(14, 12))
                    print(f"Correlation matrix computed for {len(corr_cols)} columns")
                    
                    if correlation_matrix is not None and not correlation_matrix.empty:
                        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
                        corr_upper = correlation_matrix.where(mask)
                        
                        corr_pairs = []
                        for i in range(len(corr_upper.columns)):
                            for j in range(i+1, len(corr_upper.columns)):
                                val = corr_upper.iloc[i, j]
                                if not pd.isna(val):
                                    corr_pairs.append((corr_upper.columns[i], corr_upper.columns[j], val))
                        
                        if corr_pairs:
                            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                            print("\nTop 10 strongest correlations:")
                            for col1, col2, corr_val in corr_pairs[:10]:
                                print(f"  {col1} <-> {col2}: {corr_val:.3f}")
                        
                        corr_path = os.path.join(eda_output_dir, "correlation_matrix.csv")
                        correlation_matrix.to_csv(corr_path)
                        print(f"Correlation matrix saved to: {corr_path}")
                else:
                    print("Not enough numeric columns for correlation analysis")
                
                if len(numeric_cols) > 2:
                    print("\nPerforming PCA analysis...")
                    pca_cols = []
                    for col in numeric_cols[:30]: 
                        if listings[col].var() > 0 and not listings[col].isnull().all():
                            pca_cols.append(col)
                    
                    if len(pca_cols) >= 2:
                        n_components = min(10, len(pca_cols))
                        pca_df, explained_var = eda.pca_analysis(listings, columns=pca_cols, n_components=n_components)
                        
                        print(f"PCA analysis completed with {n_components} components")
                        print(f"Explained variance by component:")
                        for i, var in enumerate(explained_var):
                            print(f"  PC{i+1}: {var:.2%}")
                        
                        cumulative_var = np.cumsum(explained_var)
                        print(f"\nCumulative explained variance:")
                        for i, cum_var in enumerate(cumulative_var):
                            print(f"  PC1-PC{i+1}: {cum_var:.2%}")
                        
                        if n_components >= 2:
                            try:
                                labels = None
                                if 'price' in listings.columns:
                                    labels = listings['price'].values
                                eda.plot_pca(pca_df.iloc[:, :2], labels=labels)
                                print("PCA plot saved")
                            except Exception as e:
                                print(f"PCA plot error: {e}")
                        
                        pca_path = os.path.join(eda_output_dir, "pca_components.csv")
                        pca_df.to_csv(pca_path, index=False)
                        print(f"PCA components saved to: {pca_path}")
                    else:
                        print("Not enough valid numeric columns for PCA (need at least 2)")
                else:
                    print("Not enough numeric columns for PCA analysis (need at least 2)")
                
                if len(numeric_cols) >= 2:
                    print("\nGenerating pairplot (sampled if dataset is large)...")
                    try:
                        pairplot_cols = []
                        priority_keywords = ['price', 'rating', 'score', 'review', 'accommodates']
                        
                        for col in numeric_cols:
                            col_lower = col.lower()
                            if any(keyword in col_lower for keyword in priority_keywords):
                                pairplot_cols.append(col)
                                if len(pairplot_cols) >= 5:
                                    break
                        
                        if len(pairplot_cols) < 5:
                            remaining = [col for col in numeric_cols[:10] if col not in pairplot_cols]
                            pairplot_cols.extend(remaining[:5-len(pairplot_cols)])
                        
                        if len(pairplot_cols) >= 2:
                            plot_data = listings[pairplot_cols].copy()
                            if len(plot_data) > 1000:
                                plot_data = plot_data.sample(n=1000, random_state=42)
                                print(f"Sampled {len(plot_data)} rows for pairplot")
                            
                            plot_data = plot_data.dropna()
                            
                            if len(plot_data) > 0 and len(pairplot_cols) >= 2:
                                eda.pairplot(plot_data, columns=pairplot_cols)
                                print(f"Pairplot generated for {len(pairplot_cols)} columns")
                            else:
                                print("Not enough valid data for pairplot")
                        else:
                            print("Not enough columns selected for pairplot")
                    except Exception as e:
                        print(f"Pairplot error: {e}")
                        import traceback
                        traceback.print_exc()
                
                print("\nComputing grouped summaries...")
                try:
                    if 'neighbourhood_cleansed' in listings.columns and 'price' in listings.columns:
                        grouped_summary = eda.grouped_summary(listings, 'neighbourhood_cleansed', 'price')
                        print("\nPrice summary by neighbourhood (top 10):")
                        print(grouped_summary.head(10).to_string())
                        
                        grouped_path = os.path.join(eda_output_dir, "grouped_summary_neighbourhood_price.csv")
                        grouped_summary.to_csv(grouped_path)
                        print(f"Grouped summary saved to: {grouped_path}")
                    
                    if 'room_type' in listings.columns and 'price' in listings.columns:
                        grouped_summary = eda.grouped_summary(listings, 'room_type', 'price')
                        print("\nPrice summary by room type:")
                        print(grouped_summary.to_string())
                except Exception as e:
                    print(f"Grouped summary error: {e}")
                    
            except Exception as e:
                print(f"Multivariate analysis error: {e}")
                import traceback
                traceback.print_exc()
            
            print("\n--- EDA Summary ---")
            eda_summary = eda.get_summary()
            print(f"EDA analysis completed. Summary keys: {list(eda_summary.keys())}")
            print(f"EDA plots saved to: {eda_output_dir}")
            
        else:
            print("Skipping EDA: listings is None")
        print(f"EDA took {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        print("\n=== STEP 11: OUTLIER DETECTION ===")
        if listings is not None:
            print("Detecting outliers using multiple methods...")
            detector = OutlierDetector()
            
            numeric_cols = listings.select_dtypes(include=['number']).columns.tolist()
            

            exclude_cols = ['id', 'host_id', 'scrape_id', 'listing_id', 'reviewer_id']
            meaningful_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            meaningful_cols = [col for col in meaningful_cols if not col.startswith('outlier_')]
            key_cols = []
            priority_keywords = ['price', 'rating', 'score', 'review', 'accommodates', 
                               'bedroom', 'bathroom', 'availability', 'count']
            
            for col in meaningful_cols:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in priority_keywords):
                    key_cols.append(col)

            if len(key_cols) > 15:
                key_cols_var = [(col, listings[col].var()) for col in key_cols if listings[col].var() > 0]
                key_cols_var.sort(key=lambda x: x[1], reverse=True)
                key_cols = [col for col, _ in key_cols_var[:15]]
            
            print(f"Using {len(key_cols)} key columns for IQR and Z-Score detection")
            print(f"Key columns: {', '.join(key_cols[:10])}{'...' if len(key_cols) > 10 else ''}")

            print("\n--- 11.1: IQR Outlier Detection ---")
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
            
            print("\n--- 11.2: Z-Score Outlier Detection ---")
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
            
            print("\n--- 11.3: Isolation Forest Outlier Detection ---")
            try:
                feature_cols = meaningful_cols.copy()
                if len(feature_cols) > 50:
                    feature_vars = [(col, listings[col].var()) for col in feature_cols if listings[col].var() > 0]
                    feature_vars.sort(key=lambda x: x[1], reverse=True)
                    feature_cols = [col for col, _ in feature_vars[:50]]
                
                listings = detector.detect_isolation_forest(listings, feature_cols, contamination=0.05)
                iforest_count = detector.get_summary().get('isolation_forest', 0)
                print(f"Isolation Forest outliers detected: {iforest_count}")
            except Exception as e:
                print(f"Isolation Forest detection error: {e}")
            
            print("\n--- 11.4: Local Outlier Factor (LOF) Detection ---")
            try:
                listings = detector.detect_lof(listings, feature_cols, contamination=0.05)
                lof_count = detector.get_summary().get('lof', 0)
                print(f"LOF outliers detected: {lof_count}")
            except Exception as e:
                print(f"LOF detection error: {e}")
            
            # 5. Mahalanobis Distance (if PCA components exist)
            print("\n--- 11.5: Mahalanobis Distance Detection ---")
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
            print("\n--- 11.6: Computing Combined Outlier Score ---")
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
            
            # 7. Filter False Detections
            print("\n--- 11.7: Filtering False Detections ---")
            try:
                # Validate outliers (require at least 2 methods to agree)
                listings = detector.validate_outliers(listings, min_agreement=2, use_multivariate=True)
                
                # Filter false detections using agreement method
                listings = detector.filter_false_detections(
                    listings, 
                    method='agreement', 
                    min_agreement=2
                )
                
                # Get false detection report
                false_detection_report = detector.get_false_detection_report(listings)
                
                if false_detection_report:
                    print("False Detection Report:")
                    if "total_detected" in false_detection_report:
                        print(f"  Total detected outliers: {false_detection_report['total_detected']}")
                    if "false_positives" in false_detection_report:
                        print(f"  False positives: {false_detection_report['false_positives']}")
                        print(f"  False positive rate: {false_detection_report['false_positive_rate']:.1f}%")
                    if "confirmed_outliers" in false_detection_report:
                        print(f"  Confirmed outliers: {false_detection_report['confirmed_outliers']}")
                        print(f"  Filtered out: {false_detection_report.get('filtered_out', 0)}")
                    
                    # Show which methods contributed to false positives
                    if "method_contributions" in false_detection_report and false_detection_report["method_contributions"]:
                        print("\n  Method contributions to false positives:")
                        for method, count in sorted(false_detection_report["method_contributions"].items(), 
                                                   key=lambda x: x[1], reverse=True)[:5]:
                            method_name = method.replace("outlier_", "").replace("iqr_", "IQR-").replace("zscore_", "Z-Score-")
                            print(f"    {method_name}: {count}")
                
                # Show validation statistics
                if "outlier_validated" in listings.columns:
                    validated_count = listings["outlier_validated"].sum()
                    total_with_flags = (listings.get("outlier_score", pd.Series([0] * len(listings))) > 0).sum()
                    print(f"\nValidation Results:")
                    print(f"  Validated outliers (≥2 methods agree): {validated_count}")
                    print(f"  Records requiring validation: {total_with_flags}")
                    if total_with_flags > 0:
                        validation_rate = (validated_count / total_with_flags) * 100
                        print(f"  Validation rate: {validation_rate:.1f}%")
                
            except Exception as e:
                print(f"False detection filtering error: {e}")
                import traceback
                traceback.print_exc()
            
            print("\n--- Outlier Detection Summary ---")
            final_summary = detector.get_summary()
            print(f"Total outlier detection methods executed: {len([k for k in final_summary.keys() if not k.startswith('iqr_') and not k.startswith('zscore_')])}")
            if "validated_outliers" in final_summary:
                print(f"Validated outliers: {final_summary['validated_outliers']}")
            if "false_positives" in final_summary:
                print(f"False positives filtered: {final_summary['false_positives']}")
            print(f"Dataset shape after outlier detection: {listings.shape}")
            
        else:
            print("Skipping outlier detection: listings is None")
        print(f"Outlier detection took {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        print("\n=== STEP 12: SAVE PROCESSED DATA ===")

        if listings is not None:
            final_df = listings.copy()

            if 'listings_pca' in locals() and not listings_pca.empty:
                pca_cols = [c for c in listings_pca.columns if c not in final_df.columns]
                final_df = pd.concat([final_df, listings_pca[pca_cols]], axis=1)

            if 'listings_selected' in locals() and not listings_selected.empty:
                selected_cols = [c for c in listings_selected.columns if c not in final_df.columns]
                final_df = pd.concat([final_df, listings_selected[selected_cols]], axis=1)
            
            outlier_cols = [col for col in final_df.columns if col.startswith('outlier_')]
            if outlier_cols:
                print(f"Outlier detection columns included: {len(outlier_cols)}")
                print(f"  - Outlier flags: {len([c for c in outlier_cols if not c.endswith('_score') and c != 'outlier_type'])}")
                print(f"  - Outlier score: {'outlier_score' in outlier_cols}")
                print(f"  - Outlier type: {'outlier_type' in outlier_cols}")

            integrated_path = os.path.join(processed_dir, "integrated_processed_listings.csv")
            final_df.to_csv(integrated_path, index=False)
            print(f"All processed data saved in one CSV: {integrated_path}")
            print(f"Final dataset shape: {final_df.shape}")
            print(f"Final dataset columns: {len(final_df.columns)} total columns")
            
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
