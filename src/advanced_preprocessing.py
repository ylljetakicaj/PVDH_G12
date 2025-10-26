import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AdvancedPreprocessor:
    """
    Advanced preprocessing class for dimension reduction, feature selection,
    property creation, discretization, and transformation.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.pca_models = {}
        self.feature_selectors = {}
        self.discretization_bins = {}
        
    def dimension_reduction(self, df, target_col=None, method='pca', n_components=0.95, 
                          feature_types='numeric'):
        """
        Perform dimension reduction using various techniques.
        
        Args:
            df: DataFrame to process
            target_col: Target column for supervised feature selection
            method: 'pca', 'variance_threshold', 'univariate', 'mutual_info'
            n_components: Number of components (PCA) or threshold (variance)
            feature_types: 'numeric', 'categorical', 'all'
        
        Why this approach:
        - PCA: Reduces dimensionality while preserving variance, good for visualization
        - Variance threshold: Removes low-variance features that don't contribute much
        - Univariate: Selects features based on statistical tests with target
        - Mutual info: Captures non-linear relationships with target
        """
        print(f"\n=== DIMENSION REDUCTION: {method.upper()} ===")
        
        # Select appropriate columns based on feature_types
        if feature_types == 'numeric':
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        elif feature_types == 'categorical':
            feature_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        else:  # 'all'
            feature_cols = df.columns.tolist()
            
        # Remove target column if specified
        if target_col and target_col in feature_cols:
            feature_cols.remove(target_col)
            
        print(f"Working with {len(feature_cols)} features of type '{feature_types}'")
        
        if len(feature_cols) == 0:
            print("No suitable features found for dimension reduction")
            return df
            
        # Handle missing values for numeric features
        if feature_types in ['numeric', 'all']:
            numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
        result_df = df.copy()
        
        if method == 'pca':
            return self._apply_pca(result_df, feature_cols, n_components)
        elif method == 'variance_threshold':
            return self._apply_variance_threshold(result_df, feature_cols, n_components)
        elif method == 'univariate' and target_col:
            return self._apply_univariate_selection(result_df, feature_cols, target_col, n_components)
        elif method == 'mutual_info' and target_col:
            return self._apply_mutual_info_selection(result_df, feature_cols, target_col, n_components)
        else:
            print(f"Method '{method}' not supported or target column missing for supervised methods")
            return result_df
    
    def _apply_pca(self, df, feature_cols, n_components):
        """Apply PCA dimension reduction."""
        # Only use numeric columns for PCA
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            print("Need at least 2 numeric features for PCA")
            return df
            
        X = df[numeric_cols].values
        
        # Standardize features before PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        if isinstance(n_components, float) and n_components < 1:
            # Keep components that explain n_components variance
            pca = PCA(n_components=n_components)
        else:
            # Keep specific number of components
            pca = PCA(n_components=min(int(n_components), len(numeric_cols)))
            
        X_pca = pca.fit_transform(X_scaled)
        
        # Create new DataFrame with PCA components
        pca_cols = [f'PC_{i+1}' for i in range(X_pca.shape[1])]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
        
        # Keep non-numeric columns and add PCA components
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
        result_df = pd.concat([df[non_numeric_cols], pca_df], axis=1)
        
        # Store models for later use
        self.scalers['pca_scaler'] = scaler
        self.pca_models['pca'] = pca
        
        print(f"PCA reduced {len(numeric_cols)} features to {X_pca.shape[1]} components")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}")  # Show first 5
        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return result_df
    
    def _apply_variance_threshold(self, df, feature_cols, threshold):
        """Remove low-variance features."""
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            print("No numeric features for variance threshold")
            return df
            
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(df[numeric_cols])
        
        selected_features = [col for col, selected in zip(numeric_cols, selector.get_support()) if selected]
        
        # Keep selected numeric features and all non-numeric features
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
        result_df = df[non_numeric_cols + selected_features].copy()
        
        self.feature_selectors['variance_threshold'] = selector
        
        print(f"Variance threshold removed {len(numeric_cols) - len(selected_features)} features")
        print(f"Kept {len(selected_features)} features: {selected_features[:10]}...")  # Show first 10
        
        return result_df
    
    def _apply_univariate_selection(self, df, feature_cols, target_col, k):
        """Select k best features using univariate statistical tests."""
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0 or target_col not in df.columns:
            print("No numeric features or target column not found")
            return df
            
        X = df[numeric_cols]
        y = df[target_col]
        
        # Remove rows where target is NaN
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            print("No valid samples after removing NaN targets")
            return df
            
        k_best = min(int(k), len(numeric_cols)) if isinstance(k, (int, float)) else len(numeric_cols)
        selector = SelectKBest(score_func=f_regression, k=k_best)
        
        try:
            X_selected = selector.fit_transform(X, y)
            selected_features = [col for col, selected in zip(numeric_cols, selector.get_support()) if selected]
            
            # Keep selected features and all non-numeric features
            non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
            result_df = df[non_numeric_cols + selected_features].copy()
            
            self.feature_selectors['univariate'] = selector
            
            print(f"Univariate selection kept {len(selected_features)} best features")
            print(f"Selected features: {selected_features}")
            
            return result_df
        except Exception as e:
            print(f"Univariate selection failed: {e}")
            return df
    
    def _apply_mutual_info_selection(self, df, feature_cols, target_col, k):
        """Select features using mutual information."""
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0 or target_col not in df.columns:
            print("No numeric features or target column not found")
            return df
            
        X = df[numeric_cols]
        y = df[target_col]
        
        # Remove rows where target is NaN
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            print("No valid samples after removing NaN targets")
            return df
            
        k_best = min(int(k), len(numeric_cols)) if isinstance(k, (int, float)) else len(numeric_cols)
        selector = SelectKBest(score_func=mutual_info_regression, k=k_best)
        
        try:
            X_selected = selector.fit_transform(X, y)
            selected_features = [col for col, selected in zip(numeric_cols, selector.get_support()) if selected]
            
            # Keep selected features and all non-numeric features
            non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
            result_df = df[non_numeric_cols + selected_features].copy()
            
            self.feature_selectors['mutual_info'] = selector
            
            print(f"Mutual info selection kept {len(selected_features)} best features")
            print(f"Selected features: {selected_features}")
            
            return result_df
        except Exception as e:
            print(f"Mutual info selection failed: {e}")
            return df

    def select_property_subsets(self, df, subset_type='high_value', custom_conditions=None):
        """
        Select subsets of properties based on various criteria.
        
        Args:
            df: DataFrame to filter
            subset_type: 'high_value', 'popular', 'superhosts', 'recent', 'custom'
            custom_conditions: Dict of column:condition pairs for custom filtering
            
        Why this approach:
        - High value: Focus on premium properties for luxury market analysis
        - Popular: Properties with many reviews for understanding guest preferences
        - Superhosts: Quality properties for benchmarking standards
        - Recent: New properties for trend analysis
        - Custom: Flexible filtering for specific research questions
        """
        print(f"\n=== PROPERTY SUBSET SELECTION: {subset_type.upper()} ===")
        
        original_count = len(df)
        
        if subset_type == 'high_value':
            # Properties in top 25% of price range
            if 'price' in df.columns:
                price_threshold = df['price'].quantile(0.75)
                subset_df = df[df['price'] >= price_threshold].copy()
                print(f"High-value properties: price >= ${price_threshold:.2f}")
            else:
                print("Price column not found, returning original dataset")
                return df
                
        elif subset_type == 'popular':
            # Properties with above-average number of reviews
            if 'number_of_reviews' in df.columns:
                review_threshold = df['number_of_reviews'].median()
                subset_df = df[df['number_of_reviews'] >= review_threshold].copy()
                print(f"Popular properties: reviews >= {review_threshold}")
            else:
                print("Number of reviews column not found, returning original dataset")
                return df
                
        elif subset_type == 'superhosts':
            # Properties hosted by superhosts
            if 'host_is_superhost' in df.columns:
                subset_df = df[df['host_is_superhost'] == True].copy()
                print("Superhost properties selected")
            else:
                print("Superhost column not found, returning original dataset")
                return df
                
        elif subset_type == 'recent':
            # Properties with recent reviews (last 2 years)
            if 'last_review' in df.columns:
                df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
                cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=2)
                subset_df = df[df['last_review'] >= cutoff_date].copy()
                print(f"Recent properties: last review >= {cutoff_date.date()}")
            else:
                print("Last review column not found, returning original dataset")
                return df
                
        elif subset_type == 'custom' and custom_conditions:
            # Apply custom filtering conditions
            subset_df = df.copy()
            for column, condition in custom_conditions.items():
                if column in df.columns:
                    if isinstance(condition, dict):
                        # Handle range conditions
                        if 'min' in condition:
                            subset_df = subset_df[subset_df[column] >= condition['min']]
                        if 'max' in condition:
                            subset_df = subset_df[subset_df[column] <= condition['max']]
                        if 'values' in condition:
                            subset_df = subset_df[subset_df[column].isin(condition['values'])]
                    else:
                        # Handle simple equality condition
                        subset_df = subset_df[subset_df[column] == condition]
                    print(f"Applied condition on {column}: {condition}")
                else:
                    print(f"Column {column} not found, skipping condition")
        else:
            print("Invalid subset type or missing custom conditions")
            return df
            
        final_count = len(subset_df)
        print(f"Subset selection: {original_count} -> {final_count} properties ({final_count/original_count*100:.1f}%)")
        
        return subset_df
        """Generate a summary of all preprocessing steps applied."""
        print(f"\n=== PREPROCESSING SUMMARY ===")
        print(f"Original dataset: {original_df.shape[0]} rows, {original_df.shape[1]} columns")
        print(f"Processed dataset: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
        print(f"Features added: {processed_df.shape[1] - original_df.shape[1]}")
        
        # Show new columns
        new_cols = set(processed_df.columns) - set(original_df.columns)
        if new_cols:
            print(f"\nNew features created ({len(new_cols)}):")
            for col in sorted(new_cols)[:20]:  # Show first 20
                print(f"  - {col}")
            if len(new_cols) > 20:
                print(f"  ... and {len(new_cols) - 20} more")
                
        # Show memory usage
        original_memory = original_df.memory_usage(deep=True).sum() / 1024**2
        processed_memory = processed_df.memory_usage(deep=True).sum() / 1024**2
        print(f"\nMemory usage: {original_memory:.1f} MB -> {processed_memory:.1f} MB")
        
        return {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'features_added': processed_df.shape[1] - original_df.shape[1],
            'new_columns': list(new_cols),
            'memory_change': processed_memory - original_memory
        }
