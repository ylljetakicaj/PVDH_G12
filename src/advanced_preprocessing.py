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

    def create_derived_properties(self, df):
        """
        Create new properties from existing features.
        
        Why these derived features:
        - Price per person: Normalizes price by capacity for fair comparison
        - Review density: Indicates how actively reviewed a property is
        - Host experience: Captures host tenure which affects service quality
        - Availability ratio: Shows how often property is available
        - Amenity count: Quantifies property features
        - Location desirability: Combines location-based metrics
        """
        print(f"\n=== PROPERTY CREATION ===")
        
        result_df = df.copy()
        created_features = []
        
        # 1. Price per person
        if 'price' in df.columns and 'accommodates' in df.columns:
            result_df['price_per_person'] = result_df['price'] / result_df['accommodates'].replace(0, 1)
            created_features.append('price_per_person')
            
        # 2. Review density (reviews per month since first review)
        if 'number_of_reviews' in df.columns and 'first_review' in df.columns:
            result_df['first_review'] = pd.to_datetime(result_df['first_review'], errors='coerce')
            current_date = pd.Timestamp.now()
            months_active = (current_date - result_df['first_review']).dt.days / 30.44
            months_active = months_active.replace(0, 1)  # Avoid division by zero
            result_df['review_density'] = result_df['number_of_reviews'] / months_active
            result_df['review_density'] = result_df['review_density'].fillna(0)
            created_features.append('review_density')
            
        # 3. Host experience (years since host started)
        if 'host_since' in df.columns:
            result_df['host_since'] = pd.to_datetime(result_df['host_since'], errors='coerce')
            current_date = pd.Timestamp.now()
            result_df['host_experience_years'] = (current_date - result_df['host_since']).dt.days / 365.25
            result_df['host_experience_years'] = result_df['host_experience_years'].fillna(0)
            created_features.append('host_experience_years')
            
        # 4. Availability ratio
        if 'availability_365' in df.columns:
            result_df['availability_ratio'] = result_df['availability_365'] / 365
            created_features.append('availability_ratio')
            
        # 5. Amenity count
        if 'amenities' in df.columns:
            result_df['amenity_count'] = result_df['amenities'].apply(
                lambda x: len(eval(x)) if isinstance(x, str) and x.startswith('[') else 0
            )
            created_features.append('amenity_count')
            
        # 6. Bathroom-bedroom ratio
        if 'bathrooms' in df.columns and 'bedrooms' in df.columns:
            result_df['bathroom_bedroom_ratio'] = result_df['bathrooms'] / result_df['bedrooms'].replace(0, 1)
            created_features.append('bathroom_bedroom_ratio')
            
        # 7. Review score average
        review_score_cols = [col for col in df.columns if col.startswith('review_scores_')]
        if len(review_score_cols) > 0:
            result_df['avg_review_score'] = result_df[review_score_cols].mean(axis=1)
            created_features.append('avg_review_score')
            
        # 8. Location desirability score (based on review_scores_location and price)
        if 'review_scores_location' in df.columns and 'price' in df.columns:
            # Normalize both metrics to 0-1 scale
            location_norm = (result_df['review_scores_location'] - result_df['review_scores_location'].min()) / \
                           (result_df['review_scores_location'].max() - result_df['review_scores_location'].min())
            price_norm = (result_df['price'] - result_df['price'].min()) / \
                        (result_df['price'].max() - result_df['price'].min())
            
            # Combine location score (positive) with price (negative, as higher price might indicate desirability)
            result_df['location_desirability'] = (location_norm * 0.7) + (price_norm * 0.3)
            result_df['location_desirability'] = result_df['location_desirability'].fillna(0)
            created_features.append('location_desirability')
            
        # 9. Is luxury property (based on price and amenities)
        if 'price' in df.columns and 'amenity_count' in result_df.columns:
            price_threshold = result_df['price'].quantile(0.8)
            amenity_threshold = result_df['amenity_count'].quantile(0.8)
            result_df['is_luxury'] = ((result_df['price'] >= price_threshold) & 
                                     (result_df['amenity_count'] >= amenity_threshold)).astype(int)
            created_features.append('is_luxury')
            
        print(f"Created {len(created_features)} derived features:")
        for feature in created_features:
            print(f"  - {feature}")
            
        return result_df
        """
        Apply discretization and binarization to continuous variables.
        
        Args:
            df: DataFrame to process
            discretization_config: Dict specifying which columns to discretize and how
            
        Why discretization:
        - Reduces noise in continuous variables
        - Makes patterns more interpretable
        - Can improve performance of some algorithms
        - Useful for creating categorical features from numeric ones
        
        Why binarization:
        - Simplifies complex relationships
        - Useful for creating yes/no features
        - Can highlight important thresholds
        """
    def discretize_and_binarize(self, df, discretization_config=None):
        print(f"\n=== DISCRETIZATION AND BINARIZATION ===")
        result_df = df.copy()
        print(f"Input shape: {result_df.shape}")
        print(f"Input columns: {list(result_df.columns)}")
        
        # Default discretization configuration
        if discretization_config is None:
            discretization_config = {
                'price': {'method': 'quantile', 'bins': 5, 'labels': ['Very Low', 'Low', 'Medium', 'High', 'Very High']},
                'number_of_reviews_x': {'method': 'equal_width', 'bins': 4, 'labels': ['Few', 'Some', 'Many', 'Lots']},
                'availability_365': {'method': 'equal_width', 'bins': 3, 'labels': ['Low', 'Medium', 'High']},
                'accommodates': {'method': 'custom', 'bins': [0, 2, 4, 6, float('inf')], 'labels': ['Small', 'Medium', 'Large', 'Extra Large']}
            }
            
        # Apply discretization
        for column, config in discretization_config.items():
            if column not in df.columns:
                print(f"Column {column} not found, skipping discretization")
                continue
            if df[column].dtype not in [np.number, 'float64', 'int64']:
                print(f"Column {column} is not numeric, skipping discretization")
                continue
            try:
                if config['method'] == 'quantile':
                    result_df[f'{column}_binned'] = pd.qcut(
                        df[column], 
                        q=config['bins'], 
                        labels=config.get('labels', None),
                        duplicates='drop'
                    )
                elif config['method'] == 'equal_width':
                    result_df[f'{column}_binned'] = pd.cut(
                        df[column], 
                        bins=config['bins'], 
                        labels=config.get('labels', None)
                    )
                elif config['method'] == 'custom':
                    result_df[f'{column}_binned'] = pd.cut(
                        df[column], 
                        bins=config['bins'], 
                        labels=config.get('labels', None)
                    )
                print(f"Discretized {column} into {config['bins']} bins using {config['method']} method")
                self.discretization_bins[column] = config
            except Exception as e:
                print(f"Failed to discretize {column}: {e}")
                
        # Apply binarization
        binarization_config = {
            'price': {'threshold': df['price'].median() if 'price' in df.columns else 100, 'name': 'is_expensive'},
            'number_of_reviews_x': {'threshold': 10, 'name': 'has_many_reviews'},
            'host_is_superhost': {'threshold': 0.5, 'name': 'is_superhost_binary'},
            'instant_bookable': {'threshold': 0.5, 'name': 'is_instant_bookable_binary'}
        }
        
        for column, config in binarization_config.items():
            if column not in df.columns:
                print(f"Column {column} not found, skipping binarization")
                continue
            try:
                if column in ['host_is_superhost', 'instant_bookable']:
                    print(f"Unique values in {column}: {df[column].unique()}")
                    # Convert to string to handle category type, then map to boolean
                    result_df[column] = df[column].astype(str).replace({'t': 'True', 'f': 'False', 'Unknown': 'False', 'nan': 'False'})
                    result_df[config['name']] = result_df[column].map({'True': 1, 'False': 0}).astype(int)
                else:
                    result_df[config['name']] = (df[column] >= config['threshold']).astype(int)
                print(f"Binarized {column} with threshold {config['threshold']} -> {config['name']}")
            except Exception as e:
                print(f"Failed to binarize {column}: {e}")
                
        print(f"Output shape: {result_df.shape}")
        print(f"New columns added: {[col for col in result_df.columns if col not in df.columns]}")
        return result_df

    def apply_transformations(self, df, transformation_config=None):
        """
        Apply various data transformations.
        
        Args:
            df: DataFrame to transform
            transformation_config: Dict specifying transformations to apply
            
        Why these transformations:
        - StandardScaler: Centers data around 0, unit variance - good for algorithms sensitive to scale
        - MinMaxScaler: Scales to [0,1] range - preserves relationships, good for neural networks
        - RobustScaler: Uses median/IQR - robust to outliers
        - Log transform: Reduces skewness in right-skewed distributions
        - Square root: Mild transformation for moderately skewed data
        - One-hot encoding: Converts categorical to binary features for ML algorithms
        """
        print(f"\n=== DATA TRANSFORMATIONS ===")
        
        result_df = df.copy()
        
        # Default transformation configuration
        if transformation_config is None:
            transformation_config = {
                'scaling': {
                    'method': 'standard',  # 'standard', 'minmax', 'robust'
                    'columns': 'numeric'   # 'numeric', 'all', or list of columns
                },
                'log_transform': ['price', 'number_of_reviews'],
                'sqrt_transform': ['availability_365'],
                'one_hot_encode': ['room_type', 'property_type', 'neighbourhood_cleansed']
            }
            
        # 1. Scaling transformations
        scaling_config = transformation_config.get('scaling', {})
        if scaling_config:
            method = scaling_config.get('method', 'standard')
            columns = scaling_config.get('columns', 'numeric')
            
            # Select columns to scale
            if columns == 'numeric':
                scale_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
            elif columns == 'all':
                scale_cols = result_df.columns.tolist()
            elif isinstance(columns, list):
                scale_cols = [col for col in columns if col in result_df.columns]
            else:
                scale_cols = []
                
            # Remove non-numeric columns
            scale_cols = [col for col in scale_cols if result_df[col].dtype in [np.number, 'float64', 'int64']]
            
            if scale_cols:
                # Choose scaler
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                elif method == 'robust':
                    scaler = RobustScaler()
                else:
                    print(f"Unknown scaling method: {method}")
                    scaler = StandardScaler()
                    
                # Apply scaling
                try:
                    scaled_data = scaler.fit_transform(result_df[scale_cols])
                    scaled_df = pd.DataFrame(scaled_data, columns=[f'{col}_scaled' for col in scale_cols], index=result_df.index)
                    result_df = pd.concat([result_df, scaled_df], axis=1)
                    
                    self.scalers[f'{method}_scaler'] = scaler
                    print(f"Applied {method} scaling to {len(scale_cols)} columns")
                    
                except Exception as e:
                    print(f"Scaling failed: {e}")
                    
        # 2. Log transformation
        log_cols = transformation_config.get('log_transform', [])
        for col in log_cols:
            if col in result_df.columns and result_df[col].dtype in [np.number, 'float64', 'int64']:
                try:
                    # Add small constant to handle zeros
                    result_df[f'{col}_log'] = np.log1p(result_df[col].clip(lower=0))
                    print(f"Applied log transformation to {col}")
                except Exception as e:
                    print(f"Log transformation failed for {col}: {e}")
                    
        # 3. Square root transformation
        sqrt_cols = transformation_config.get('sqrt_transform', [])
        for col in sqrt_cols:
            if col in result_df.columns and result_df[col].dtype in [np.number, 'float64', 'int64']:
                try:
                    result_df[f'{col}_sqrt'] = np.sqrt(result_df[col].clip(lower=0))
                    print(f"Applied square root transformation to {col}")
                except Exception as e:
                    print(f"Square root transformation failed for {col}: {e}")
                    
        # 4. One-hot encoding
        onehot_cols = transformation_config.get('one_hot_encode', [])
        for col in onehot_cols:
            if col in result_df.columns:
                try:
                    # Limit categories to top N to avoid too many columns
                    top_categories = result_df[col].value_counts().head(10).index.tolist()
                    result_df[f'{col}_limited'] = result_df[col].apply(
                        lambda x: x if x in top_categories else 'Other'
                    )
                    # Create dummy variables
                    dummies = pd.get_dummies(result_df[f'{col}_limited'], prefix=col)
                    result_df = pd.concat([result_df, dummies], axis=1)
                    
                    # Remove the temporary column
                    result_df.drop(f'{col}_limited', axis=1, inplace=True)
                    
                    print(f"One-hot encoded {col} into {len(dummies.columns)} binary features")
                    
                except Exception as e:
                    print(f"One-hot encoding failed for {col}: {e}")
                    
        return result_df

    def get_preprocessing_summary(self, original_df, processed_df):
        """Generate a summary of all preprocessing steps applied."""
        print(f"\n=== PREPROCESSING SUMMARY ===")
        print(f"Original dataset: {original_df.shape[0]} rows, {original_df.shape[1]} columns")
        print(f"Processed dataset: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
        print(f"Features added: {processed_df.shape[1] - original_df.shape[1]}")
        
        new_cols = set(processed_df.columns) - set(original_df.columns)
        if new_cols:
            print(f"\nNew features created ({len(new_cols)}):")
            for col in sorted(new_cols)[:20]:  
                print(f"  - {col}")
            if len(new_cols) > 20:
                print(f"  ... and {len(new_cols) - 20} more")
                
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