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
    