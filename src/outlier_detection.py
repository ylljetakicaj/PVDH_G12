import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore
from numpy.linalg import inv

class OutlierDetector:

    def __init__(self):
        self.summary = {}

    # ------------------------------------------
    # 1. IQR Outliers (only meaningful columns)
    # ------------------------------------------
    def detect_iqr(self, df, columns):
        for col in columns:
            if col not in df.columns:
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            df[f"outlier_iqr_{col}"] = (df[col] < lower) | (df[col] > upper)

            self.summary[f"iqr_{col}"] = df[f"outlier_iqr_{col}"].sum()

        return df

    # ------------------------------------------
    # 2. Z-Score Outliers
    # ------------------------------------------
    def detect_zscore(self, df, columns, threshold=3):
        for col in columns:
            if col not in df.columns:
                continue

            df[f"outlier_zscore_{col}"] = (
                abs(zscore(df[col].fillna(df[col].mean()))) > threshold
            )

            self.summary[f"zscore_{col}"] = df[f"outlier_zscore_{col}"].sum()

        return df

    # ------------------------------------------
    # 3. Isolation Forest
    # ------------------------------------------
    def detect_isolation_forest(self, df, features, contamination=0.035):

        X = df[features].fillna(0)

        iso = IsolationForest(
            n_estimators=250,
            contamination=contamination,
            random_state=42
        )

        pred = iso.fit_predict(X)
        df["outlier_iforest"] = (pred == -1)

        self.summary["isolation_forest"] = df["outlier_iforest"].sum()

        return df

    # ------------------------------------------
    # 4. Local Outlier Factor (LOF)
    # ------------------------------------------
    def detect_lof(self, df, features, contamination=0.035):

        X = df[features].fillna(0)

        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination
        )

        pred = lof.fit_predict(X)
        df["outlier_lof"] = (pred == -1)

        self.summary["lof"] = df["outlier_lof"].sum()

        return df

    # ------------------------------------------
    # 5. Mahalanobis Distance (PCA-based)
    # ------------------------------------------
    def detect_mahalanobis(self, df, pca_components, threshold=3.5):

        X = df[pca_components].fillna(0).values

        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        inv_cov = inv(cov)

        diff = X - mean
        md = np.sqrt(np.sum(diff.dot(inv_cov) * diff, axis=1))

        df["outlier_mahalanobis"] = (md > threshold)

        self.summary["mahalanobis"] = df["outlier_mahalanobis"].sum()

        return df

    # ------------------------------------------
    # 6. Combined Score
    # ------------------------------------------
    def compute_outlier_score(self, df):
        outlier_cols = [c for c in df.columns if c.startswith("outlier_")]
        df["outlier_score"] = df[outlier_cols].sum(axis=1)
        return df

    # ------------------------------------------
    # 7. Outlier Type Mapping
    # ------------------------------------------
    def map_outlier_type(self, score):
        if score == 0:
            return "normal"
        elif score == 1:
            return "mild"
        elif score == 2:
            return "strong"
        else:
            return "extreme"

    # ------------------------------------------
    # 8. Validate Outliers (Remove False Positives)
    # ------------------------------------------
    def validate_outliers(self, df, min_agreement=2, use_multivariate=True):
        """
        Validate outliers by requiring agreement from multiple methods.
        This helps remove false positives.
        
        Args:
            df: DataFrame with outlier flags
            min_agreement: Minimum number of methods that must agree (default: 2)
            use_multivariate: Whether to include multivariate methods in validation
        
        Returns:
            DataFrame with validated outlier flags
        """
        # Get all outlier flag columns (excluding score and type)
        outlier_flags = [c for c in df.columns if c.startswith("outlier_") 
                        and not c.endswith("_score") and c != "outlier_type"]
        
        # Separate univariate and multivariate methods
        univariate_flags = [c for c in outlier_flags if "iqr_" in c or "zscore_" in c]
        multivariate_flags = [c for c in outlier_flags if c in ["outlier_iforest", "outlier_lof", "outlier_mahalanobis"]]
        
        # Count agreements for each record
        if use_multivariate:
            all_flags = univariate_flags + multivariate_flags
        else:
            all_flags = univariate_flags
        
        if len(all_flags) == 0:
            print("Warning: No outlier flags found for validation")
            return df
        
        # Count how many methods flagged each record as outlier
        df["outlier_agreement_count"] = df[all_flags].sum(axis=1)
        
        # Create validated outlier flag (only if min_agreement methods agree)
        df["outlier_validated"] = df["outlier_agreement_count"] >= min_agreement
        
        # Count false positives (detected but not validated)
        if "outlier_score" in df.columns:
            detected_outliers = (df["outlier_score"] > 0)
            false_positives = detected_outliers & ~df["outlier_validated"]
            df["outlier_false_positive"] = false_positives
            
            self.summary["validated_outliers"] = df["outlier_validated"].sum()
            self.summary["false_positives"] = false_positives.sum()
            self.summary["false_positive_rate"] = false_positives.sum() / max(detected_outliers.sum(), 1) * 100
        
        return df
    
    # ------------------------------------------
    # 9. Filter False Detections
    # ------------------------------------------
    def filter_false_detections(self, df, method='agreement', min_agreement=2, 
                                confidence_threshold=0.5, use_score=True):
        """
        Filter out false detections using various strategies.
        
        Args:
            df: DataFrame with outlier flags
            method: 'agreement' (require multiple methods) or 'confidence' (use score threshold)
            min_agreement: Minimum methods that must agree (for 'agreement' method)
            confidence_threshold: Minimum score threshold (for 'confidence' method)
            use_score: Whether to use outlier_score for confidence
        
        Returns:
            DataFrame with filtered outlier flags
        """
        if method == 'agreement':
            # Use agreement-based filtering
            df = self.validate_outliers(df, min_agreement=min_agreement)
            
            # Mark false positives
            if "outlier_validated" in df.columns:
                # Only keep validated outliers
                df["outlier_confirmed"] = df["outlier_validated"]
            else:
                df["outlier_confirmed"] = False
                
        elif method == 'confidence':
            # Use score-based filtering
            if "outlier_score" in df.columns:
                # Calculate confidence as score / max_possible_score
                max_possible_score = len([c for c in df.columns if c.startswith("outlier_") 
                                         and not c.endswith("_score") and c != "outlier_type"])
                if max_possible_score > 0:
                    df["outlier_confidence"] = df["outlier_score"] / max_possible_score
                    df["outlier_confirmed"] = (df["outlier_confidence"] >= confidence_threshold) & (df["outlier_score"] > 0)
                else:
                    df["outlier_confirmed"] = False
            else:
                print("Warning: outlier_score not found, cannot use confidence method")
                df["outlier_confirmed"] = False
        else:
            print(f"Unknown method: {method}, using agreement method")
            df = self.filter_false_detections(df, method='agreement', min_agreement=min_agreement)
        
        # Count filtered results
        if "outlier_confirmed" in df.columns:
            self.summary["confirmed_outliers"] = df["outlier_confirmed"].sum()
            if "outlier_score" in df.columns:
                original_outliers = (df["outlier_score"] > 0).sum()
                filtered_out = original_outliers - df["outlier_confirmed"].sum()
                self.summary["filtered_out"] = filtered_out
                self.summary["filter_rate"] = filtered_out / max(original_outliers, 1) * 100
        
        return df
    
    # ------------------------------------------
    # 10. Remove Outliers from Dataset
    # ------------------------------------------
    def remove_outliers(self, df, method='agreement', min_agreement=2, 
                       remove_extreme_only=False, keep_flags=True):
        """
        Remove outliers from the dataset.
        
        Args:
            df: DataFrame with outlier flags
            method: 'agreement' or 'confidence'
            min_agreement: Minimum methods that must agree
            remove_extreme_only: If True, only remove extreme outliers (score >= 3)
            keep_flags: If True, keep outlier flag columns in the result
        
        Returns:
            DataFrame with outliers removed
        """
        original_count = len(df)
        
        # First filter false detections
        df = self.filter_false_detections(df, method=method, min_agreement=min_agreement)
        
        # Determine which records to remove
        if remove_extreme_only:
            # Only remove extreme outliers
            if "outlier_type" in df.columns:
                to_remove = df["outlier_type"] == "extreme"
            elif "outlier_score" in df.columns:
                to_remove = df["outlier_score"] >= 3
            else:
                to_remove = df.get("outlier_confirmed", pd.Series([False] * len(df)))
        else:
            # Remove all confirmed outliers
            to_remove = df.get("outlier_confirmed", pd.Series([False] * len(df)))
        
        # Count what will be removed
        removed_count = to_remove.sum()
        self.summary["outliers_removed"] = removed_count
        self.summary["removal_rate"] = removed_count / original_count * 100
        
        # Remove outliers
        df_cleaned = df[~to_remove].copy()
        
        # Optionally remove outlier flag columns
        if not keep_flags:
            outlier_cols = [c for c in df_cleaned.columns if c.startswith("outlier_")]
            df_cleaned = df_cleaned.drop(columns=outlier_cols)
        
        return df_cleaned
    
    # ------------------------------------------
    # 11. Get False Detection Report
    # ------------------------------------------
    def get_false_detection_report(self, df):
        """
        Generate a report on false detections.
        
        Args:
            df: DataFrame with outlier flags and validation results
        
        Returns:
            Dictionary with false detection statistics
        """
        report = {}
        
        if "outlier_false_positive" in df.columns:
            false_positives = df["outlier_false_positive"].sum()
            total_detected = (df.get("outlier_score", pd.Series([0] * len(df))) > 0).sum()
            
            report["total_detected"] = total_detected
            report["false_positives"] = false_positives
            report["true_positives"] = total_detected - false_positives
            report["false_positive_rate"] = (false_positives / max(total_detected, 1)) * 100
            
            # Analyze which methods contributed to false positives
            if false_positives > 0:
                fp_records = df[df["outlier_false_positive"]]
                outlier_flags = [c for c in df.columns if c.startswith("outlier_") 
                               and not c.endswith("_score") and c != "outlier_type" 
                               and c not in ["outlier_validated", "outlier_confirmed", "outlier_false_positive"]]
                
                method_contributions = {}
                for flag in outlier_flags:
                    method_contributions[flag] = fp_records[flag].sum()
                
                report["method_contributions"] = method_contributions
        
        if "outlier_confirmed" in df.columns:
            report["confirmed_outliers"] = df["outlier_confirmed"].sum()
            report["filtered_out"] = report.get("total_detected", 0) - report["confirmed_outliers"]
        
        return report
        
    # ------------------------------------------
    # 12. Categorical Outliers (Rare Labels)
    # ------------------------------------------
    def detect_rare_categories(self, df, column, min_freq=0.01):
        """
        Flag rare categories in a categorical column.

        Args:
            df: DataFrame
            column: categorical column name
            min_freq: minimum frequency threshold (default 1%)
        """
        if column not in df:
            return df

        freq = df[column].value_counts(normalize=True)
        rare_values = freq[freq < min_freq].index

    # ------------------------------------------
    # Summary
    # ------------------------------------------
    def get_summary(self):
        return self.summary
