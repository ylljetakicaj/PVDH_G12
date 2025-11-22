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
    # Summary
    # ------------------------------------------
    def get_summary(self):
        return self.summary
