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

   