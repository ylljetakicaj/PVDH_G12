import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class EDAAnalyzer:

    def __init__(self, save_plots=True, output_dir="eda_plots"):
        self.summary = {}
        self.save_plots = save_plots
        self.output_dir = output_dir

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------
    # Helper to save plots
    # ------------------------------------------
    def _save_plot(self, filename):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    # ==========================================================
    # ⭐⭐⭐  STATISTIKA PËRMBLEDHËSE – UNIVARIATE ANALYSIS ⭐⭐⭐
    # ==========================================================

    # 1. Numerical Summary
    def numerical_summary(self, df, columns=None):
        if columns is None:
            columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        summary = df[columns].describe().T
        summary["missing"] = df[columns].isnull().sum()
        summary["skewness"] = df[columns].skew()
        summary["kurtosis"] = df[columns].kurtosis()

        self.summary["numerical_summary"] = summary
        return summary

    # 2. Categorical Summary
    def categorical_summary(self, df, columns=None):
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns
        
        summary = {col: df[col].value_counts(dropna=False) for col in columns}
        self.summary["categorical_summary"] = summary
        return summary

    # 7. Distribution Plots
    def distribution_plots(self, df, columns):
        for col in columns:
            plt.figure(figsize=(7, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")

            self._save_plot(f"distribution_{col}.png")

    # 8. Boxplots
    def boxplot(self, df, columns):
        for col in columns:
            plt.figure(figsize=(7, 4))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")

            self._save_plot(f"boxplot_{col}.png")

    # ==========================================================
    # ⭐⭐⭐  ANALIZA MULTIVARIANTE – MULTIVARIATE ANALYSIS ⭐⭐⭐
    # ==========================================================

    # 3. Correlation Matrix + Heatmap
    def correlation_matrix(self, df, columns=None, figsize=(12,10)):
        if columns is None:
            columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        corr = df[columns].corr()

        plt.figure(figsize=figsize)
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Correlation Heatmap")

        self._save_plot("correlation_heatmap.png")

        self.summary["correlation_matrix"] = corr
        return corr

    # 4. Pairplot
    def pairplot(self, df, columns, hue=None):
        plot = sns.pairplot(df[columns], hue=hue)
        filename = os.path.join(self.output_dir, "pairplot.png")
        plot.savefig(filename, dpi=300)
        plt.close()

    # 5. PCA Analysis
    def pca_analysis(self, df, columns, n_components=2):
        scaler = StandardScaler()
        X = scaler.fit_transform(df[columns].fillna(df[columns].mean()))

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X)

        explained_var = pca.explained_variance_ratio_
        self.summary["pca_explained_variance"] = explained_var

        pca_df = pd.DataFrame(
            components,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )

        return pca_df, explained_var

    # 6. Plot PCA Components
    def plot_pca(self, pca_df, labels=None):
        plt.figure(figsize=(10, 7))
        plt.scatter(pca_df["PC1"], pca_df["PC2"], c=labels)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA: PC1 vs PC2")

        self._save_plot("pca_plot.png")

    # 9. Grouped Summary
    def grouped_summary(self, df, group_col, target_col):
        grouped = df.groupby(group_col)[target_col].agg(
            ["mean", "median", "std", "count"]
        )

        self.summary[f"grouped_{target_col}_by_{group_col}"] = grouped
        return grouped

    # ==========================================================
    # Summary Getter
    # ==========================================================
    def get_summary(self):
        return self.summary
