import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin


# source: https://www.kaggle.com/code/hassanrady/1st-place-cmi-model-v4-1-1-reduced
def bin_data(train, test, columns, n_bins=10):
    # Combine train and test for consistent bin edges
    combined = pd.concat([train, test], axis=0)

    bin_edges = {}
    for col in columns:
        # Compute quantile bin edges
        edges = pd.qcut(
            combined[col], n_bins, retbins=True, labels=range(n_bins), duplicates="drop"
        )[1]
        bin_edges[col] = edges

    # Apply the same bin edges to both train and test
    for col, edges in bin_edges.items():
        train[col] = pd.cut(
            train[col], bins=edges, labels=range(len(edges) - 1), include_lowest=True
        ).astype(float)
        test[col] = pd.cut(
            test[col], bins=edges, labels=range(len(edges) - 1), include_lowest=True
        ).astype(float)

    return train, test


class Impute_With_Model(BaseEstimator, TransformerMixin):
    def __init__(self, model, na_frac=0.5, min_samples=0):
        self.model = model
        self.na_frac = na_frac
        self.min_samples = min_samples
        self.model_dict = {}
        self.mean_dict = {}
        self.features = None

    def find_features(self, data, feature_idx, tmp_indices):
        """Find valid features based on missingness constraints."""
        missing_rows = np.isnan(data[:, feature_idx])
        na_fraction = np.mean(np.isnan(data[missing_rows][:, tmp_indices]), axis=0)
        valid_indices = [
            idx for idx, frac in zip(tmp_indices, na_fraction) if frac <= self.na_frac
        ]
        return valid_indices

    def fit(self, X, y=None):
        """Fit models to predict missing values."""
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X must be a numpy ndarray.")

        self.features = list(range(X.shape[1]))
        for idx in self.features:
            self.mean_dict[idx] = np.nanmean(X[:, idx])

        for idx in tqdm(self.features):
            if np.isnan(X[:, idx]).sum() > 0:
                model_clone = clone(self.model)
                not_na_mask = ~np.isnan(X[:, idx])
                X_feature = X[not_na_mask, :]
                tmp_indices = [i for i in self.features if i != idx]
                tmp_indices = self.find_features(X, idx, tmp_indices)

                if len(tmp_indices) >= 1 and X_feature.shape[0] > self.min_samples:
                    X_filled = X_feature[:, tmp_indices].copy()
                    for j, col_idx in enumerate(tmp_indices):
                        col_mean = self.mean_dict[col_idx]
                        X_filled[:, j] = np.where(np.isnan(X_filled[:, j]), col_mean, X_filled[:, j])
                    model_clone.fit(X_filled, X[not_na_mask, idx])
                    self.model_dict[idx] = (model_clone, tmp_indices)
                else:
                    self.model_dict[idx] = ("mean", self.mean_dict[idx])

        return self

    def transform(self, X):
        """Impute missing values in the data."""
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X must be a numpy ndarray.")

        imputed_data = X.copy()
        for idx, model in self.model_dict.items():
            missing_rows = np.isnan(imputed_data[:, idx])
            if np.any(missing_rows):
                if model[0] == "mean":
                    imputed_data[missing_rows, idx] = model[1]
                else:
                    tmp_indices = model[1]
                    X_missing = imputed_data[missing_rows][:, tmp_indices].copy()
                    for j, col_idx in enumerate(tmp_indices):
                        col_mean = self.mean_dict[col_idx]
                        X_missing[:, j] = np.where(np.isnan(X_missing[:, j]), col_mean, X_missing[:, j])
                    imputed_data[missing_rows, idx] = model[0].predict(X_missing)

        return imputed_data
