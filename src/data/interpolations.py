from scipy.interpolate import PchipInterpolator, Akima1DInterpolator, interp1d
import numpy as np


class InterpolationTransformer:
    def __init__(self, method: str):
        self.method = method
        self.interpolator = None

    def fit(self, X, y=None):
        self.interpolator = None
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        for col in range(X.shape[1]):
            mask = ~np.isnan(X[:, col])
            if np.sum(mask) < 2:
                continue

            x_known = np.arange(len(X[:, col]))[mask]
            y_known = X[:, col][mask]

            if self.method in ["linear", "quadratic", "cubic", "nearest"]:
                interpolator = interp1d(
                    x_known, y_known, kind=self.method, fill_value="extrapolate"
                )
            elif self.method == "pchip":
                interpolator = PchipInterpolator(x_known, y_known)
            elif self.method == "akima":
                interpolator = Akima1DInterpolator(x_known, y_known)
            else:
                raise ValueError(f"Unknown interpolation method: {self.method}")

            X[:, col][~mask] = interpolator(np.arange(len(X[:, col]))[~mask])

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
