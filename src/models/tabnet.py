from copy import deepcopy
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
import os
from pytorch_tabnet.callbacks import Callback


class TabNetWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.model = TabNetRegressor(**kwargs)
        self.kwargs = kwargs
        self.best_model_path = "best_tabnet_model.pt"

        if os.path.exists(self.best_model_path):
            self.model.load_model(self.best_model_path)
            os.remove(self.best_model_path)

    def fit(self, X, y):
        if hasattr(y, "values"):
            y = y.values

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(
            X_train=X_train,
            y_train=y_train.reshape(-1, 1),
            eval_set=[(X_valid, y_valid.reshape(-1, 1))],
            eval_name=["valid"],
            eval_metric=["mse"],
            max_epochs=200,
            patience=20,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
            callbacks=[
                TabNetPretrainedModelCheckpoint(
                    filepath=self.best_model_path,
                    monitor="valid_mse",
                    mode="min",
                    save_best_only=True,
                    verbose=1,
                )
            ],
        )

        if os.path.exists(self.best_model_path):
            self.model.load_model(self.best_model_path)
            os.remove(self.best_model_path)

        return self

    def predict(self, X):
        return self.model.predict(X).flatten()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __name__(self):
        return "TabNetWrapper"


class TabNetPretrainedModelCheckpoint(Callback):
    def __init__(
        self, filepath, monitor="val_loss", mode="min", save_best_only=True, verbose=1
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = float("inf") if mode == "min" else -float("inf")
        self.model_instance = None

    def on_train_begin(self, logs=None):
        self.model_instance = self.trainer  # Access trainer model

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if (self.mode == "min" and current < self.best) or (
            self.mode == "max" and current > self.best
        ):
            if self.verbose:
                print(
                    f"\nEpoch {epoch}: {self.monitor} improved from {self.best:.4f} to {current:.4f}"
                )
            self.best = current
            if self.save_best_only:
                self.model_instance.save_model(
                    self.filepath
                )  # Save the model checkpoint
