from functools import partial
from pathlib import Path
from typing import List, Any, Optional, Callable

import optuna
import pandas as pd
import torch
import torch.nn as nn
import lightning as pl
from lightning import Trainer

from config import AutoencoderHPOConfig, init_autoencoder_hpo_config
from data.data_manager import read_parquet, save_csv
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.core import run_hpo
from models.hpo_spaces import autoencoder_hpo_space
from utils.args import parse_config_path_args
from utils.registry import sklearn_scaler_registry, ActivationLayerType, ScalerType

from src.utils.registry import activation_layer_registry


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation_layer: ActivationLayerType,
        learning_rate: float,
    ):
        super(AutoEncoder, self).__init__()

        encoder_layers_list = []
        prev_dim = input_dim
        for neurons in hidden_dims:
            encoder_layers_list.append(nn.Linear(prev_dim, neurons))
            encoder_layers_list.append(activation_layer())
            prev_dim = neurons
        self.encoder = nn.Sequential(*encoder_layers_list)

        decoder_layers_list = []
        prev_dim = hidden_dims[-1]
        for neurons in reversed(hidden_dims):
            decoder_layers_list.append(nn.Linear(prev_dim, neurons))
            decoder_layers_list.append(activation_layer())
            prev_dim = neurons
        decoder_layers_list.append(nn.Linear(prev_dim, input_dim))
        decoder_layers_list.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers_list)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch: List, batch_idx) -> torch.Tensor:
        x = batch[0]
        y_hat = self(x)
        loss = self.criterion(y_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x = batch[0]
        y_hat = self(x)
        loss = self.criterion(y_hat, x)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def autoencode(
    df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    scaler: type[ScalerType],
    save_path: Optional[Path] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    data_scaled = scaler().fit_transform(df)

    data_tensor = torch.FloatTensor(data_scaled)
    input_dim = data_tensor.shape[1]

    data_loader = DataLoader(
        TensorDataset(data_tensor), batch_size=batch_size, shuffle=True, num_workers=4
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    autoencoder = AutoEncoder(input_dim, **kwargs).to(device)

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        enable_progress_bar=True,
    )
    trainer.fit(autoencoder, train_dataloaders=data_loader)

    with torch.no_grad():
        encoded_data = autoencoder.encoder(data_tensor).numpy()

    df_encoded = pd.DataFrame(
        encoded_data, columns=[f"encoded_{i + 1}" for i in range(encoded_data.shape[1])]
    )

    if save_path:
        save_csv(df_encoded, save_path / "encoded.csv")

    return df_encoded


def hpo_objective(
    df: pd.DataFrame,
    hpo_space: Callable[[optuna.Trial], dict[str, Any]],
) -> Callable[[optuna.Trial], float]:
    def _hpo_objective(
        trial: optuna.Trial,
        df: pd.DataFrame,
        hpo_space: Callable[[optuna.Trial], dict[str, Any]],
    ) -> float:
        params = hpo_space(trial)
        scaler = params.pop("scaler")
        epochs = params.pop("epochs")
        batch_size = params.pop("batch_size")

        data_scaled = scaler().fit_transform(df)
        data_tensor = torch.FloatTensor(data_scaled)

        dataset = TensorDataset(data_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        input_dim = data_tensor.shape[1]
        autoencoder = AutoEncoder(input_dim, **params).to(device)

        trainer = Trainer(
            max_epochs=epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            enable_progress_bar=True,
        )
        trainer.fit(
            autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        val_loss = trainer.callback_metrics.get("val_loss", None)
        if val_loss is None:
            raise ValueError("Validation loss not found in trainer metrics")

        return val_loss.item()

    return partial(
        _hpo_objective,
        df=df,
        hpo_space=hpo_space,
    )


def run_hpo_pipeline(
    config: AutoencoderHPOConfig,
    df: pd.DataFrame,
):
    _, best_params, _, _ = run_hpo(
        n_trials=config.n_trials,
        hpo_path=config.hpo_path / "autoencoder",
        study_name=config.study_name,
        objective=hpo_objective(
            df=df,
            hpo_space=autoencoder_hpo_space,
        ),
    )

    hidden_dims = []
    for layer in range(best_params.pop("n_layers")):
        hidden_dims.append(best_params.pop(f"n_units_l{layer}"))

    autoencode(
        df=df,
        scaler=sklearn_scaler_registry[best_params.pop("scaler")],
        epochs=best_params.pop("epochs"),
        batch_size=best_params.pop("batch_size"),
        save_path=config.save_data_path,
        hidden_dims=hidden_dims,
        activation_layer=activation_layer_registry[best_params.pop("activation_layer")],
        **best_params,
    )


if __name__ == "__main__":
    args = parse_config_path_args()
    config = init_autoencoder_hpo_config(args.config_path)

    df = read_parquet(
        config.dataset_path,
    )

    run_hpo_pipeline(
        config=config,
        df=df,
    )
