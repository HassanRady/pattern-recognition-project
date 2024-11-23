from pathlib import Path
from typing import List, Any, Optional

import pandas as pd
import torch
import torch.nn as nn
import lightning as pl
from lightning import Trainer

from data.data_manager import read_parquet, save_csv
from torch.utils.data import DataLoader, TensorDataset

from utils.registry import sklearn_scaler_registry, ActivationFunctionType, ScalerType

from src.utils.registry import activation_layer_registry


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation_layer: ActivationFunctionType,
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
        y_hat = self(batch)
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
    scaler: ScalerType,
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


if __name__ == "__main__":
    df = read_parquet(
        "/mnt/MAIN/Master/WS-24-25/Pattern-Recognition-Project/pattern-recognition-project/data/series_train.parquet/id=0a418b57"
    )
    encoded_df = autoencode(
        df=df,
        epochs=1,
        batch_size=64,
        scaler=sklearn_scaler_registry["StandardScaler"],
        hidden_dims=[5, 3, 2],
        activation_layer=activation_layer_registry["relu"],
        learning_rate=0.01,
    )
    print(encoded_df)
