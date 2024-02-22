import json
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

import torch.nn as nn
from torch.nn import Linear

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf

import import_ipynb


def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask

def smape_loss(y_pred, target):
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8 )
    return loss.mean()  

class TimeSeriesForcasting(pl.LightningModule):
    def __init__(self, n_encoder_inputs, n_decoder_inputs, channels, dropout, lr):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.dropout = dropout

        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=8, #8
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels,
            nhead=8, #8
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)#8
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)#8

        self.input_projection = Linear(n_encoder_inputs, channels)
        self.output_projection = Linear(n_decoder_inputs, channels)

        self.linear = Linear(channels, 1)

        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src):
        
        src_start = self.input_projection(src).permute(1, 0, 2)

        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)

        src = src_start + pos_encoder

        src = self.encoder(src) + src_start

        return src

    def decode_trg(self, trg, memory):
        

        trg_start = self.output_projection(trg).permute(1, 0, 2)

        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)

        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)

        trg = pos_decoder + trg_start

        trg_mask = gen_trg_mask(out_sequence_len, trg.device)

        out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask) + trg_start

        out = out.permute(1, 0, 2)

        out = self.linear(out)

        return out

    def forward(self, x):
        src, trg = x

        src = self.encode_src(src)

        out = self.decode_trg(trg=trg, memory=src)

        return out

    def training_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)
        mse = mean_squared_error(y.cpu().numpy(), y_hat.cpu().detach().numpy())
        mae = mean_absolute_error(y.cpu().numpy(), y_hat.cpu().detach().numpy())

        #self.log("train_loss", loss,  prog_bar=True, logger=True)
        #self.log("train_mae", mae,  prog_bar=True, logger=True)
        self.log("train_loss", loss)
        self.log("train_mae", mae)
        self.log("train_mse", mse)


        return {"loss": loss, "mae": mae, "mse": mse}

    def validation_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)
        mse = mean_squared_error(y.cpu().numpy(), y_hat.cpu().detach().numpy())
        mae = mean_absolute_error(y.cpu().numpy(), y_hat.cpu().detach().numpy())

        #self.log("valid_loss", loss, prog_bar=True, logger=True)
        #self.log("valid_mae", mae, prog_bar=True, logger=True)
        self.log("valid_loss", loss )
        self.log("valid_mae", mae)
        self.log("valid_mse", mse)

        return {"loss": loss, "mae": mae, "mse": mse}
     
    def test_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch
        y_hat = self((src, trg_in)).view(-1)
        y = trg_out.view(-1)
        
        loss = smape_loss(y_hat, y)
        mae = mean_absolute_error(y.cpu().numpy(), y_hat.cpu().detach().numpy())
        mse = mean_squared_error(y.cpu().numpy(), y_hat.cpu().detach().numpy())
        #self.log("test_loss", loss, prog_bar=True, logger=True)
        #self.log("test_mae", mae, prog_bar=True, logger=True)
        self.log("test_loss", loss)
        self.log("test_mae", mae)
        self.log("test_mse", mse)

        return {"test_loss": loss, "test_mae": mae, "test_mse": mse}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }
def split_df(
    df: pd.DataFrame, split: str, history_size, horizon_size
):
   

    if history_size:
        if split == "train":
            end_index = random.randint(horizon_size + 1, df.shape[0] - horizon_size)
        elif split in ["val", "test"]:
            end_index = df.shape[0]
        else:
            raise ValueError

        label_index = end_index - horizon_size
        start_index = max(0, label_index - history_size) # no negative value

        history = df[start_index:label_index]
        targets = df[label_index:end_index]

        return history, targets
    else:
        train_percent = 0.70
        val_percent = 0.15

        total_size = len(df)
        train_size = int(total_size * train_percent)
        val_size = int(total_size * val_percent)

        if split == "train":
            history = df[:train_size]
            targets = df[train_size: train_size+ horizon_size]
        elif split == "val":
            history = df[train_size:train_size + val_size]
            targets = df[train_size + val_size: train_size + val_size+horizon_size]
        elif split == "test":
            history = df[train_size + val_size:-horizon_size]
            targets = df[train_size + val_size-horizon_size:]
        else:
            raise ValueError("Ungültiger Wert für 'split'. Erwartet 'train', 'val' oder 'test'.")

        return history, targets

def pad_arr(arr: np.ndarray, expected_size):
    
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
    return arr


def df_to_np(df, history_size):
    arr = np.array(df)
    arr = pad_arr(arr, history_size)
    return arr

class Dataset(torch.utils.data.Dataset):
    def __init__(self, groups, grp_by, split, features, target, history, horizon):
        self.groups = groups
        self.grp_by = grp_by
        self.split = split
        self.features = features
        self.target = target
        self.history = history
        self.horizon = horizon
        

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        df = self.grp_by.get_group(group)
        

        src, trg = split_df(df, split=self.split, history_size=self.history, horizon_size=self.horizon)

        src = src[self.features+ self.target]
        

        #src = df_to_np(src)

        #trg_in = trg[self.features + ["p_(mbar)_lag_1", "T_(degC)_lag_1"]]
        trg_in = trg[self.features+ self.target]
        

        # String Konvertieren zu Datetime
        
        if "Date_Time" in self.features:
            date_time_str = "Date_Time"
            trg_in[date_time_str] = pd.to_datetime(trg_in[date_time_str], format='%Y-%m-%d %H:%M:%S')
            src[date_time_str] = pd.to_datetime(src[date_time_str], format='%Y-%m-%d')
            

            # Funktion zum Konvertieren von datetime in Stunden (als Float)
            def convert_to_hours(dt):
                return dt.timestamp() / 3600

            # Konvertieren zu Stunden
            trg_in[date_time_str] = trg_in[date_time_str].apply(convert_to_hours)
            src[date_time_str] = src[date_time_str].apply(convert_to_hours)
            
            # Konvertieren zu float
            trg_in = trg_in.astype(np.float32)
            src = src.astype(np.float32)

            
        src = df_to_np(src, self.history)
        
        trg_in = np.array(trg_in)
        trg_out = np.array(trg[self.target])
        
        src = torch.tensor(src, dtype=torch.float)
        trg_in = torch.tensor(trg_in, dtype=torch.float)
        trg_out = torch.tensor(trg_out, dtype=torch.float)

        return src, trg_in, trg_out


    
def train(
    data_csv_path: str,
    feature_target_names_path: str,
    output_json_path: str,
    log_dir: str,
    model_dir: str ,
):
    
    data = pd.read_csv(data_csv_path)

    with open(feature_target_names_path) as f:
        feature_target_names = json.load(f)


    data_train = data
    # data_train = data[~data[feature_target_names["target"]].isna()]


    grp_by_train = data_train.groupby(by=feature_target_names["group_by_key"])
    

    full_groups = []
    groups = list(grp_by_train.groups)
    for grp in groups:
        #print(grp_by_train.get_group(grp).shape[0], horizon_size)
        if grp_by_train.get_group(grp).shape[0] > 2 * feature_target_names["horizon_size"]:
            full_groups.append(grp)


    #full_groups = [
    #    grp for grp in groups if grp_by_train.get_group(grp).shape[0] > 2 * horizon_size
    #]
    
    train_data = Dataset(
        groups=full_groups,
        grp_by=grp_by_train,
        split="train",
        features=feature_target_names["features"],
        target=feature_target_names["target"],
        horizon=feature_target_names["horizon_size"],
        history=feature_target_names["history_size"],
    )
    val_data = Dataset(
        groups=full_groups,
        grp_by=grp_by_train,
        split="val",
        features=feature_target_names["features"],
        target=feature_target_names["target"],
        horizon=feature_target_names["horizon_size"],
        history=feature_target_names["history_size"],
    )

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=feature_target_names["batch_size"],
        num_workers=12,
        persistent_workers=True,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=feature_target_names["batch_size"],
        num_workers=12,
        persistent_workers=True,
        shuffle=False,
    )
    print("1")

    model = TimeSeriesForcasting(
        n_encoder_inputs=len(feature_target_names["features"]) + 1,
        n_decoder_inputs=len(feature_target_names["features"]) + 1,
        lr=feature_target_names["lr"],
        dropout=feature_target_names["dropout"],
        channels=feature_target_names["channels"]
    )
    print("2")

    logger = TensorBoardLogger(
        save_dir=log_dir,
    )
    print("3")
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename="ts",
    )
    print("4")
    trainer = pl.Trainer(
        log_every_n_steps=10,
        max_epochs=feature_target_names["epochs"],
        accelerator='auto',
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    print("5")
    trainer.fit(model, train_loader, val_loader)
    print("6")
    result_val = trainer.test(dataloaders=val_loader)
    result_val = trainer.test(dataloaders=val_loader)

    output_json = {
        "val_loss": result_val[0]["test_loss"],
        "test_mae": result_val[0]["test_mae"],
        "test_mse": result_val[0]["test_mse"],
        "best_model_path": checkpoint_callback.best_model_path,
    }
    print("8")  
    if output_json_path is not None:
        with open(output_json_path, "w") as f:
            json.dump(output_json, f, indent=4)


if __name__ == '__main__':


    data_csv_path = "data/processed_weather_data_1H.csv"
    feature_target_names_path = "data/config_1H_1feature.json"
    output_json_path = "models/trained_config_1H_1feature.json"
    log_dir = "models/ts_views_logs"
    model_dir = "models/ts_views_models"



    train(
        data_csv_path=data_csv_path,
        feature_target_names_path=feature_target_names_path,
        output_json_path=output_json_path,
        log_dir=log_dir,
        model_dir=model_dir,
    )