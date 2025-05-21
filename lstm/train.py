"""
Train an LSTM to predict next-day crude-oil price
from the filtered_merged_dataset_trimmed.csv.
Author: you
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model import AttnBiLSTMRegressor
from dataset import SeqDataset
from utils import build_dataloaders_wf

# -----------------------------------------------------------
# 1. Hyper-params
# -----------------------------------------------------------
CSV_PATH      = "/home/ilayda/Workspace/oil_price/final_ds.csv"
SEQ_LEN       = 120        # look-back window (days)
BATCH_SIZE    = 32
LR            = 1e-3
EPOCHS        = 200
PATIENCE      = 25        # early-stop patience
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_COL    = "Crude Oil ($/barrel)"

# -----------------------------------------------------------
# 2. Load & basic cleaning
# -----------------------------------------------------------
df = (
    pd.read_csv(CSV_PATH, parse_dates=["Date"])
      .sort_values("Date")
      .reset_index(drop=True)
)

df["days_since_last_trade"] = df["Date"].diff().dt.days

# 2.  The very first row has NaN (no previous day) → replace with 1
df["days_since_last_trade"].fillna(1, inplace=True)

# forward fill the tiny Fed-rate NaNs
df["value"] = df["value"].ffill()

# percent column to float
if "Change %" in df.columns:
    df["Change %"] = (
        df["Change %"].astype(str).str.replace("%", "", regex=False).astype(float)
    )

# just keep numeric features (drop Date for modeling)
numeric_df = df.select_dtypes(include=["float64", "int64"]).copy()

# -----------------------------------------------------------
# 3. Scale features 0-1   (fit on training only!)
# -----------------------------------------------------------
train_size = int(len(numeric_df) * 0.8)
train_raw  = numeric_df.iloc[:train_size]
test_raw   = numeric_df.iloc[train_size:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw)
test_scaled  = scaler.transform(test_raw)

scaled = np.vstack([train_scaled, test_scaled])  # full array back



target_index = numeric_df.columns.get_loc(TARGET_COL)

cutoffs = pd.to_datetime(
    ["2020-06-30", "2020-12-31", "2021-06-30", "2021-12-31"]
)
train_loader, val_loader = build_dataloaders_wf(
    df,scaled, SEQ_LEN, target_index, cutoffs, batch=BATCH_SIZE
)

n_features = train_scaled.shape[1]
model = AttnBiLSTMRegressor(n_features).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------------------------------------
# 6. Training loop with early stopping
# -----------------------------------------------------------
best_val_rmse = math.inf
epochs_no_improve = 0

def evaluate(loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for seq, y in loader:
            seq, y = seq.to(DEVICE), y.to(DEVICE)
            o = model(seq)
            preds.append(o.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    rmse = math.sqrt(mean_squared_error(trues, preds))
    mae  = mean_absolute_error(trues, preds)
    return rmse, mae, preds, trues

for epoch in range(1, EPOCHS + 1):
    model.train()
    for seq, y in train_loader:
        seq, y = seq.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(seq)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    val_rmse, val_mae, _, _ = evaluate(val_loader)
    print(f"Epoch {epoch:03d} | Val RMSE {val_rmse:.4f}  MAE {val_mae:.4f}")

    # Early stop
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        epochs_no_improve = 0
        torch.save(model.state_dict(), "lstm_best.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early-stopping triggered at epoch {epoch}.")
            break

# -----------------------------------------------------------
# 7. Load best & final evaluation (de-scale for reporting)
# -----------------------------------------------------------
model.load_state_dict(torch.load("lstm_best.pt"))
best_rmse, best_mae, scaled_preds, scaled_true = evaluate(val_loader)

# De-scale the predictions (only target column back-transform)
def inverse_transform_target(arr_scaled):
    dummy = np.zeros((len(arr_scaled), scaler.n_features_in_))
    dummy[:, target_index] = arr_scaled
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_index]

pred_price = inverse_transform_target(scaled_preds)
true_price = inverse_transform_target(scaled_true)

final_rmse = math.sqrt(mean_squared_error(true_price, pred_price))
final_mae  = mean_absolute_error(true_price, pred_price)

print("\n============= Final Evaluation =============")
print(f"Best Val RMSE (scaled): {best_rmse:.4f}")
print(f"Best Val MAE  (scaled): {best_mae:.4f}")
print(f"Test RMSE on price   : {final_rmse:.2f}")
print(f"Test MAE  on price   : {final_mae:.2f}")
print("Best model weights saved to lstm_best.pt")

import matplotlib.pyplot as plt
plt.plot(true_price, label="actual")
plt.plot(pred_price, label="LSTM")
plt.legend(); plt.savefig("lstm_pred_vs_actual.png")