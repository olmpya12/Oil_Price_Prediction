#!/usr/bin/env python3
"""
visualize_sliding_predictions.py
--------------------------------
• Loads your trimmed dataset and a trained LSTM checkpoint
• Generates one-step-ahead forecasts with a 240-day sliding window
• Saves a PNG chart of Actual vs Predicted prices
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")          # head-less backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from model import AttnBiLSTMRegressor

# ------------------------------------------------------------------
# Config – adjust paths or parameters if needed
# ------------------------------------------------------------------
CSV_PATH   = Path("/home/ilayda/Workspace/oil_price/final_ds.csv")
CKPT_PATH  = Path("/home/ilayda/Workspace/oil_price/lstm_best.pt")       # trained weights
SEQ_LEN    = 120                        # look-back window
TARGET_COL = "Crude Oil ($/barrel)"
OUT_PNG    = Path("lstm_sliding_predictions.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# 1. Load & basic preprocessing (same as training pipeline)
# ------------------------------------------------------------------
df = (
    pd.read_csv(CSV_PATH, parse_dates=["Date"])
      .sort_values("Date")
      .reset_index(drop=True)
)

df["value"] = df["value"].ffill()
df["Change %"] = (
    df["Change %"].astype(str).str.replace("%", "", regex=False).astype(float)
)
df["days_since_last_trade"] = df["Date"].diff().dt.days.fillna(1)

numeric_df = (
    df.select_dtypes(include=["float64", "int64"])
      .dropna()
      .reset_index(drop=True)
)

feature_names = numeric_df.columns.tolist()
target_idx    = feature_names.index(TARGET_COL)

# Min-max scale **entire dataset** for consistent plotting
scaler     = MinMaxScaler()
scaled_all = scaler.fit_transform(numeric_df.values)

# ------------------------------------------------------------------
# 2. Model definition (same structure as during training)
# ------------------------------------------------------------------

n_features = scaled_all.shape[1]
model = AttnBiLSTMRegressor(n_features).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

# ------------------------------------------------------------------
# 3. Sliding-window predictions
# ------------------------------------------------------------------
pred_scaled = []
with torch.no_grad():
    for start in range(0, len(scaled_all) - SEQ_LEN):
        seq = torch.tensor(
            scaled_all[start : start + SEQ_LEN],
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)                    # [1,SEQ_LEN,F]
        pred = model(seq).item()
        pred_scaled.append(pred)

pred_scaled = np.array(pred_scaled)                       # length = N-SEQ_LEN
actual_scaled = scaled_all[SEQ_LEN:, target_idx]

# Inverse-transform target back to $/bbl
def inverse_target(arr_scaled):
    dummy = np.zeros((len(arr_scaled), scaled_all.shape[1]))
    dummy[:, target_idx] = arr_scaled
    return scaler.inverse_transform(dummy)[:, target_idx]

pred_price   = inverse_target(pred_scaled)
actual_price = inverse_target(actual_scaled)
dates        = df["Date"].iloc[SEQ_LEN:].values

# ------------------------------------------------------------------
# 4. Plot & save
# ------------------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(dates, actual_price, label="Actual", linewidth=1.2)
plt.plot(dates, pred_price,   label="Predicted", linewidth=1.2, alpha=0.8)
plt.title(f"LSTM one-day-ahead forecasts (window = {SEQ_LEN} days)")
plt.xlabel("Date"); plt.ylabel("Price ($/bbl)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(OUT_PNG)
plt.close()

print(f"✅ Plot saved to {OUT_PNG.resolve()}")
