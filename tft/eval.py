#!/usr/bin/env python3
"""
eval_tft_sliding.py — Sliding-window inference on the validation set
===================================================================
• Re-creates the TimeSeriesDataSet with the same hyper-params.
• Loads the best trained Temporal Fusion Transformer checkpoint.
• Generates 1-day-ahead forecasts for every row of `val_df`.
• Computes RMSE & MAE and stores a “pred_vs_actual_YYYYMMDD.csv”.
• Plots predictions vs. actuals (saved as PNG).
"""

from __future__ import annotations
import argparse, datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE
from sklearn.metrics import mean_absolute_error,root_mean_squared_error

###############################################################################
# 0 - CLI
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="Path to trained model checkpoint (.ckpt)")
parser.add_argument("--train_csv", required=True, help="Path to train.csv")
parser.add_argument("--val_csv", required=True, help="Path to val.csv")
parser.add_argument("--outdir", default="eval_out", help="Output folder")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Path(args.outdir).mkdir(parents=True, exist_ok=True)

###############################################################################
# Load data
###############################################################################
train_df = pd.read_csv(args.train_csv)
val_df = pd.read_csv(args.val_csv)
ENC_LEN_DAYS = 120
PRED_LEN = 1
target_col = "Crude Oil ($/barrel)"
series_col = "series"

lag_cols = [c for c in train_df.columns if c.startswith("ret_") or c.startswith("vol_")]
macro_real_cols = [
    "Crude Oil Production", "Dry Natural Gas Production", "Coal Production",
    "Total Energy Production (qBtu)", "Liquid Fuels Consumption", "Natural Gas Consumption",
    "Coal Consumption", "Electricity Consumption", "Renewables Consumption",
    "Total Energy Consumption (qBtu)", "Real Gross Domestic Product (Trillions)",
    "RGDP Percent change YOY (%)", "GDP Implicit Price Deflator", "GDP IPD Percent change YOY (%)",
    "Real Disposable Personal Income", "RDPI Percent change YOY (%)",
    "Manufacturing Production Index", "MPI Percent change YOY (%)",
]
known_reals = ["time_idx"] + macro_real_cols + lag_cols

common_kw = dict(
    time_idx="time_idx",
    target=target_col,
    group_ids=[series_col],
    min_encoder_length=ENC_LEN_DAYS,
    max_encoder_length=ENC_LEN_DAYS,
    min_prediction_length=PRED_LEN,
    max_prediction_length=PRED_LEN,
    static_categoricals=[series_col],
    time_varying_known_categoricals=["month", "weekday"],
    time_varying_known_reals=known_reals,
    time_varying_unknown_reals=[target_col],
    target_normalizer=GroupNormalizer(groups=[series_col]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

train_ds = TimeSeriesDataSet(train_df, **common_kw)

###############################################################################
# Load trained model
###############################################################################
print("[INFO] Loading model checkpoint...")
try:
    model = TemporalFusionTransformer.load_from_checkpoint(
        args.ckpt, map_location=DEVICE, dataset=train_ds
    )
except TypeError:
    model = TemporalFusionTransformer.load_from_checkpoint(
        args.ckpt, map_location=DEVICE
    )
    model.dataset_parameters = train_ds.get_parameters()

model = model.to(DEVICE).eval()

###############################################################################
# Sliding-window predictions
###############################################################################
print("[INFO] Running 1-step sliding-window predictions…")

preds, actuals, dates = [], [], []
for i in range(ENC_LEN_DAYS, len(val_df)):
    window_df = val_df.iloc[i - ENC_LEN_DAYS : i + PRED_LEN].copy()

    # Build dataset for this small window
    try:
        window_ds = TimeSeriesDataSet.from_dataset(train_ds, window_df, predict=True, stop_randomization=True)
        window_dl = window_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

        with torch.no_grad():
            pred = model.predict(window_dl).squeeze().item()
        actual = val_df.iloc[i][target_col]
        date = val_df.iloc[i]["Date"]

        preds.append(pred)
        actuals.append(actual)
        dates.append(date)
    except Exception as e:
        print(f"[WARN] Skipped index {i}: {e}")
        continue

###############################################################################
# Metrics and Output
###############################################################################
out_df = pd.DataFrame({
    "date": dates,
    "actual": actuals,
    "prediction": preds
})

rmse = root_mean_squared_error(out_df.actual, out_df.prediction)
mae = mean_absolute_error(out_df.actual, out_df.prediction)

print(f"[RESULT] RMSE: {rmse:.4f}")
print(f"[RESULT] MAE : {mae:.4f}")

timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = Path(args.outdir) / f"pred_vs_actual_{timestamp}.csv"
plot_path = Path(args.outdir) / f"loss_curve_{timestamp}.png"

out_df.to_csv(csv_path, index=False)
print(f"[INFO] Saved CSV → {csv_path}")

plt.figure(figsize=(10, 4))
plt.plot(out_df.date, out_df.actual, label="Actual", linewidth=1.5)
plt.plot(out_df.date, out_df.prediction, label="Prediction", linewidth=1.5)
plt.title("TFT Sliding Forecast – 1-step-ahead")
plt.xlabel("Date")
plt.ylabel("Crude Oil ($/barrel)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path)
print(f"[INFO] Saved plot → {plot_path}")
