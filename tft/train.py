"""run_tft_best_version.py — Train a Temporal Fusion Transformer to forecast
next‑day WTI crude‑oil prices with macro‑fundamental covariates

▪ Handles mixed‑frequency data (daily price + monthly/quarterly macro).
▪ Converts calendar to business‑day index to avoid weekend gaps.
▪ Adds lagged & rolling statistical features.
▪ Walk‑forward validation + early stopping + LR scheduler + checkpoint.
▪ Saves loss curves and best model checkpoint.

"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE

###############################################################################
# 1 ‑ Configuration
###############################################################################
DATA_PATH = Path("/home/ilayda/Workspace/oil_price/data/processed/filtered_merged_dataset_trimmed.csv")
RUN_NAME = "tft-oil-best"
ENC_LEN_DAYS = 120           # past horizon shown to encoder (≈6 months)
PRED_LEN = 1                 # 1‑day ahead forecast
VAL_WINDOW_DAYS = 365        # rolling window size for walk‑forward validation
MAX_EPOCHS = 300
LR = 3e-4
BATCH_SIZE = 64
NUM_WORKERS = 4              # plenty for 2 k‑row dataset
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)





train_df = pd.read_csv('/home/ilayda/Workspace/oil_price/train.csv')
val_df = pd.read_csv('/home/ilayda/Workspace/oil_price/val.csv')
lag_cols = [c for c in train_df.columns if c.startswith("ret_") or c.startswith("vol_")]
###############################################################################
# 6 ‑ Dataset definitions
###############################################################################
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
    target="Crude Oil ($/barrel)",
    group_ids=["series"],
    min_encoder_length=ENC_LEN_DAYS,
    max_encoder_length=ENC_LEN_DAYS,
    min_prediction_length=PRED_LEN,
    max_prediction_length=PRED_LEN,
    static_categoricals=["series"],
    time_varying_known_categoricals=["month", "weekday"],
    time_varying_known_reals=known_reals,
    time_varying_unknown_reals=["Crude Oil ($/barrel)"],
    target_normalizer=GroupNormalizer(groups=["series"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,  
)

train_ds = TimeSeriesDataSet(train_df, **common_kw)
val_ds   = TimeSeriesDataSet.from_dataset(train_ds, val_df, predict=True, stop_randomization=True)

train_loader = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
val_loader   = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

###############################################################################
# 6 ‑ Build TFT
###############################################################################
print("[INFO] Building model…")

tft = TemporalFusionTransformer.from_dataset(
    train_ds,
    learning_rate=LR,
    hidden_size=256,
    dropout=0.3,
    attention_head_size=32,
    hidden_continuous_size=64,
    loss=RMSE(),
)

# Custom optimiser + scheduler (ReduceLROnPlateau)

def configure_optimizers(self):
    opt = torch.optim.Adam(self.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4, threshold=1e-4
    )
    return {"optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}
tft.configure_optimizers = configure_optimizers.__get__(tft, TemporalFusionTransformer)

###############################################################################
# 7 ‑ Trainer, early‑stopping & checkpoints
###############################################################################
logger = CSVLogger("logs", name=RUN_NAME)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-4, verbose=True, mode="min"),
    ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"{RUN_NAME}-best",
    ),
    RichProgressBar(),
]

trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator=DEVICE,
    devices=1,
    callbacks=callbacks,
    logger=logger,
    gradient_clip_val=0.1,
    log_every_n_steps=1,
)

###############################################################################
# 8 ‑ Fit
###############################################################################
print("[INFO] Starting training…")
trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

###############################################################################
# 9 ‑ Report stop reason
###############################################################################
stop_reason = "max_epochs reached"
for cb in trainer.callbacks:
    if isinstance(cb, EarlyStopping) and cb.stopped_epoch > 0:
        stop_reason = f"early-stopped @ epoch {cb.stopped_epoch} (best val_loss={cb.best_score:.4f})"
        break
print(f"\n[INFO] Training finished → {stop_reason}")

###############################################################################
# 10 ‑ Plot loss curves (headless)
###############################################################################
print("[INFO] Plotting loss curves…")
metrics = pd.read_csv(Path(logger.log_dir) / "metrics.csv")
train_curve = metrics.groupby("epoch")["train_loss_step"].mean()
val_curve   = metrics.groupby("epoch")["val_loss"].mean()

plt.figure(figsize=(9, 4))
plt.plot(train_curve.index, train_curve.values, label="Train loss")
plt.plot(val_curve.index, val_curve.values, label="Val loss")
plt.yscale("log")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.grid(True); plt.legend(); plt.title("TFT Training vs Validation Loss")
plt.tight_layout()
plot_path = Path(logger.log_dir) / "loss_curves.png"
plt.savefig(plot_path)
print(f"[INFO] Saved loss plot to {plot_path}")

###############################################################################
# 11 ‑ Save final checkpoint
###############################################################################
final_ckpt = Path(logger.log_dir) / "final.ckpt"
trainer.save_checkpoint(final_ckpt)
print(f"[INFO] Saved final checkpoint to {final_ckpt}")

