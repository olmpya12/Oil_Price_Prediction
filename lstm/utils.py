import numpy as np
import pandas as pd
from dataset import SeqDataset
from torch.utils.data import Dataset, DataLoader
def build_dataloaders_wf(
    df,
    arr_scaled: np.ndarray,
    seq_len: int,
    target_idx: int,
    fold_end_dates: list[pd.Timestamp],
    batch=64,
):
    """
    Returns  (train_loader, val_loader)  for the latest fold only,
    and a list of (rmse, mae) scores for previous folds.
    """
    last_end = 0
    for i, end_date in enumerate(fold_end_dates):
        split_idx = df.index[df["Date"] <= end_date].max()   # inclusive idx
        train_raw = arr_scaled[: split_idx + 1]
        val_raw   = arr_scaled[split_idx + 1 :]

        train_ds = SeqDataset(train_raw, seq_len, target_idx)
        val_ds   = SeqDataset(val_raw,   seq_len, target_idx)

        train_loader = DataLoader(train_ds, batch_size=batch,
                                  shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds,  batch_size=batch,
                                  shuffle=False, drop_last=False)


        return train_loader, val_loader
