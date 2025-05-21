import numpy as np
from torch.utils.data import Dataset
import torch
# -----------------------------------------------------------
# Sequence dataset
# -----------------------------------------------------------
class SeqDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, target_idx: int):
        self.data = data
        self.seq_len = seq_len
        self.tgt = target_idx

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        seq   = self.data[idx : idx + self.seq_len]
        label = self.data[idx + self.seq_len, self.tgt]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)