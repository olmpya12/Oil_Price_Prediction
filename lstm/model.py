import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttnBiLSTMRegressor(nn.Module):
    """
    Bidirectional LSTM + Self-Attention + FC head
    ------------------------------------------------
    • Input:  [B, T, F]   – T timesteps, F features
    • Output: [B]         – next-day scalar
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        attn_size: int = 128,
        fc_sizes: tuple[int, int] = (64, 32),
        dropout: float = 0.3,
    ):
        super().__init__()

        self.bi_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )  # output dim = 2*hidden_size

        self.attn_query = nn.Linear(2 * hidden_size, attn_size, bias=False)
        self.attn_key   = nn.Linear(2 * hidden_size, attn_size, bias=False)
        self.attn_value = nn.Linear(2 * hidden_size, attn_size, bias=False)

        self.layer_norm = nn.LayerNorm(attn_size)

        # Fully-connected head
        in_dim = attn_size
        fc1_out, fc2_out = fc_sizes
        self.fc1 = nn.Linear(in_dim, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.out = nn.Linear(fc2_out, 1)

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Self-attention block (scaled dot-product over time axis)
    # ------------------------------------------------------------------
    def _temporal_attention(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: [B, T, 2H]
        Q = self.attn_query(lstm_out[:, -1:, :])      # query => last time-step (B,1,attn)
        K = self.attn_key(lstm_out)                   # keys   (B,T,attn)
        V = self.attn_value(lstm_out)                 # values (B,T,attn)

        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(K.size(-1))  # (B,1,T)
        weights = F.softmax(scores, dim=-1)           # (B,1,T)
        context = torch.matmul(weights, V).squeeze(1) # (B,attn)

        return context                                 # context vector

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        lstm_out, _ = self.bi_lstm(x)                 # (B,T,2H)

        # Attention context vector
        context = self._temporal_attention(lstm_out)  # (B, attn)

        # Residual skip from last hidden step after projection
        last_step = lstm_out[:, -1, :]                # (B, 2H)
        last_proj = self.attn_value(last_step)        # (B, attn)

        combined = self.layer_norm(context + last_proj)  # residual + layer norm

        # FC head
        h = self.dropout(F.relu(self.fc1(combined)))
        h = self.dropout(F.relu(self.fc2(h)))
        out = self.out(h).squeeze(1)                  # [B]

        return out
