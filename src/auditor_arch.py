import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────────────────────────────────────
#  BASELINE: Original Bi-LSTM Auditor (kept for ablation / comparison)
# ─────────────────────────────────────────────────────────────────────────────
class TrajectoryAuditor(nn.Module):
    """
    Bidirectional LSTM auditor.
    Kept as a baseline so you can run direct ablations against
    the new Transformer architecture.
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(TrajectoryAuditor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.fc      = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.sigmoid(self.fc(last_hidden))


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Trajectory Transformer Auditor
# ─────────────────────────────────────────────────────────────────────────────
class TrajectoryTransformer(nn.Module):
    """
    Transformer encoder for membership inference from adversarial trajectories.

    Key design decisions
    ────────────────────
    • Learnable positional embeddings  (not sinusoidal) — the model can learn
      which PGD step indices carry the most discriminative signal.
    • Mean-pool across the time axis before the classifier — more robust than
      using a [CLS] token given the short sequence length (12 steps).
    • LayerNorm before the classifier head (pre-norm style) for stability.
    • Dropout on the classifier head to guard against overfitting on small
      validation sets.

    Default parameters are sized for input_size=33 (11 features × 3 ε-scales)
    and seq_len=12.
    """
    def __init__(
        self,
        input_size: int = 33,   # features per step  (11 per ε-scale × 3 scales)
        d_model:    int = 128,  # Transformer hidden dim
        nhead:      int = 4,    # attention heads  (d_model must be divisible by nhead)
        num_layers: int = 3,    # Transformer encoder layers
        dim_ff:     int = 256,  # feed-forward inner dim
        dropout:    float = 0.1,
        seq_len:    int = 12,   # number of PGD steps kept
    ):
        super(TrajectoryTransformer, self).__init__()

        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        # Project raw features into d_model space
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable positional embeddings — one per PGD step
        self.pos_embed = nn.Embedding(seq_len, d_model)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',       # GELU outperforms ReLU here in practice
            batch_first=True,        # (B, T, d_model) convention
            norm_first=True,         # Pre-norm → more stable gradients
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Weight init
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, input_size)  — batch of trajectory sequences
        Returns:
            prob : (B, 1)  — membership probability
        """
        B, T, _ = x.shape

        # Positional encoding
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.input_proj(x) + self.pos_embed(positions)   # (B, T, d_model)

        # Transformer
        x = self.transformer(x)                               # (B, T, d_model)

        # Global mean pool over the time axis
        x = self.norm(x.mean(dim=1))                          # (B, d_model)

        return self.classifier(x)                             # (B, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity-check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, F = 8, 12, 33
    dummy = torch.randn(B, T, F)

    model = TrajectoryTransformer(input_size=F, seq_len=T)
    out   = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}")           # Expected: (8, 1)
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")

    # Baseline comparison
    baseline = TrajectoryAuditor(input_size=3)
    dummy_old = torch.randn(B, T, 3)
    out_old   = baseline(dummy_old)
    print(f"\nBaseline output : {out_old.shape}")
    print(f"Baseline params : {sum(p.numel() for p in baseline.parameters()):,}")