"""
maestro/agents/regime/transformer_regime.py
=============================================
Transformer-based sequence classifier for market regime detection.

Why a Transformer on top of the HMM?
--------------------------------------
The HMM assumes Gaussian emissions and Markov transitions — both are
violated in real financial data. The Transformer learns arbitrary,
non-linear regime boundaries from raw sequences without these assumptions.
Together (in the ensemble), they complement each other perfectly:
HMM = probabilistic structure, Transformer = discriminative power.

Architecture
------------
- Input:  sliding window of T=60 bars × F features
- Encoder: 4-layer Transformer encoder (d_model=64, 4 heads, FFN=256)
- Pooling: attention-weighted mean pooling over sequence
- Head:   linear → 4-class softmax (bull, bear, sideways, crisis)
- Loss:   cross-entropy with class-frequency weighting (handles imbalance)
- Output: P(regime | X_{t-T:t}) — probability over 4 regimes at each bar

Training details
----------------
- Labels come from HMM (self-supervised bootstrapping for initial training,
  then refined with human-verified crisis labels)
- Data augmentation: random Gaussian noise + temporal jitter
- Regularisation: dropout=0.1, weight decay=1e-4
- Optimiser: AdamW with cosine LR schedule + warmup

Dependencies: torch, numpy, pandas, scikit-learn
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Must match HMM feature set (both models see same inputs)
TRANSFORMER_FEATURES = [
    "log_return_1",
    "log_return_5",
    "vol_realised",
    "vol_gk",
    "rsi",
    "macd_histogram",
    "bb_pct",
    "atr_normalised",
    "volume_zscore",
    "adx",
    "vix_zscore",
    "yield_curve_inverted",
    "rate_change_1m",
    "cross_corr_20",         # EUR/USD vs GBP/USD correlation
    "return_zscore_252",
]

REGIME_NAMES = {
    0: "bull_trend",
    1: "bear_trend",
    2: "sideways",
    3: "crisis",
}
N_REGIMES = 4


@dataclass
class TransformerConfig:
    seq_len:       int   = 60      # lookback window in bars (60 × 5min = 5 hours)
    d_model:       int   = 64      # embedding dimension
    n_heads:       int   = 4       # attention heads (d_model must be divisible by n_heads)
    n_layers:      int   = 4       # transformer encoder layers
    d_ff:          int   = 256     # feedforward network dimension
    dropout:       float = 0.1
    n_classes:     int   = N_REGIMES
    lr:            float = 3e-4
    weight_decay:  float = 1e-4
    batch_size:    int   = 128
    max_epochs:    int   = 50
    patience:      int   = 8       # early stopping patience
    warmup_steps:  int   = 200
    device:        str   = "cpu"   # set to "cuda" if GPU available


class PositionalEncoding(object):
    """
    Sinusoidal positional encoding — injects bar position into embeddings.
    Implemented as a plain object to avoid torch import at module level.
    """
    @staticmethod
    def build(seq_len: int, d_model: int) -> "torch.Tensor":
        import torch
        pe     = torch.zeros(seq_len, d_model)
        pos    = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div    = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # (1, seq_len, d_model)


class RegimeTransformer(object):
    """
    PyTorch Transformer encoder for regime classification.
    Defined as a plain class to keep torch import lazy.
    """
    _model_class = None

    @classmethod
    def _get_model_class(cls):
        """Build the nn.Module class on first access (lazy torch import)."""
        if cls._model_class is not None:
            return cls._model_class

        import torch
        import torch.nn as nn

        class _RegimeTransformerModule(nn.Module):
            def __init__(self, n_features: int, cfg: TransformerConfig):
                super().__init__()
                self.cfg = cfg
                self.input_proj  = nn.Linear(n_features, cfg.d_model)
                self.pos_enc     = PositionalEncoding.build(cfg.seq_len, cfg.d_model)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model         = cfg.d_model,
                    nhead           = cfg.n_heads,
                    dim_feedforward = cfg.d_ff,
                    dropout         = cfg.dropout,
                    batch_first     = True,
                    norm_first      = True,    # Pre-LN for training stability
                )
                self.encoder   = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
                self.attn_pool = nn.Linear(cfg.d_model, 1)   # attention pooling weights
                self.head      = nn.Linear(cfg.d_model, cfg.n_classes)
                self.dropout   = nn.Dropout(cfg.dropout)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                # x: (batch, seq_len, n_features)
                B, T, _ = x.shape
                h = self.input_proj(x)              # (B, T, d_model)
                h = h + self.pos_enc[:, :T, :].to(x.device)
                h = self.encoder(h)                 # (B, T, d_model)
                # Attention-weighted pooling
                w = torch.softmax(self.attn_pool(h), dim=1)   # (B, T, 1)
                pooled = (w * h).sum(dim=1)         # (B, d_model)
                out = self.head(self.dropout(pooled))          # (B, n_classes)
                return out

        cls._model_class = _RegimeTransformerModule
        return cls._model_class


class TransformerRegimeClassifier:
    """
    Train and infer with the Transformer regime classifier.

    Usage
    -----
    >>> clf = TransformerRegimeClassifier()
    >>> clf.fit(train_df, label_series)          # label_series from HMM
    >>> probs = clf.predict_proba(test_df)       # pd.DataFrame (n_bars, 4)
    >>> preds = clf.predict(test_df)             # pd.Series[int]
    """

    def __init__(self, config: TransformerConfig | None = None) -> None:
        self.cfg     = config or TransformerConfig()
        self.model   = None
        self.scaler  = None
        self.fitted  = False
        self._features: list[str] = []

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        val_df: pd.DataFrame | None = None,
        val_labels: pd.Series | None = None,
    ) -> "TransformerRegimeClassifier":
        """
        Train the Transformer regime classifier.

        Parameters
        ----------
        df        : feature DataFrame (train set)
        labels    : integer regime labels {0,1,2,3}, same index as df
        val_df    : optional validation DataFrame for early stopping
        val_labels: labels for val_df

        Returns self for chaining.
        """
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        self._features = [f for f in TRANSFORMER_FEATURES if f in df.columns]
        missing = [f for f in TRANSFORMER_FEATURES if f not in df.columns]
        if missing:
            logger.warning("Transformer: features missing (zero-filled): %s", missing)

        # Scale features
        self.scaler = StandardScaler()
        X_all  = self._extract(df)
        X_all  = self.scaler.fit_transform(X_all)

        # Build windowed sequences
        X_seq, y_seq = self._make_sequences(X_all, labels.values)
        logger.info(
            "Transformer training: %d sequences × (seq_len=%d, features=%d)",
            len(X_seq), self.cfg.seq_len, X_seq.shape[2]
        )

        # Class weights (handle imbalanced regimes)
        class_counts = np.bincount(y_seq, minlength=N_REGIMES).astype(float)
        class_weights = torch.tensor(
            1.0 / np.maximum(class_counts, 1), dtype=torch.float32
        )

        # Build model
        ModelClass = RegimeTransformer._get_model_class()
        self.model = ModelClass(n_features=X_seq.shape[2], cfg=self.cfg)
        self.model.to(self.cfg.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.cfg.device))
        optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr           = self.cfg.lr,
            weight_decay = self.cfg.weight_decay,
        )

        # Cosine LR schedule with warmup
        total_steps = (len(X_seq) // self.cfg.batch_size) * self.cfg.max_epochs
        scheduler   = self._build_scheduler(optimiser, total_steps)

        # Build validation sequences if provided
        X_val_seq, y_val_seq = None, None
        if val_df is not None and val_labels is not None:
            X_val = self.scaler.transform(self._extract(val_df))
            X_val_seq, y_val_seq = self._make_sequences(X_val, val_labels.values)

        best_val_loss = float("inf")
        patience_ctr  = 0
        best_state    = None

        for epoch in range(self.cfg.max_epochs):
            # ── Training ──
            self.model.train()
            train_loss = self._run_epoch(
                X_seq, y_seq, criterion, optimiser, scheduler, training=True
            )

            # ── Validation ──
            val_loss = None
            if X_val_seq is not None:
                self.model.eval()
                val_loss = self._run_epoch(
                    X_val_seq, y_val_seq, criterion, None, None, training=False
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_ctr  = 0
                    best_state    = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_ctr += 1

            if (epoch + 1) % 5 == 0:
                logger.info(
                    "Epoch %3d/%d | train_loss=%.4f%s",
                    epoch + 1, self.cfg.max_epochs, train_loss,
                    f" | val_loss={val_loss:.4f}" if val_loss else ""
                )

            if patience_ctr >= self.cfg.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.fitted = True
        logger.info("Transformer training complete. Best val loss: %.4f", best_val_loss)
        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime class for each bar."""
        proba = self.predict_proba(df)
        preds = proba.values.argmax(axis=1)
        return pd.Series(preds, index=proba.index, name="transformer_regime", dtype=int)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities for each bar.

        Returns
        -------
        pd.DataFrame shape (n_bars, 4) — columns: transformer_prob_{regime_name}
        Only bars with a full lookback window are predicted (first seq_len-1 bars are NaN).
        """
        import torch

        self._check_fitted()
        X = self.scaler.transform(self._extract(df))
        n = len(X)

        all_probs = np.full((n, N_REGIMES), np.nan)

        self.model.eval()
        with torch.no_grad():
            for i in range(self.cfg.seq_len - 1, n, self.cfg.batch_size):
                batch_end   = min(i + self.cfg.batch_size, n)
                batch_seqs  = []
                batch_indices = []
                for j in range(i, batch_end):
                    if j >= self.cfg.seq_len - 1:
                        seq = X[j - self.cfg.seq_len + 1: j + 1]
                        batch_seqs.append(seq)
                        batch_indices.append(j)

                if not batch_seqs:
                    continue

                x_tensor = torch.tensor(
                    np.array(batch_seqs), dtype=torch.float32
                ).to(self.cfg.device)
                logits = self.model(x_tensor)
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()

                for k, idx in enumerate(batch_indices):
                    all_probs[idx] = probs[k]

        cols = [f"transformer_prob_{REGIME_NAMES[i]}" for i in range(N_REGIMES)]
        return pd.DataFrame(all_probs, index=df.index, columns=cols)

    # ── Training helpers ──────────────────────────────────────────────────────
    def _run_epoch(
        self, X, y, criterion, optimiser, scheduler, training: bool
    ) -> float:
        import torch

        n          = len(X)
        indices    = np.random.permutation(n) if training else np.arange(n)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, n, self.cfg.batch_size):
            batch_idx = indices[start: start + self.cfg.batch_size]
            x_batch   = torch.tensor(X[batch_idx], dtype=torch.float32).to(self.cfg.device)
            y_batch   = torch.tensor(y[batch_idx], dtype=torch.long).to(self.cfg.device)

            if training:
                optimiser.zero_grad()
                # Data augmentation: add small Gaussian noise during training
                x_batch = x_batch + 0.01 * torch.randn_like(x_batch)

            logits = self.model(x_batch)
            loss   = criterion(logits, y_batch)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimiser.step()
                scheduler.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    @staticmethod
    def _build_scheduler(optimiser, total_steps: int):
        import torch
        warmup = 200

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slide a window of length seq_len over X to create sequences."""
        seqs, labels = [], []
        for i in range(self.cfg.seq_len - 1, len(X)):
            seqs.append(X[i - self.cfg.seq_len + 1: i + 1])
            labels.append(y[i])
        return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _extract(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and zero-fill feature matrix from DataFrame."""
        X = pd.DataFrame(index=df.index)
        for f in TRANSFORMER_FEATURES:
            X[f] = df[f] if f in df.columns else 0.0
        return X.fillna(0.0).values.astype(np.float64)

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        import torch
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler":      self.scaler,
            "config":      self.cfg,
            "features":    self._features,
        }, path)
        logger.info("Transformer saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "TransformerRegimeClassifier":
        import torch
        data     = torch.load(path, map_location="cpu", weights_only=False)
        obj      = cls(config=data["config"])
        obj.scaler    = data["scaler"]
        obj._features = data["features"]
        ModelClass    = RegimeTransformer._get_model_class()
        obj.model     = ModelClass(n_features=len(TRANSFORMER_FEATURES), cfg=obj.cfg)
        obj.model.load_state_dict(data["model_state"])
        obj.fitted = True
        logger.info("Transformer loaded from %s", path)
        return obj

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Classifier not fitted. Call .fit(df, labels) first.")
