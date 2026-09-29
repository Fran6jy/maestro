"""
maestro/agents/signal/tft_model.py
====================================
Temporal Fusion Transformer (TFT) for multi-horizon price direction
forecasting on EUR/USD and GBP/USD.

Why TFT over plain Transformer?
---------------------------------
The TFT (Lim et al., 2021) was specifically designed for time series
with mixed input types — exactly what we have:
  - Static covariates:  instrument ID, regime label (fixed per sequence)
  - Known future inputs: calendar features (hour, dow — always known ahead)
  - Observed inputs:    OHLCV, technical indicators (only known up to now)

TFT's key advantages over the MSc's linear regression:
  1. Variable Selection Networks  — learns which features matter per regime
  2. Gating mechanisms            — suppresses irrelevant information paths
  3. Multi-head attention         — captures long-range temporal dependencies
  4. Quantile outputs             — predicts P10/P50/P90, not just point estimates
  5. Interpretable attention      — tells us which past bars drove each prediction

Architecture internals
-----------------------
  Input → Variable Selection → LSTM encoder (past) → LSTM decoder (future)
       → Static enrichment   → Temporal Self-Attention → Point-wise FFN
       → Quantile output heads

Regime conditioning
--------------------
The regime label from Agent 1 is injected as a static covariate.
This means the TFT learns different temporal patterns per regime —
trending behaviour in bull markets, mean-reversion in sideways, etc.
This is the primary mechanism for beating the 37% hit ratio baseline.

References
----------
Lim, B. et al. (2021). Temporal Fusion Transformers for interpretable
  multi-horizon time series forecasting. Int. J. of Forecasting 37(4).
"""
from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _auto_device() -> str:
    """Pick CUDA when available, else CPU. Lazy torch import keeps module load light."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ── Feature definitions ───────────────────────────────────────────────────────
# Observed past inputs (unknown in future)
PAST_FEATURES = [
    "log_return_1", "log_return_5", "log_return_10",
    "vol_realised", "vol_gk", "atr_normalised",
    "rsi", "macd_histogram", "macd_cross",
    "bb_pct", "bb_width", "bb_squeeze",
    "adx", "adx_plus", "adx_minus", "trend_strong",
    "stoch_k", "stoch_d", "williams_r", "cci",
    "volume_zscore", "obv_trend",
    "body_ratio", "close_position", "bar_direction",
    "return_zscore_252",
    "vix_zscore", "yield_curve_inverted", "rate_change_1m",
    "cross_corr_20", "cross_spread_zscore",
]

# Known future inputs (calendar — always available ahead of time)
FUTURE_FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "session_overlap", "session_london",
]

# Static covariates (fixed for entire sequence)
STATIC_FEATURES = [
    "regime",          # from Agent 1 — integer {0,1,2,3}
    "instrument_id",   # integer encoding: EUR_USD=0, GBP_USD=1
]

# Prediction horizons (bars ahead)
HORIZONS = [1, 3, 6, 12]   # 5m data: 5min, 15min, 30min, 1hour


@dataclass
class TFTConfig:
    # Architecture
    hidden_size:       int   = 128      # core hidden dimension
    lstm_layers:       int   = 2        # LSTM encoder/decoder layers
    attention_heads:   int   = 4        # multi-head attention heads
    dropout:           float = 0.1
    hidden_continuous: int   = 32       # continuous variable processing width

    # Sequence
    seq_len:           int   = 120      # encoder lookback (120 × 5min = 10h)
    pred_len:          int   = 12       # max prediction horizon (1h ahead)
    quantiles:         list  = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Training
    lr:                float = 1e-3
    weight_decay:      float = 1e-4
    batch_size:        int   = 64
    max_epochs:        int   = 60
    patience:          int   = 10
    gradient_clip:     float = 0.1
    device:            str   = field(default_factory=_auto_device)

    # Regime conditioning
    regime_embed_dim:  int   = 8        # embedding size for regime categorical


class _GateAddNorm(object):
    """Gated Residual Network — the core building block of TFT."""
    _cls = None

    @classmethod
    def get(cls):
        if cls._cls is not None:
            return cls._cls
        import torch.nn as nn
        import torch

        class GRN(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1,
                         context_dim=None):
                super().__init__()
                self.fc1     = nn.Linear(input_dim, hidden_dim)
                self.fc2     = nn.Linear(hidden_dim, output_dim)
                self.gate    = nn.Linear(hidden_dim, output_dim)
                self.norm    = nn.LayerNorm(output_dim)
                self.dropout = nn.Dropout(dropout)
                self.skip    = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
                self.ctx_fc  = nn.Linear(context_dim, hidden_dim) if context_dim else None
                self.elu     = nn.ELU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x, context=None):
                residual = self.skip(x)
                h = self.fc1(x)
                if context is not None and self.ctx_fc is not None:
                    h = h + self.ctx_fc(context)
                h = self.elu(h)
                h = self.dropout(h)
                gate = self.sigmoid(self.gate(h))
                out  = gate * self.fc2(h)
                return self.norm(out + residual)

        cls._cls = GRN
        return cls._cls


class _TFTModule(object):
    """Lazy-loaded full TFT nn.Module."""
    _cls = None

    @classmethod
    def get(cls):
        if cls._cls is not None:
            return cls._cls
        import torch
        import torch.nn as nn
        GRN = _GateAddNorm.get()

        class TFT(nn.Module):
            def __init__(self, n_past, n_future, n_static, cfg: TFTConfig):
                super().__init__()
                H = cfg.hidden_size
                Q = len(cfg.quantiles)

                # ── Input projections ──
                self.past_proj   = nn.Linear(n_past,   H)
                self.future_proj = nn.Linear(n_future, H)
                self.static_proj = nn.Linear(n_static, H)

                # Regime embedding (categorical)
                self.regime_embed = nn.Embedding(4, cfg.regime_embed_dim)

                # ── Variable Selection Networks ──
                self.past_vsn   = GRN(H, H, H, cfg.dropout, context_dim=H)
                self.future_vsn = GRN(H, H, H, cfg.dropout, context_dim=H)

                # ── LSTM encoder / decoder ──
                self.encoder = nn.LSTM(
                    H, H, num_layers=cfg.lstm_layers,
                    batch_first=True, dropout=cfg.dropout if cfg.lstm_layers > 1 else 0
                )
                self.decoder = nn.LSTM(
                    H, H, num_layers=cfg.lstm_layers,
                    batch_first=True, dropout=cfg.dropout if cfg.lstm_layers > 1 else 0
                )

                # ── Static enrichment ──
                self.static_enrich = GRN(H, H, H, cfg.dropout, context_dim=H)

                # ── Temporal Self-Attention ──
                self.attn = nn.MultiheadAttention(
                    H, cfg.attention_heads, dropout=cfg.dropout, batch_first=True
                )
                self.attn_norm = nn.LayerNorm(H)

                # ── Position-wise FFN ──
                self.ffn = GRN(H, H * 4, H, cfg.dropout)

                # ── Output heads — one per quantile per horizon ──
                self.out_heads = nn.ModuleList([
                    nn.Linear(H, Q) for _ in range(cfg.pred_len)
                ])
                self.cfg = cfg

            def forward(self, past, future, static, regime):
                B = past.shape[0]
                # Static context
                static_ctx  = self.static_proj(static)                 # (B, H)
                regime_emb  = self.regime_embed(regime.long())         # (B, embed_dim)

                # Past sequence
                past_h      = self.past_proj(past)                     # (B, T, H)
                past_sel    = self.past_vsn(
                    past_h,
                    context=static_ctx.unsqueeze(1).expand(-1, past_h.size(1), -1)
                )
                enc_out, (hn, cn) = self.encoder(past_sel)

                # Future sequence
                fut_h       = self.future_proj(future)                 # (B, pred_len, H)
                fut_sel     = self.future_vsn(
                    fut_h,
                    context=static_ctx.unsqueeze(1).expand(-1, fut_h.size(1), -1)
                )
                dec_out, _  = self.decoder(fut_sel, (hn, cn))

                # Static enrichment on decoder output
                enriched    = self.static_enrich(dec_out, context=static_ctx.unsqueeze(1).expand_as(dec_out))

                # Temporal self-attention (decoder attends to encoder output)
                attn_out, attn_weights = self.attn(enriched, enc_out, enc_out)
                attn_out    = self.attn_norm(attn_out + enriched)

                # Point-wise FFN
                ffn_out     = self.ffn(attn_out)                       # (B, pred_len, H)

                # Output: one quantile prediction per horizon
                outputs     = []
                for i, head in enumerate(self.out_heads):
                    if i < ffn_out.size(1):
                        outputs.append(head(ffn_out[:, i, :]))         # (B, Q)
                    else:
                        outputs.append(head(ffn_out[:, -1, :]))

                return (
                    torch.stack(outputs, dim=1),  # (B, pred_len, Q)
                    attn_weights                  # (B, pred_len, seq_len) — for XAI
                )

        cls._cls = TFT
        return cls._cls


# ── Main signal model ─────────────────────────────────────────────────────────
class TFTSignalModel:
    """
    TFT-based directional signal generator.

    Usage
    -----
    >>> model = TFTSignalModel()
    >>> model.fit(train_df, regime_series)
    >>> signals = model.predict(test_df, regime_series)
    >>> attention = model.get_attention_weights(test_df, regime_series)

    Output signals
    --------------
    pd.DataFrame with columns per horizon:
      signal_{h}     : {-1, 0, +1} — directional call
      confidence_{h} : float — quantile spread (P90-P10), wider = more uncertain
      pred_p10_{h}   : 10th percentile predicted return
      pred_p50_{h}   : median predicted return
      pred_p90_{h}   : 90th percentile predicted return
    """

    def __init__(self, config: TFTConfig | None = None) -> None:
        self.cfg     = config or TFTConfig()
        self.model   = None
        self.scaler_past   = StandardScaler()
        self.scaler_future = StandardScaler()
        # Std of the cumulative k-bar return in training, per step k. Targets are
        # divided by it so the network works in unit-variance space; raw 5-minute
        # returns (~1e-4) let it cut the loss fastest by ignoring its inputs.
        self.target_scale  = np.ones(self.cfg.pred_len, dtype=np.float32)
        self.fitted  = False
        self._n_past   = len(PAST_FEATURES)
        self._n_future = len(FUTURE_FEATURES)
        self._n_static = 2   # instrument_id + regime (after embedding)

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(
        self,
        df:          pd.DataFrame,
        regimes:     pd.Series,
        instrument:  int = 0,
        val_df:      pd.DataFrame | None = None,
        val_regimes: pd.Series | None    = None,
    ) -> "TFTSignalModel":
        """
        Train the TFT on historical features + regime labels.

        Parameters
        ----------
        df         : feature DataFrame with PAST_FEATURES + FUTURE_FEATURES columns
        regimes    : pd.Series[int] from RegimeDetectionAgent, same index as df
        instrument : 0=EUR_USD, 1=GBP_USD
        val_df     : optional validation set for early stopping
        val_regimes: regime labels for val_df
        """
        import torch
        import torch.nn as nn

        logger.info("Fitting TFT Signal Model | bars=%d | seq=%d | horizons=%s",
                    len(df), self.cfg.seq_len, HORIZONS)

        # ── Build dataset ──
        past_arr, fut_arr, static_arr, reg_arr, target_arr = self._prepare(
            df, regimes, instrument, fit_scalers=True
        )
        self.target_scale = np.maximum(target_arr.std(axis=0), 1e-8).astype(np.float32)
        target_arr = target_arr / self.target_scale

        TFTModule = _TFTModule.get()
        self.model = TFTModule(
            n_past   = past_arr.shape[-1],
            n_future = fut_arr.shape[-1],
            n_static = static_arr.shape[-1],
            cfg      = self.cfg,
        ).to(self.cfg.device)

        optimiser = torch.optim.Adam(
            self.model.parameters(),
            lr           = self.cfg.lr,
            weight_decay = self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=3, factor=0.5
        )

        best_loss  = float("inf")
        patience   = 0
        best_state = None

        for epoch in range(self.cfg.max_epochs):
            train_loss = self._run_epoch(
                past_arr, fut_arr, static_arr, reg_arr, target_arr,
                optimiser, training=True
            )

            val_loss = None
            if val_df is not None and val_regimes is not None:
                vp, vf, vs, vr, vt = self._prepare(
                    val_df, val_regimes, instrument, fit_scalers=False
                )
                val_loss = self._run_epoch(vp, vf, vs, vr, vt / self.target_scale, None, training=False)
                scheduler.step(val_loss)
                monitor = val_loss
            else:
                monitor = train_loss

            if monitor < best_loss:
                best_loss  = monitor
                patience   = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1

            if (epoch + 1) % 10 == 0:
                logger.info("  Epoch %3d | train=%.5f%s",
                            epoch + 1, train_loss,
                            f" | val={val_loss:.5f}" if val_loss else "")

            if patience >= self.cfg.patience:
                logger.info("  Early stop at epoch %d", epoch + 1)
                break

        if best_state:
            self.model.load_state_dict(best_state)

        self.fitted = True
        logger.info("TFT training complete. Best loss: %.5f", best_loss)
        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(
        self,
        df:         pd.DataFrame,
        regimes:    pd.Series,
        instrument: int = 0,
        threshold:  float = 0.0002,   # minimum |P50 return| to issue signal
    ) -> pd.DataFrame:
        """
        Generate directional signals for each bar.

        Parameters
        ----------
        df        : feature DataFrame
        regimes   : regime labels from Agent 1
        threshold : minimum predicted return magnitude to issue non-zero signal
                    (filters out near-flat predictions → label 0)

        Returns
        -------
        pd.DataFrame — one row per bar, columns:
            signal_{h}, confidence_{h}, pred_p10_{h}, pred_p50_{h}, pred_p90_{h}
            + attention_entropy (mean attention entropy — XAI signal)
        """
        import torch

        self._check_fitted()
        past_arr, fut_arr, static_arr, reg_arr, _ = self._prepare(
            df, regimes, instrument, fit_scalers=False
        )

        n           = len(past_arr)
        all_preds   = np.full((n, self.cfg.pred_len, len(self.cfg.quantiles)), np.nan)
        all_attn_entropy = np.full(n, np.nan)

        self.model.eval()
        with torch.no_grad():
            for start in range(0, n, self.cfg.batch_size):
                end    = min(start + self.cfg.batch_size, n)
                p      = torch.tensor(past_arr[start:end],   dtype=torch.float32).to(self.cfg.device)
                f      = torch.tensor(fut_arr[start:end],    dtype=torch.float32).to(self.cfg.device)
                s      = torch.tensor(static_arr[start:end], dtype=torch.float32).to(self.cfg.device)
                r      = torch.tensor(reg_arr[start:end],    dtype=torch.long).to(self.cfg.device)

                preds, attn = self.model(p, f, s, r)   # (B, pred_len, Q), (B, pred_len, T)
                all_preds[start:end] = preds.cpu().numpy() * self.target_scale[None, :, None]

                # Attention entropy (lower = more focused = more confident)
                attn_np = attn.cpu().numpy()
                eps     = 1e-8
                entropy = -(attn_np * np.log(attn_np + eps)).sum(axis=-1).mean(axis=-1)
                all_attn_entropy[start:end] = entropy

        # Build output DataFrame. Window i ends at bar i + seq_len - 1: that is the
        # bar the forecast is made at (its close is the last thing the model saw),
        # so the forecast is labelled there. Step k is the cumulative return over
        # the next k + 1 bars, so horizon h is step h - 1.
        result = pd.DataFrame(index=df.index[self.cfg.seq_len - 1 : self.cfg.seq_len - 1 + n])
        for h in HORIZONS:
            if h > self.cfg.pred_len:
                break
            p10 = all_preds[:, h - 1, 0]
            p50 = all_preds[:, h - 1, 1]
            p90 = all_preds[:, h - 1, 2]

            # Directional confidence (scale-invariant) derived from the quantile
            # forecast. Under a normal approximation P90-P10 ≈ 2.5631·σ, so
            # σ̂ = spread / 2.5631, and z = |P50|/σ̂ is the signal-to-noise ratio.
            # The model's implied probability that the return has the predicted
            # sign is Φ(z); we report the *edge over chance* = 2·Φ(z) − 1 ∈ [0,1].
            #   • large |P50| relative to spread → high confidence
            #   • P50 ≈ 0 relative to spread     → confidence ≈ 0
            from scipy.special import erf
            spread = np.maximum(p90 - p10, 1e-9)
            sigma  = spread / 2.5631
            z      = np.abs(p50) / sigma
            conf   = np.clip(2.0 * (0.5 * (1.0 + erf(z / np.sqrt(2.0)))) - 1.0, 0.0, 1.0)

            # Direction signal: sign of P50, filtered by threshold
            sig = np.where(p50 >  threshold,  1,
                  np.where(p50 < -threshold, -1, 0))

            result[f"signal_{h}"]    = sig
            result[f"pred_p10_{h}"]  = p10
            result[f"pred_p50_{h}"]  = p50
            result[f"pred_p90_{h}"]  = p90
            result[f"confidence_{h}"]= conf

        result["attn_entropy"] = all_attn_entropy
        return result

    def get_attention_weights(
        self,
        df:         pd.DataFrame,
        regimes:    pd.Series,
        instrument: int = 0,
    ) -> np.ndarray:
        """
        Return raw attention weights for XAI analysis.
        Shape: (n_samples, pred_len, seq_len)
        Used by xai/shap_engine.py to identify which past bars mattered.
        """
        import torch
        self._check_fitted()
        past_arr, fut_arr, static_arr, reg_arr, _ = self._prepare(
            df, regimes, instrument, fit_scalers=False
        )
        all_attn = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(past_arr), self.cfg.batch_size):
                end = min(start + self.cfg.batch_size, len(past_arr))
                p = torch.tensor(past_arr[start:end], dtype=torch.float32).to(self.cfg.device)
                f = torch.tensor(fut_arr[start:end],  dtype=torch.float32).to(self.cfg.device)
                s = torch.tensor(static_arr[start:end], dtype=torch.float32).to(self.cfg.device)
                r = torch.tensor(reg_arr[start:end],  dtype=torch.long).to(self.cfg.device)
                _, attn = self.model(p, f, s, r)
                all_attn.append(attn.cpu().numpy())
        return np.concatenate(all_attn, axis=0)

    # ── Data preparation ──────────────────────────────────────────────────────
    def _prepare(
        self,
        df: pd.DataFrame,
        regimes: pd.Series,
        instrument: int,
        fit_scalers: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build windowed arrays for TFT training/inference.

        Returns
        -------
        past_arr   : (n_samples, seq_len, n_past_features)
        fut_arr    : (n_samples, pred_len, n_future_features)
        static_arr : (n_samples, n_static)
        reg_arr    : (n_samples,) — regime integer for embedding
        target_arr : (n_samples, pred_len) — cumulative future log-return over 1..pred_len bars
        """
        # Extract feature matrices
        past_df   = _extract_cols(df, PAST_FEATURES)
        future_df = _extract_cols(df, FUTURE_FEATURES)

        if fit_scalers:
            past_mat   = self.scaler_past.fit_transform(past_df.fillna(0).values)
            future_mat = self.scaler_future.fit_transform(future_df.fillna(0).values)
        else:
            past_mat   = self.scaler_past.transform(past_df.fillna(0).values)
            future_mat = self.scaler_future.transform(future_df.fillna(0).values)

        returns    = df["log_return_1"].fillna(0).values
        reg_arr_full = regimes.reindex(df.index).fillna(2).values.astype(int)

        T     = self.cfg.seq_len
        H     = self.cfg.pred_len
        n     = len(df) - T - H + 1

        past_seqs   = np.zeros((n, T,    past_mat.shape[1]),   dtype=np.float32)
        fut_seqs    = np.zeros((n, H,    future_mat.shape[1]), dtype=np.float32)
        static_seqs = np.zeros((n, 2),                         dtype=np.float32)
        reg_seqs    = np.zeros(n,                              dtype=np.int64)
        target_seqs = np.zeros((n, H),                         dtype=np.float32)

        for i in range(n):
            t_start  = i
            t_end    = i + T
            f_end    = t_end + H
            past_seqs[i]   = past_mat[t_start:t_end]
            fut_seqs[i]    = future_mat[t_end:f_end]
            static_seqs[i] = [instrument, reg_arr_full[t_end - 1]]
            reg_seqs[i]    = reg_arr_full[t_end - 1]
            target_seqs[i] = np.cumsum(returns[t_end:f_end])

        return past_seqs, fut_seqs, static_seqs, reg_seqs, target_seqs

    def _run_epoch(
        self, past, fut, static, reg, targets, optimiser, training: bool
    ) -> float:
        import torch
        import torch.nn as nn

        n      = len(past)
        idx    = np.random.permutation(n) if training else np.arange(n)
        losses = []

        if training:
            self.model.train()
        else:
            self.model.eval()

        ctx = torch.no_grad() if not training else torch.enable_grad()
        with ctx:
            for start in range(0, n, self.cfg.batch_size):
                batch = idx[start: start + self.cfg.batch_size]
                p  = torch.tensor(past[batch],    dtype=torch.float32).to(self.cfg.device)
                f  = torch.tensor(fut[batch],     dtype=torch.float32).to(self.cfg.device)
                s  = torch.tensor(static[batch],  dtype=torch.float32).to(self.cfg.device)
                r  = torch.tensor(reg[batch],     dtype=torch.long).to(self.cfg.device)
                y  = torch.tensor(targets[batch], dtype=torch.float32).to(self.cfg.device)

                preds, _ = self.model(p, f, s, r)   # (B, pred_len, Q)
                loss     = _quantile_loss(preds, y, self.cfg.quantiles)

                if training and optimiser:
                    optimiser.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.gradient_clip
                    )
                    optimiser.step()

                losses.append(loss.item())

        return float(np.mean(losses)) if losses else 0.0

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        import torch
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":    self.model.state_dict(),
            "scaler_past":    self.scaler_past,
            "scaler_future":  self.scaler_future,
            "config":         self.cfg,
            "n_past":         self._n_past,
            "n_future":       self._n_future,
            "target_scale":   self.target_scale,
        }, path)
        logger.info("TFT saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "TFTSignalModel":
        import torch
        data = torch.load(path, map_location="cpu", weights_only=False)
        obj  = cls(config=data["config"])
        TFTModule = _TFTModule.get()
        obj.model = TFTModule(
            n_past   = data["n_past"],
            n_future = data["n_future"],
            n_static = 2,
            cfg      = data["config"],
        )
        obj.model.load_state_dict(data["model_state"])
        # Re-resolve device for the current machine and move the model onto it.
        obj.cfg.device = _auto_device()
        obj.model.to(obj.cfg.device)
        obj.scaler_past   = data["scaler_past"]
        obj.scaler_future = data["scaler_future"]
        obj.target_scale  = data.get("target_scale", np.ones(obj.cfg.pred_len, dtype=np.float32))
        obj.fitted = True
        logger.info("TFT loaded from %s (device=%s)", path, obj.cfg.device)
        return obj

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("TFT not fitted. Call .fit() first.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Extract columns, zero-filling any that are missing."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = df[c] if c in df.columns else 0.0
    return out


def _quantile_loss(preds: "torch.Tensor", targets: "torch.Tensor",
                   quantiles: list[float]) -> "torch.Tensor":
    """
    Pinball (quantile) loss — trains each output head to predict
    the correct quantile of the return distribution.
    """
    import torch
    losses = []
    for i, q in enumerate(quantiles):
        pred_q = preds[:, :, i]
        err    = targets - pred_q
        losses.append(torch.max((q - 1) * err, q * err).mean())
    return torch.stack(losses).mean()
