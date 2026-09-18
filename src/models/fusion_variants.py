from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.preprocessing import QuantileTransformer, StandardScaler


class FeatureConditioner:

    def __init__(self, mode: str = "rankgauss", n_quantiles: int = 1000, random_state: int = 0):
        if mode not in ("standard", "log", "rankgauss"):
            raise ValueError(mode)
        self.mode = mode
        self.n_quantiles = n_quantiles
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "FeatureConditioner":
        X = np.asarray(X, dtype=np.float64)
        self.nonneg_ = (X >= 0).all(axis=0)
        Z = self._pre(X)
        if self.mode == "rankgauss":
            self.qt_ = QuantileTransformer(n_quantiles=min(self.n_quantiles, len(Z)),
                                           output_distribution="normal",
                                           subsample=10 ** 7, random_state=self.random_state).fit(Z)
            Z = self.qt_.transform(Z)
        self.scaler_ = StandardScaler().fit(Z)
        return self

    def _pre(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "standard":
            return X
        Z = X.copy()
        Z[:, self.nonneg_] = np.log1p(Z[:, self.nonneg_])
        return Z

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self._pre(np.asarray(X, dtype=np.float64))
        if self.mode == "rankgauss":
            Z = self.qt_.transform(Z)
        return self.scaler_.transform(Z).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class FlexibleFusion(nn.Module):

    def __init__(self, text_dim: int, ling_dim: int, hidden_dim: int = 128,
                 gate_mode: str = "complementary", gate_on_features: bool = False,
                 raw_feature_dim: int = 0, gate_temperature: float = 1.0):
        super().__init__()
        self.gate_mode = gate_mode
        self.gate_on_features = gate_on_features
        self.temperature = gate_temperature
        self.proj_text = nn.Linear(text_dim, hidden_dim)
        self.proj_ling = nn.Linear(ling_dim, hidden_dim)

        gate_in = hidden_dim * 2 + (raw_feature_dim if gate_on_features else 0)
        if gate_mode == "independent":
            gate_out = hidden_dim * 2
        elif gate_mode == "convex_scalar":
            gate_out = 1
        else:
            gate_out = hidden_dim
        if gate_mode != "none":
            self.gate = nn.Linear(gate_in, gate_out)
            nn.init.zeros_(self.gate.bias)
            nn.init.normal_(self.gate.weight, std=0.05)

        # The convex modes mix the streams by summation, so the fused vector keeps
        # the hidden width instead of doubling it. That is the whole point: under
        # concatenation a constant gate is absorbed into this layer's weights
        # (W1 diag(g) is just another learnable matrix), so collapsing costs the
        # model nothing. Summing makes a constant gate destroy information.
        fused_dim = hidden_dim if gate_mode in ("convex", "convex_scalar") else hidden_dim * 2
        self.fusion_out = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.last_gate = None

    def forward(self, h_text: torch.Tensor, h_ling: torch.Tensor,
                raw_feats: torch.Tensor = None) -> torch.Tensor:
        z_t = F.relu(self.proj_text(h_text))
        z_l = F.relu(self.proj_ling(h_ling))
        if self.gate_mode == "none":
            self.last_gate = None
            return self.fusion_out(torch.cat([z_t, z_l], dim=-1))

        gate_in = torch.cat([z_t, z_l], dim=-1)
        if self.gate_on_features and raw_feats is not None:
            gate_in = torch.cat([gate_in, raw_feats], dim=-1)
        pre = self.gate(gate_in) / self.temperature

        if self.gate_mode in ("convex", "convex_scalar"):
            g = torch.sigmoid(pre)          # (B, H) or (B, 1), broadcast over H
            self.last_gate = g
            fused = g * z_t + (1.0 - g) * z_l
        elif self.gate_mode == "independent":
            g_t, g_l = torch.sigmoid(pre).chunk(2, dim=-1)
            self.last_gate = g_t
            fused = torch.cat([g_t * z_t, g_l * z_l], dim=-1)
        elif self.gate_mode == "residual":
            g = torch.sigmoid(pre)
            self.last_gate = g
            fused = torch.cat([z_t, z_l + g * z_t], dim=-1)
        else:
            g = torch.sigmoid(pre)
            self.last_gate = g
            fused = torch.cat([g * z_t, (1.0 - g) * z_l], dim=-1)
        return self.fusion_out(fused)


class MENetV2(nn.Module):

    def __init__(self, vocab_size: int, ling_feature_dim: int = 13, embed_dim: int = 128,
                 num_filters: int = 128, kernel_size: int = 5, hidden_dim: int = 128,
                 dropout: float = 0.3, gate_mode: str = "complementary",
                 gate_on_features: bool = False, gate_temperature: float = 1.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(embed_dim, num_filters, kernel_size=kernel_size, padding="same")
        self.bn = nn.BatchNorm1d(num_filters)
        self.dropout = nn.Dropout(dropout)
        self.ling_encoder = nn.Sequential(
            nn.Linear(ling_feature_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.fusion = FlexibleFusion(num_filters, 64, hidden_dim, gate_mode=gate_mode,
                                     gate_on_features=gate_on_features,
                                     raw_feature_dim=ling_feature_dim,
                                     gate_temperature=gate_temperature)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_ling: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds = self.embedding(x_seq).permute(0, 2, 1)
        h_conv = F.relu(self.bn(self.conv1d(embeds)))
        h_text = self.dropout(F.adaptive_max_pool1d(h_conv, 1).squeeze(-1))
        h_ling = self.ling_encoder(x_ling)
        h_fused = self.fusion(h_text, h_ling, raw_feats=x_ling)
        return self.classifier(h_fused).squeeze(-1), h_fused

    def gate_variance_penalty(self) -> torch.Tensor:
        g = self.fusion.last_gate
        if g is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return -g.var(dim=0).mean()
