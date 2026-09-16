import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Any
from sklearn.preprocessing import StandardScaler
from src.evaluation.metrics import compute_classification_metrics

class MorphoTextDataset(Dataset):
    def __init__(self, sequences: np.ndarray, linguistic_feats: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.linguistic_feats = torch.tensor(linguistic_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.linguistic_feats[idx], self.labels[idx]

class GatedCrossModalFusion(nn.Module):
    def __init__(self, text_dim: int, ling_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.proj_text = nn.Linear(text_dim, hidden_dim)
        self.proj_ling = nn.Linear(ling_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.fusion_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, h_text: torch.Tensor, h_ling: torch.Tensor) -> torch.Tensor:
        # h_text: (batch, text_dim), h_ling: (batch, ling_dim)
        z_t = F.relu(self.proj_text(h_text))
        z_l = F.relu(self.proj_ling(h_ling))
        
        # Calculate dynamic information gate
        concat_z = torch.cat([z_t, z_l], dim=-1)
        g = self.gate(concat_z)
        
        # Modulate representations
        z_fused = torch.cat([g * z_t, (1 - g) * z_l], dim=-1)
        out = self.fusion_out(z_fused)
        return out

class MorphoEvidentialNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ling_feature_dim: int = 16,
        embed_dim: int = 128,
        num_filters: int = 128,
        kernel_size: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        # 1. Contextual Text Stream
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=kernel_size, padding="same")
        self.bn = nn.BatchNorm1d(num_filters)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Symbolic Morpho-Evidential Stream
        self.ling_encoder = nn.Sequential(
            nn.Linear(ling_feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 3. Gated Cross-Modal Fusion
        self.fusion = GatedCrossModalFusion(text_dim=num_filters, ling_dim=64, hidden_dim=hidden_dim)
        
        # 4. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq: torch.Tensor, x_ling: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Contextual Text Stream
        embeds = self.embedding(x_seq).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        h_conv = F.relu(self.bn(self.conv1d(embeds)))
        h_text = F.adaptive_max_pool1d(h_conv, 1).squeeze(-1)  # (batch, num_filters)
        h_text = self.dropout(h_text)
        
        # Symbolic Morpho-Evidential Stream
        h_ling = self.ling_encoder(x_ling)  # (batch, 64)
        
        # Fusion Representation
        h_fused = self.fusion(h_text, h_ling)  # (batch, hidden_dim)
        
        # Binary Logits
        logits = self.classifier(h_fused).squeeze(-1)
        return logits, h_fused

class MorphoEvidentialTrainer:
    def __init__(
        self,
        tokenizer,
        profiler,
        ling_dim: int = 16,
        embed_dim: int = 128,
        lr: float = 1e-3,
        device: str = None
    ):
        self.tokenizer = tokenizer
        self.profiler = profiler
        self.ling_dim = ling_dim
        self.embed_dim = embed_dim
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()

    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: List[str] = None,
        val_labels: np.ndarray = None,
        epochs: int = 6,
        batch_size: int = 64
    ) -> Dict[str, List[float]]:
        # Extract features
        train_seqs = self.tokenizer.texts_to_sequences(train_texts)
        train_ling_df = self.profiler.profile_dataframe(pd.DataFrame({"Text": train_texts}))
        train_ling = self.scaler.fit_transform(train_ling_df.values)

        actual_vocab = len(self.tokenizer.word2idx)
        self.ling_dim = train_ling.shape[1]
        self.model = MorphoEvidentialNet(
            vocab_size=actual_vocab,
            ling_feature_dim=self.ling_dim,
            embed_dim=self.embed_dim
        ).to(self.device)

        train_ds = MorphoTextDataset(train_seqs, train_ling, train_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        history = {"train_loss": [], "val_f1": []}

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for seqs, lings, labels in train_loader:
                seqs = seqs.to(self.device)
                lings = lings.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(seqs, lings)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(labels)

            epoch_loss = running_loss / len(train_ds)
            history["train_loss"].append(epoch_loss)

            if val_texts is not None and val_labels is not None:
                val_res = self.evaluate(val_texts, val_labels, batch_size=batch_size)
                history["val_f1"].append(val_res["Macro_F1"])

        return history

    def evaluate(self, test_texts: List[str], test_labels: np.ndarray, batch_size: int = 64) -> Dict[str, float]:
        self.model.eval()
        test_seqs = self.tokenizer.texts_to_sequences(test_texts)
        test_ling_df = self.profiler.profile_dataframe(pd.DataFrame({"Text": test_texts}))
        test_ling = self.scaler.transform(test_ling_df.values)

        test_ds = MorphoTextDataset(test_seqs, test_ling, test_labels)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for seqs, lings, _ in test_loader:
                seqs = seqs.to(self.device)
                lings = lings.to(self.device)
                logits, _ = self.model(seqs, lings)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)

        all_probs = np.array(all_probs)
        preds = (all_probs >= 0.5).astype(int)
        self.last_probs = all_probs

        return compute_classification_metrics(test_labels, preds, y_prob=all_probs, n_bins=15)
