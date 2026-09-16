import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from src.models.me_fusion import MorphoTextDataset, MorphoEvidentialNet, GatedCrossModalFusion
from src.evaluation.metrics import compute_classification_metrics

class NaiveConcatFusionNet(nn.Module):
    def __init__(self, vocab_size: int, ling_dim: int, embed_dim: int = 128, num_filters: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=5, padding="same")
        self.bn = nn.BatchNorm1d(num_filters)
        self.dropout = nn.Dropout(dropout)
        
        self.ling_encoder = nn.Sequential(
            nn.Linear(ling_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(num_filters + 64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq: torch.Tensor, x_ling: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds = self.embedding(x_seq).permute(0, 2, 1)
        h_conv = F.relu(self.bn(self.conv1d(embeds)))
        h_text = self.dropout(F.adaptive_max_pool1d(h_conv, 1).squeeze(-1))
        h_ling = self.ling_encoder(x_ling)
        h_fused = torch.cat([h_text, h_ling], dim=-1)
        logits = self.classifier(h_fused).squeeze(-1)
        return logits, h_fused

class Pure1DCNNNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_filters: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=5, padding="same")
        self.bn = nn.BatchNorm1d(num_filters)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq: torch.Tensor, x_ling: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds = self.embedding(x_seq).permute(0, 2, 1)
        h_conv = F.relu(self.bn(self.conv1d(embeds)))
        h_text = self.dropout(F.adaptive_max_pool1d(h_conv, 1).squeeze(-1))
        logits = self.classifier(h_text).squeeze(-1)
        return logits, h_text

class PureSymbolicMLP(nn.Module):
    def __init__(self, ling_dim: int = 13, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ling_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_seq: torch.Tensor = None, x_ling: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x_ling).squeeze(-1)
        return logits, x_ling

class AblationStudyRunner:
    def __init__(self, tokenizer, profiler, device: str = None):
        self.tokenizer = tokenizer
        self.profiler = profiler
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()

    def _train_and_eval_model(
        self,
        model: nn.Module,
        train_seqs: np.ndarray,
        train_ling: np.ndarray,
        train_labels: np.ndarray,
        test_seqs: np.ndarray,
        test_ling: np.ndarray,
        test_labels: np.ndarray,
        epochs: int = 5,
        batch_size: int = 64,
        lr: float = 1e-3
    ) -> Dict[str, float]:
        model = model.to(self.device)
        train_ds = MorphoTextDataset(train_seqs, train_ling, train_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_ds = MorphoTextDataset(test_seqs, test_ling, test_labels)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            model.train()
            for seqs, lings, labels in train_loader:
                seqs, lings, labels = seqs.to(self.device), lings.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits, _ = model(seqs, lings)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        all_probs = []
        with torch.no_grad():
            for seqs, lings, _ in test_loader:
                seqs, lings = seqs.to(self.device), lings.to(self.device)
                logits, _ = model(seqs, lings)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)

        all_probs = np.array(all_probs)
        preds = (all_probs >= 0.5).astype(int)
        self.last_probs = all_probs
        return compute_classification_metrics(test_labels, preds, y_prob=all_probs, n_bins=15)

    def run_full_ablation(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        test_texts: List[str],
        test_labels: np.ndarray,
        epochs: int = 5
    ) -> pd.DataFrame:
        train_seqs = self.tokenizer.texts_to_sequences(train_texts)
        test_seqs = self.tokenizer.texts_to_sequences(test_texts)
        vocab_size = len(self.tokenizer.word2idx)

        train_ling_df = self.profiler.profile_dataframe(pd.DataFrame({"Text": train_texts}))
        test_ling_df = self.profiler.profile_dataframe(pd.DataFrame({"Text": test_texts}))

        train_ling_scaled = self.scaler.fit_transform(train_ling_df.values)
        test_ling_scaled = self.scaler.transform(test_ling_df.values)
        ling_cols = list(train_ling_df.columns)

        # Feature subsets
        morph_cols = [c for c in ling_cols if any(k in c for k in ["prefix", "ke_an", "particle"])]
        morph_idx = [ling_cols.index(c) for c in morph_cols]

        evid_cols = [c for c in ling_cols if any(k in c for k in ["evidential", "attribution", "sensational", "ratio"])]
        evid_idx = [ling_cols.index(c) for c in evid_cols]

        variants = [
            ("Full ME-Net (Proposed)",
             lambda: MorphoEvidentialNet(vocab_size=vocab_size, ling_feature_dim=train_ling_scaled.shape[1]),
             train_ling_scaled, test_ling_scaled,
             "Optimal multi-modal balance and calibration."),

            ("w/o Evidentiality Priors",
             lambda: MorphoEvidentialNet(vocab_size=vocab_size, ling_feature_dim=len(morph_idx)),
             train_ling_scaled[:, morph_idx], test_ling_scaled[:, morph_idx],
             "Relies on morphological markers only."),

            ("w/o Morphological Priors",
             lambda: MorphoEvidentialNet(vocab_size=vocab_size, ling_feature_dim=len(evid_idx)),
             train_ling_scaled[:, evid_idx], test_ling_scaled[:, evid_idx],
             "Relies on evidential & sensational markers only."),

            ("Naive Concatenation (w/o Gate)",
             lambda: NaiveConcatFusionNet(vocab_size=vocab_size, ling_dim=train_ling_scaled.shape[1]),
             train_ling_scaled, test_ling_scaled,
             "Uncontrolled linear concatenation without dynamic gating."),

            ("Pure 1D-CNN (w/o Symbolic)",
             lambda: Pure1DCNNNet(vocab_size=vocab_size),
             train_ling_scaled, test_ling_scaled,
             "Lacks explicit symbolic linguistic priors."),

            ("Pure Symbolic MLP (13 feats only)",
             lambda: PureSymbolicMLP(ling_dim=train_ling_scaled.shape[1]),
             train_ling_scaled, test_ling_scaled,
             "Evaluates predictive power of linguistic features alone without text tokens.")
        ]

        records = []
        baseline_f1 = None

        for name, model_fn, t_ling, val_ling, finding in variants:
            print(f" > Running Ablation: {name}...")
            model = model_fn()
            res = self._train_and_eval_model(
                model, train_seqs, t_ling, train_labels,
                test_seqs, val_ling, test_labels, epochs=epochs
            )
            f1 = res["Macro_F1"]
            if baseline_f1 is None:
                baseline_f1 = f1
            delta = (f1 - baseline_f1) * 100

            records.append({
                "Architecture_Variant": name,
                "Macro_F1": f1,
                "Accuracy": res["Accuracy"],
                "ROC_AUC": res["ROC_AUC"],
                "ECE": res.get("ECE", 0.0),
                "Brier_Score": res.get("Brier_Score", 0.0),
                "Performance_Delta_F1": delta,
                "Scientific_Finding": finding
            })

        return pd.DataFrame(records)
