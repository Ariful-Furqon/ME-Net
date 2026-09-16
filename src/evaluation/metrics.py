import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    pred = (y_prob >= 0.5).astype(int)
    conf = np.where(pred == 1, y_prob, 1.0 - y_prob)
    correct = (pred == y_true).astype(float)
    bins = np.minimum(((conf - 0.5) / 0.5 * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_classification_metrics(y_true, y_pred, y_prob=None, n_bins: int = 15) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    res = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro")),
        "Hoax_Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Hoax_Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Hoax_F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        res["ROC_AUC"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
        res["ECE"] = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
        res["Brier_Score"] = float(brier_score_loss(y_true, y_prob))
    return res
