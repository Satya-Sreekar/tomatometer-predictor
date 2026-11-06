"""Shared metric calculation, used by both train_models.py and app.py."""
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)


def evaluate(model, X, y):
    """The six assignment metrics. Multi-class -> macro averages, AUC is one-vs-rest (nan if a class is absent from y)."""
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=model.classes_),
        "Precision": precision_score(y, pred, average="macro", zero_division=0),
        "Recall": recall_score(y, pred, average="macro", zero_division=0),
        "F1": f1_score(y, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
