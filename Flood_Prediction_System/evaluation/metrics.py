"""
evaluation/metrics.py

Small wrapper around sklearn's metric functions so every training module
reports numbers the exact same way instead of each one rolling its own
print statements. Used by decision_tree.py, random_forest.py, knn.py and
xgboost_model.py.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def summarize_predictions(y_true, y_pred, model_label=""):
    """Returns a dict of the usual suspects plus a printable report string."""
    scores = {
        "model": model_label,
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }
    return scores


def print_summary(scores):
    print(f"\n----- {scores['model']} -----")
    print(f"Accuracy  : {scores['accuracy']}%")
    print(f"Precision : {scores['precision']}%")
    print(f"Recall    : {scores['recall']}%")
    print(f"F1 Score  : {scores['f1_score']}%")
    print("Confusion Matrix:")
    for row in scores["confusion_matrix"]:
        print(row)
    print(scores["report"])
