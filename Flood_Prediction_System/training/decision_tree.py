"""
training/decision_tree.py

Wraps sklearn's DecisionTreeClassifier with the train / predict / evaluate
trio that all four model modules share. Depth is capped deliberately -
letting a tree grow unrestricted on a 115-row dataset just memorises the
training set.
"""

from sklearn.tree import DecisionTreeClassifier
from evaluation.metrics import summarize_predictions


def train(X_train, y_train, max_depth=5, random_state=42):
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=2,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    return model.predict(X)


def evaluate(model, X_test, y_test):
    predictions = predict(model, X_test)
    return summarize_predictions(y_test, predictions, model_label="Decision Tree")
