"""
training/knn.py

K-Nearest Neighbours classifier. This one is the most sensitive to
feature scale out of the four, which is exactly why the scaler produced
in preprocessing/data_prep.py gets applied before any of the models see
the data.
"""

from sklearn.neighbors import KNeighborsClassifier
from evaluation.metrics import summarize_predictions


def train(X_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    return model.predict(X)


def evaluate(model, X_test, y_test):
    predictions = predict(model, X_test)
    return summarize_predictions(y_test, predictions, model_label="KNN")
