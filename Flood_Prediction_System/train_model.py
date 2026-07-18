"""
train_model.py

Single entry point for (re)training the flood prediction pipeline end to
end. Running this script is a prerequisite before starting the Flask app
for the first time, since app.py just loads whatever got saved to
models/floods.save.

    python train_model.py

Steps:
    1. Load + clean the raw dataset (preprocessing/data_prep.py)
    2. Train Decision Tree, Random Forest, KNN and XGBoost
    3. Compare their scores, save a bar chart to reports/
    4. Persist the best model to models/floods.save
    5. Persist the fitted StandardScaler to models/transform.save
    6. Dump a short text summary to reports/training_summary.txt
"""

import os
import json
import joblib
from datetime import datetime

from preprocessing.data_prep import build_train_test_sets, MODELS_DIR
from compare_models import run_all_models, print_comparison_table, save_comparison_chart, pick_best_model

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "floods.save")
SUMMARY_PATH = os.path.join(REPORTS_DIR, "training_summary.txt")


def write_training_summary(results, best_label, best_scores):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    lines = [
        "FLOOD PREDICTION - MODEL TRAINING SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 50,
    ]
    for label, (_, scores) in results.items():
        lines.append(
            f"{label:<16} accuracy={scores['accuracy']}%  "
            f"precision={scores['precision']}%  recall={scores['recall']}%  "
            f"f1={scores['f1_score']}%"
        )
    lines.append("-" * 50)
    lines.append(f"Selected for deployment: {best_label} ({best_scores['accuracy']}% accuracy)")

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines))


def main():
    print("Loading dataset and building train/test split ...")
    X_train, X_test, y_train, y_test, scaler = build_train_test_sets()
    print(f"Training rows: {len(X_train)}  |  Test rows: {len(X_test)}")

    print("\nTraining Decision Tree, Random Forest, KNN and XGBoost ...")
    results = run_all_models(X_train, X_test, y_train, y_test, verbose=True)

    print_comparison_table(results)
    save_comparison_chart(results)

    best_label, best_model, best_scores = pick_best_model(results)
    print(f"\n>>> Selected model for deployment: {best_label} ({best_scores['accuracy']}% accuracy)")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    print(f"Saved model      -> {MODEL_OUTPUT_PATH}")
    print(f"Saved scaler     -> {os.path.join(MODELS_DIR, 'transform.save')}")

    write_training_summary(results, best_label, best_scores)
    print(f"Saved summary    -> {SUMMARY_PATH}")

    # Small metadata file the Flask app reads to show "which model is live"
    # on the About page without having to introspect the pickled object.
    metadata = {
        "model_name": best_label,
        "accuracy": best_scores["accuracy"],
        "precision": best_scores["precision"],
        "recall": best_scores["recall"],
        "f1_score": best_scores["f1_score"],
        "trained_on_rows": int(len(X_train) + len(X_test)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nTraining pipeline complete. Start the app with: python app.py")


if __name__ == "__main__":
    main()
