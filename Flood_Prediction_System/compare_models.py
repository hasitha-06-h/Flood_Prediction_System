"""
compare_models.py

Trains all four classifiers on the same train/test split, prints a
comparison table, saves a bar chart of accuracy scores into reports/,
and returns the scoreboard so train_model.py can decide which model to
ship. Kept separate from train_model.py so it can also be run on its own
just to eyeball how the models stack up without touching models/floods.save.
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless - no display available on a server / CI box
import matplotlib.pyplot as plt

from preprocessing.data_prep import build_train_test_sets
from evaluation.metrics import print_summary
from training import decision_tree, random_forest, knn

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def run_all_models(X_train, X_test, y_train, y_test, verbose=True):
    """Trains every model and returns {model_label: (fitted_model, scores_dict)}"""
    registry = {
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
        "KNN": knn,
    }

    # XGBoost is imported lazily and wrapped in a try/except - on some free
    # hosting tiers the wheel occasionally fails to build, and the whole
    # training pipeline shouldn't die just because one of four algorithms
    # is unavailable. If it can't be imported, the comparison simply runs
    # with the remaining three models.
    try:
        from training import xgboost_model
        registry["XGBoost"] = xgboost_model
    except ImportError:
        print("[warning] xgboost is not installed - skipping this model. "
              "Run 'pip install xgboost' to include it in the comparison.")

    results = {}
    for label, module in registry.items():
        fitted_model = module.train(X_train, y_train)
        scores = module.evaluate(fitted_model, X_test, y_test)
        results[label] = (fitted_model, scores)
        if verbose:
            print_summary(scores)

    return results


def print_comparison_table(results):
    header = f"{'Model':<18}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1 Score':>10}"
    print("\n" + "=" * len(header))
    print("MODEL COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for label, (_, scores) in results.items():
        print(
            f"{label:<18}{scores['accuracy']:>9}%{scores['precision']:>11}%"
            f"{scores['recall']:>9}%{scores['f1_score']:>9}%"
        )
    print("=" * len(header))


def save_comparison_chart(results, filename="model_comparison.png"):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    labels = list(results.keys())
    accuracies = [results[label][1]["accuracy"] for label in labels]

    colors = ["#4361ee", "#3a0ca3", "#4895ef", "#4cc9f0"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, accuracies, color=colors[: len(labels)])

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc}%", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Flood Prediction - Model Accuracy Comparison")
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = os.path.join(REPORTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved comparison chart -> {out_path}")


def pick_best_model(results):
    """Highest accuracy wins; ties broken by F1 score, then prefers XGBoost
    since boosted trees generalise better on unseen weather patterns."""
    def sort_key(item):
        label, (_, scores) = item
        is_xgboost = 1 if label == "XGBoost" else 0
        return (scores["accuracy"], scores["f1_score"], is_xgboost)

    best_label, (best_model, best_scores) = max(results.items(), key=sort_key)
    return best_label, best_model, best_scores


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = build_train_test_sets()
    results = run_all_models(X_train, X_test, y_train, y_test)
    print_comparison_table(results)
    save_comparison_chart(results)

    best_label, _, best_scores = pick_best_model(results)
    print(f"\nBest performing model: {best_label} ({best_scores['accuracy']}% accuracy)")
