"""Train, compare and evaluate models using chronological term splits."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from src.generate_data import generate_dataset, FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_performance.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SEED = 42
TARGET = "at_risk"

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply domain-range checks; imputation is learned inside each model pipeline."""
    x = df[FEATURES].copy()
    x["attendance_rate"] = x["attendance_rate"].clip(0, 100)
    x["assignment_completion"] = x["assignment_completion"].clip(0, 100)
    x["previous_avg_score"] = x["previous_avg_score"].clip(0, 100)
    x["score_trend"] = x["score_trend"].clip(-30, 30)
    x["missed_assignments"] = x["missed_assignments"].clip(0, 20)
    x["study_hours_per_week"] = x["study_hours_per_week"].clip(0, 40)
    return x

def build_pipeline(estimator, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)

def metric_row(name, y_true, y_pred):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "figures").mkdir(exist_ok=True)

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_dataset(seed=SEED)
        df.to_csv(DATA_PATH, index=False)

    # EDA / data-quality checks.
    print("Shape:", df.shape)
    print("\\nMissing values:\\n", df.isna().sum())
    print("\\nTarget balance:\\n", df[TARGET].value_counts(normalize=True))
    print("\\nDuplicate rows:", df.duplicated().sum())

    # Predictors are from the current snapshot; next score and target are excluded.
    X = clean_features(df)
    y = df[TARGET].astype(int)
    terms = df["term"]

    train_mask = terms <= 2
    val_mask = terms == 3
    test_mask = terms == 4
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    candidates = {
        "Dummy baseline": build_pipeline(DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED),
            scale=True
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=250, max_depth=8, min_samples_leaf=4,
                class_weight="balanced", random_state=SEED, n_jobs=-1
            )
        ),
    }

    validation_rows = []
    fitted = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        pred = pipeline.predict(X_val)
        validation_rows.append(metric_row(name, y_val, pred))

    validation = pd.DataFrame(validation_rows).sort_values("f1", ascending=False)
    print("\\nValidation results:\\n", validation.to_string(index=False))

    # Choose candidate by validation F1; baseline is retained for comparison.
    chosen_name = validation[validation["model"] != "Dummy baseline"].iloc[0]["model"]
    chosen_model = fitted[chosen_name]

    test_rows = []
    for name, model in fitted.items():
        test_rows.append(metric_row(name, y_test, model.predict(X_test)))
    test_results = pd.DataFrame(test_rows)
    print("\\nHeld-out test results:\\n", test_results.to_string(index=False))

    y_pred = chosen_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Save the selected model and its metadata.
    joblib.dump(chosen_model, MODEL_DIR / "student_risk_model.joblib")
    metadata = {
        "selected_model": chosen_name,
        "features": FEATURES,
        "target": TARGET,
        "target_definition": "1 if next assessment score is below 50, else 0",
        "risk_threshold": 0.5,
        "random_seed": SEED,
        "split": "Train: terms 1-2; validation: term 3; held-out test: term 4",
        "test_metrics": test_results.set_index("model").to_dict(orient="index"),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    test_results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "metrics.json").write_text(json.dumps({
        "validation": validation.to_dict(orient="records"),
        "test": test_results.to_dict(orient="records"),
        "selected_model": chosen_name,
        "selected_model_test_classification_report": report,
        "confusion_matrix_labels": ["Not at risk (0)", "At risk (1)"],
        "confusion_matrix": cm.tolist(),
    }, indent=2))

    # Figures.
    plt.figure(figsize=(6, 4))
    counts = df[TARGET].value_counts().sort_index()
    plt.bar(["Not at risk (0)", "At risk (1)"], [counts.get(0, 0), counts.get(1, 0)])
    plt.title("Target Class Distribution")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "figures" / "target_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix — {chosen_name}")
    plt.colorbar()
    plt.xticks([0, 1], ["Not at risk", "At risk"])
    plt.yticks([0, 1], ["Not at risk", "At risk"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "figures" / "confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    metric_names = ["precision", "recall", "f1"]
    x_positions = np.arange(len(test_results))
    width = 0.24
    for i, metric in enumerate(metric_names):
        plt.bar(x_positions + (i - 1) * width, test_results[metric], width, label=metric.title())
    plt.xticks(x_positions, test_results["model"], rotation=10)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Held-out Test Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "figures" / "model_comparison.png", dpi=150)
    plt.close()

    # Permutation importance on validation data, not the held-out test set.
    importance = permutation_importance(
        chosen_model, X_val, y_val, n_repeats=8,
        random_state=SEED, scoring="f1"
    )
    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    print(f"\\nSelected model: {chosen_name}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print("\\nSaved model, metadata, metrics and charts.")

if __name__ == "__main__":
    main()
