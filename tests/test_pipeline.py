import numpy as np
import pandas as pd
from src.generate_data import generate_dataset, FEATURES
from src.train import clean_features, build_pipeline
from sklearn.linear_model import LogisticRegression

def test_generator_is_reproducible():
    a = generate_dataset(n_students=20, n_terms=2, seed=7)
    b = generate_dataset(n_students=20, n_terms=2, seed=7)
    pd.testing.assert_frame_equal(a, b)

def test_target_and_required_features_exist():
    df = generate_dataset(n_students=25, n_terms=2)
    assert set(FEATURES).issubset(df.columns)
    assert set(df["at_risk"].unique()).issubset({0, 1})
    assert (df["next_assessment_score"].between(0, 100)).all()

def test_cleaning_clips_invalid_values():
    sample = pd.DataFrame([{
        "attendance_rate": 130, "assignment_completion": -5,
        "previous_avg_score": 110, "score_trend": 90,
        "missed_assignments": 50, "study_hours_per_week": 80,
    }])
    cleaned = clean_features(sample)
    assert cleaned.iloc[0]["attendance_rate"] == 100
    assert cleaned.iloc[0]["assignment_completion"] == 0
    assert cleaned.iloc[0]["previous_avg_score"] == 100
    assert cleaned.iloc[0]["score_trend"] == 30
    assert cleaned.iloc[0]["missed_assignments"] == 20
    assert cleaned.iloc[0]["study_hours_per_week"] == 40

def test_pipeline_handles_missing_values():
    df = generate_dataset(n_students=80, n_terms=2)
    X = clean_features(df)
    y = df["at_risk"]
    model = build_pipeline(LogisticRegression(max_iter=500), scale=True)
    model.fit(X, y)
    prediction = model.predict(X.iloc[:5])
    assert len(prediction) == 5
    assert set(prediction).issubset({0, 1})
