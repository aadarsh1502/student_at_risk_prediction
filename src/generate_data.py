"""Generate reproducible synthetic student-assessment records.

Each row represents information available BEFORE a next assessment.
The target is whether the next assessment score is below 50.
"""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [
    "attendance_rate",
    "assignment_completion",
    "previous_avg_score",
    "score_trend",
    "missed_assignments",
    "study_hours_per_week",
]

def generate_dataset(n_students: int = 650, n_terms: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for term in range(1, n_terms + 1):
        for student_num in range(n_students):
            # Stable student-level tendency plus term-level variation.
            ability = rng.normal(0, 1)
            attendance = np.clip(82 + 9 * ability + rng.normal(0, 12), 25, 100)
            completion = np.clip(78 + 10 * ability + rng.normal(0, 15), 10, 100)
            previous = np.clip(63 + 11 * ability + rng.normal(0, 13), 5, 98)
            trend = np.clip(1.5 * ability + rng.normal(0, 8), -25, 25)
            missed = int(np.clip(round((100 - completion) / 12 + rng.normal(0, 1.5)), 0, 12))
            study = np.clip(7 + 2.2 * ability + rng.normal(0, 3), 0, 25)

            # The next score is generated from prior signals only.
            latent_next_score = (
                0.30 * previous
                + 0.22 * attendance
                + 0.20 * completion
                + 0.45 * trend
                + 0.65 * study
                - 1.0 * missed
                + rng.normal(0, 10)
                - 15
            )
            next_score = float(np.clip(latent_next_score, 0, 100))
            at_risk = int(next_score < 50)

            rows.append({
                "student_id": f"T{term}-S{student_num+1:04d}",
                "term": term,
                "attendance_rate": round(attendance, 2),
                "assignment_completion": round(completion, 2),
                "previous_avg_score": round(previous, 2),
                "score_trend": round(trend, 2),
                "missed_assignments": missed,
                "study_hours_per_week": round(study, 2),
                "next_assessment_score": round(next_score, 2),
                "at_risk": at_risk,
            })

    df = pd.DataFrame(rows)

    # Introduce realistic data-quality issues in predictor columns only.
    for col in FEATURES:
        missing_idx = rng.choice(df.index, size=max(1, int(0.015 * len(df))), replace=False)
        df.loc[missing_idx, col] = np.nan

    # A few implausible values to demonstrate validation/clipping.
    outlier_idx = rng.choice(df.index, size=max(1, int(0.003 * len(df))), replace=False)
    df.loc[outlier_idx, "attendance_rate"] = 130
    return df

if __name__ == "__main__":
    output = Path("data/student_performance.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(output, index=False)
    print(f"Saved {len(dataset):,} rows to {output}")
    print(dataset.head())
    print("\\nAt-risk distribution:")
    print(dataset["at_risk"].value_counts(normalize=True).rename("proportion"))
