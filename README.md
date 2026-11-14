# Student Performance & At-Risk Prediction

**Algions AI Labs — AI & Machine Learning assessment**

A decision-support prototype that estimates whether a student may score below 50 in their next assessment, using only prior learning signals. It is intended to help educators consider early academic support. It must not be used for automated punishment, grading, admission or other high-stakes decisions.

## 1. Problem definition

- **Prediction unit:** one student-term snapshot.
- **Prediction time:** after the current term's attendance, assignment and previous-assessment information is available, but before the next assessment.
- **Target:** `at_risk = 1` when the next assessment score is below 50; otherwise `0`.
- **Prediction horizon:** the next assessment.
- **Features:** attendance rate, assignment completion, previous average score, score trend, missed assignments and study hours per week.
- **Excluded:** student ID, term identifier, next assessment score, target-derived fields and protected/sensitive attributes.

The data is synthetic and generated with a fixed random seed. No real student records are used.

## 2. Project structure

```text
student_at_risk_prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── reports/
│   └── figures/
├── src/
│   ├── generate_data.py
│   └── train.py
└── tests/
    └── test_pipeline.py
```

## 3. Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.11 is not installed, use an installed supported Python version (3.10+).

## 4. Generate data and train

From the project root:

```powershell
python -m src.generate_data
python -m src.train
```

The generator creates 2,600 rows (650 students per term across four terms). The training script also generates the dataset if it is absent.

The split is chronological:
- Train: terms 1 and 2
- Validation: term 3
- Held-out test: term 4

The validation set is used to select the candidate model by F1. The held-out test set is used only for final comparison/reporting. The baseline is a most-frequent-class DummyClassifier. Candidate models are Logistic Regression and Random Forest. Median imputation is fitted within each model pipeline; Logistic Regression also uses standard scaling.

Generated outputs:
- `models/student_risk_model.joblib`
- `models/metadata.json`
- `reports/model_comparison.csv`
- `reports/metrics.json`
- `reports/figures/*.png`
- `reports/feature_importance.csv`

## 5. Run the inference app

```powershell
streamlit run app.py
```

Enter a student's prior learning signals in the sidebar and click **Predict academic risk**. The app loads the saved model artifact and calls its `predict_proba` method; it does not return a hard-coded sample prediction.

Risk bands are presentation bands over the estimated probability:
- Low: below 30%
- Moderate: 30% to below 60%
- High: 60% or above

The binary model output uses a 0.50 threshold. These are advisory thresholds for the prototype, not validated intervention policies.

## 6. Run tests

```powershell
pytest -q
```

## 7. Evaluation and interpretation

Report precision, recall and F1 alongside the confusion matrix. In this use case:
- **False positive:** a student is flagged but does not score below 50. This can lead to an unnecessary check-in or use of support resources.
- **False negative:** a student is not flagged but does score below 50. This may mean a student who needs help is missed; this is an important risk.
- **Recall:** the share of truly at-risk students detected.
- **Precision:** the share of flagged students who are actually at risk.
- **F1:** harmonic mean of precision and recall.

A teacher should review the student's circumstances before acting. The model does not know personal, health, family, language, disability, classroom or teaching context.

## 8. Data quality, leakage and responsible use

The synthetic generator creates missing predictor values and a small number of out-of-range attendance values. Domain clipping is applied to predictors; median imputation is learned only from the training partition through the pipeline.

The next assessment score is used only to construct the target and is never included in model features. The student identifier and term are not model features. Chronological term splitting approximates predicting later cohorts from earlier snapshots.

Limitations:
1. Synthetic data cannot establish real-world accuracy or representativeness.
2. Relationships in the generator are simplified and may not match real school environments.
3. Probability estimates are not guarantees and have not been independently calibrated.
4. A model may perform differently across schools, subjects, age groups or assessment styles.
5. Missing or inaccurate records can distort predictions.
6. Human review and student privacy protections are essential before any real deployment.

## 9. Demo checklist

1. Show dataset generation and row count.
2. Show class balance and missing-value checks in the training output.
3. Explain the chronological split and why future assessment score is excluded.
4. Compare baseline, Logistic Regression and Random Forest test metrics.
5. Open the Streamlit app and submit a new set of inputs.
6. Explain the risk band, signals, one possible false positive/negative and limitations.

## 10. Attribution

- Dataset: generated locally by `src/generate_data.py`; synthetic, no external dataset license required.
- Libraries: pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit and Joblib.
- AI assistance: disclose any AI tools used for planning, code assistance, debugging or documentation, in accordance with the assessment form and your institution's rules.
