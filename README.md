# 🔍 Bank Fraud Detection

[![CI](https://github.com/khalil-codess/bank-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/khalil-codess/bank-fraud-detection/actions/workflows/ci.yml)

Machine-learning pipeline that detects fraudulent card transactions (Kaggle *Credit Card Fraud
Detection*), evaluated the way it would be used in production: a chronological split, a decision
threshold chosen outside the test set, confidence intervals, SHAP explanations and a Streamlit
dashboard.

## Results

**How models are evaluated.** The data is split by time: train → validation → test (the last 15%).
Models are compared by **rolling time-series cross-validation** (4 folds) using only data from
before the test period. The decision threshold is set on validation to **maximise net savings**
under a simple cost model: a missed fraud costs its amount, reviewing an alert costs 5. The test
set is scored once, at the end.

**Cross-validation (mean ± std over 4 time folds)**: the basis for model selection.

| Model               | PR-AUC        | Savings rate  | Recall        |
|---------------------|---------------|---------------|---------------|
| Logistic Regression | 0.756 ± 0.102 | 48.6% ± 17.6% | 0.704 ± 0.172 |
| Random Forest       | 0.788 ± 0.064 | 51.8% ± 22.6% | 0.777 ± 0.111 |
| **XGBoost** ✓       | 0.791 ± 0.059 | **57.2% ± 25.2%** | 0.779 ± 0.085 |
| XGBoost + SMOTE     | 0.785 ± 0.033 | 47.4% ± 19.5% | 0.728 ± 0.045 |

**Held-out test period** (42,558 transactions, only **52 frauds** worth 6,169 in total):

| Model               | PR-AUC | Precision | Recall | Alerts | Net savings | Savings rate |
|---------------------|--------|-----------|--------|--------|-------------|--------------|
| Logistic Regression | 0.694  | 0.639     | 0.750  | 61     | 3,491       | 56.6%        |
| Random Forest       | 0.770  | 0.830     | 0.750  | 47     | 3,561       | 57.7%        |
| **XGBoost** ✓       | 0.759  | 0.709     | 0.750  | 55     | 3,521       | 57.1%        |
| XGBoost + SMOTE     | 0.759  | 0.765     | 0.750  | 51     | 3,541       | 57.4%        |

The selected XGBoost model reviews 55 alerts, catches 39 of 52 frauds and saves **57% of fraud
losses net of review costs**. The threshold chosen on validation lands at the peak of the test
savings curve (`outputs/savings_curve.png`).

**What the numbers do and don't say**

- The tree models are statistically tied. A paired bootstrap on the test set gives XGBoost vs
  Random Forest a PR-AUC difference of −0.011 (95% CI −0.028 to +0.003), and the CV standard
  deviations overlap. Picking either is defensible.
- Test PR-AUC 0.759 has a 95% CI of 0.64 – 0.86: 52 frauds is a small sample.
- SMOTE doesn't help. It is the lowest tree model on CV savings.
- If analysts can review only **100 alerts/day**, the model's top 25 alerts in the test period are
  all frauds (precision 1.00, recall 0.48). With **200/day**, precision 0.78 and recall 0.75.

Everything above is regenerated into `artifacts/metrics.json` by each training run.

> **Correction of an earlier version.** A previous README reported F1 0.87 / precision 0.91 for
> XGBoost. Those values were hardcoded and did not match what the code produced (measured with the
> same protocol: precision ≈ 0.53, F1 ≈ 0.66). They have been removed.

## Design decisions

| Problem in the first version | Fix |
|---|---|
| One scaler fitted twice: `scaler.pkl` only held *Time*, and the app hardcoded *Amount* statistics (train/serving skew) | Deterministic features (`log1p(Amount)`, cyclic hour) and the scaler saved inside a single `Pipeline` |
| Scaling before the split (leakage) | Scaler fitted on training data only |
| Random split: the model sees the "future" | **Chronological** train / validation / test split |
| 1,081 duplicate rows that could land in both train and test | De-duplication |
| Fixed 0.5 threshold; model chosen and evaluated on the same test set | Threshold that maximises savings on validation; model chosen by time-series CV |
| ROC-AUC as headline metric (optimistic at 0.17% fraud) | **PR-AUC** and **net savings**, with bootstrap CIs and paired model comparisons |
| "XGBoost + SMOTE" advertised, but SMOTE only applied to logistic regression | SMOTE tested as a proper ablation: **it does not help here** |
| Metrics hardcoded in the dashboard | Dashboard reads `artifacts/metrics.json` |
| Dashboard: sliders on anonymous PCA components, V11–V28 fixed at 0 | Real test transactions, what-if on *Amount*/*Time*, CSV batch scoring |
| Tests exercised sklearn/imblearn, not the project | Tests of the project's own code, plus an end-to-end training test |

## Project layout

```
src/fraud/
  config.py      typed access to config.yaml
  data.py        loading, de-duplication, chronological split
  features.py    feature engineering shared by training and serving
  models.py      candidate pipelines (LR, RF, XGBoost, XGBoost + SMOTE)
  evaluate.py    thresholds, cost model, precision@k, bootstrap and paired bootstrap
  validation.py  rolling time-series cross-validation
  inference.py   load artifacts, score and explain transactions
  plots.py       figures written to outputs/
  synthetic.py   synthetic data in the Kaggle schema (tests, CI)
  train.py       training entry point
app.py           Streamlit dashboard
config.yaml      paths, split, threshold and model hyper-parameters
tests/           pytest suite (unit + end-to-end)
artifacts/       metrics.json, demo_transactions.csv (committed); fraud_model.joblib (generated)
outputs/         generated figures
archive/         first notebook, kept for reference
```

## Getting started

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[app,dev]"
```

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
into `data/raw/`, then:

```bash
python -m fraud.train              # ~3 min; options: --no-cv, --review-cost 10, --threshold-method fbeta, --no-shap
pytest --cov                       # tests
ruff check src tests app.py        # lint
streamlit run app.py               # dashboard
```

No Kaggle account? Try the pipeline on synthetic data:

```bash
python -m fraud.synthetic --rows 20000 --out data/synthetic.csv
python -m fraud.train --data data/synthetic.csv
```

## Known limitations

- **52 frauds in the test set**: high uncertainty. Cross-validation helps, but its fold-to-fold
  spread is itself large (savings rate ± 25%).
- V1–V28 are already an anonymised PCA, so business features (card history, velocity, geography)
  cannot be built from this dataset.
- The data is from 2013 and spans two days: no way to test drift over time.
- The cost model is deliberately simple: a flat review cost, no chargeback fees, no customer-friction
  cost of false alarms. Tune `costs.review_cost` in `config.yaml` to your own numbers.

## Roadmap

1. ~~Honest, leak-free evaluation~~
2. ~~Package layout, config, tests, CI~~
3. ~~Evaluation framework: time-series CV, cost-based threshold, precision@k~~
4. Richer dataset with card IDs and behavioural features
5. LightGBM / CatBoost, Optuna tuning, probability calibration
6. Readable reason codes
7. FastAPI scoring service + Docker, MLflow tracking, drift monitoring

Joblib model files execute code when loaded: only load models you trained yourself.
