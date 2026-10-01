# 🔍 Bank Fraud Detection

[![CI](https://github.com/khalil-codess/bank-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/khalil-codess/bank-fraud-detection/actions/workflows/ci.yml)

Machine-learning pipeline that detects fraudulent card transactions, evaluated the way it would be
used in production: a chronological split, a decision threshold chosen outside the test set to
maximise savings, time-series cross-validation, confidence intervals, SHAP explanations and a
Streamlit dashboard.

## Datasets

| Name | Source | Size | Why it is here |
|---|---|---|---|
| `sparkov` | [Kaggle: kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) | 1.85M transactions, 2019–2020 | Card, merchant, category and location fields make behavioural features and readable explanations possible. Simulated data. |
| `creditcard` | [Kaggle: mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 284,807 transactions, 2 days (2013) | Classic benchmark. Real data, but anonymised into 28 PCA components. |

Each dataset has a loader and a feature function in `src/fraud/datasets/`, and a config in
`configs/`. Everything else (splitting, cross-validation, cost model, dashboard) is shared.

## Results: `sparkov` dataset (baseline, no card history yet)

Chronological split of 1.85M transactions: train → validation → **test = the last 90 days
(277,859 transactions, 924 frauds worth 483,346 USD)**. The features describe a single
transaction only (amount, time of day, weekday, customer age and gender, city size,
home-to-merchant distance, merchant category); card-history features come next.

**Time-series cross-validation (4 folds, mean ± std)**: used to select the model.

| Model               | PR-AUC        | Savings rate     | Recall        | Precision     |
|---------------------|---------------|------------------|---------------|---------------|
| Logistic Regression | 0.202 ± 0.025 | 78.1% ± 2.2%     | 0.601 ± 0.023 | 0.121 ± 0.045 |
| **Random Forest** ✓ | 0.889 ± 0.017 | **96.9% ± 0.4%** | 0.917 ± 0.011 | 0.495 ± 0.060 |
| XGBoost             | 0.884 ± 0.016 | 96.0% ± 0.7%     | 0.941 ± 0.015 | 0.428 ± 0.087 |
| XGBoost + SMOTE     | 0.897 ± 0.017 | 96.7% ± 0.3%     | 0.940 ± 0.012 | 0.419 ± 0.063 |

**Held-out test period**

| Model               | PR-AUC | Precision | Recall | Alerts | Net savings | Savings rate |
|---------------------|--------|-----------|--------|--------|-------------|--------------|
| Logistic Regression | 0.160  | 0.114     | 0.590  | 4,800  | 386,536     | 80.0%        |
| **Random Forest** ✓ | 0.855  | 0.427     | 0.891  | 1,929  | 466,185     | 96.4%        |
| XGBoost             | 0.867  | 0.215     | 0.956  | 4,103  | 460,560     | 95.3%        |
| XGBoost + SMOTE     | 0.878  | 0.375     | 0.929  | 2,290  | 466,179     | 96.4%        |

- The selected Random Forest raises 1,929 alerts in 90 days (about 21 a day) and catches 823 of
  924 frauds. The 101 it misses are small: it stops **98.4% of the fraud amount**, and net of
  review costs it saves **96.4%** of fraud losses.
- With 924 test frauds the estimates are tight: test PR-AUC 0.855, 95% CI 0.835 – 0.875.
- **Ranking vs savings.** XGBoost + SMOTE ranks frauds significantly better (paired bootstrap
  PR-AUC +0.023, 95% CI +0.013 to +0.034), yet saves the same amount at its own best threshold.
  The model is chosen on savings, where Random Forest and XGBoost + SMOTE are tied.
- Logistic regression fails here (PR-AUC 0.16): fraud depends on combinations (a large amount
  *in* certain categories *at* night) that a linear model cannot express.
- With an analyst capacity of **10 alerts/day**, the top-ranked alerts have precision 0.82 and
  catch 80% of frauds.
- SHAP shows what drives the score: amount, night-time hours and the merchant category
  (`outputs/sparkov/shap_summary.png`).

## Results: `creditcard` dataset

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
savings curve (`outputs/creditcard/savings_curve.png`).

**What the numbers do and don't say**

- The tree models are statistically tied. A paired bootstrap on the test set gives XGBoost vs
  Random Forest a PR-AUC difference of −0.011 (95% CI −0.028 to +0.003), and the CV standard
  deviations overlap. Picking either is defensible.
- Test PR-AUC 0.759 has a 95% CI of 0.64 – 0.86: 52 frauds is a small sample.
- SMOTE doesn't help. It is the lowest tree model on CV savings.
- If analysts can review only **100 alerts/day**, the model's top 25 alerts in the test period are
  all frauds (precision 1.00, recall 0.48). With **200/day**, precision 0.78 and recall 0.75.

Everything above is regenerated into `artifacts/creditcard/metrics.json` by each training run.

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
| Metrics hardcoded in the dashboard | Dashboard reads `artifacts/<dataset>/metrics.json` |
| Dashboard: sliders on anonymous PCA components, V11–V28 fixed at 0 | Real test transactions, what-if on *Amount*/*Time*, CSV batch scoring |
| Tests exercised sklearn/imblearn, not the project | Tests of the project's own code, plus an end-to-end training test |

## Project layout

```
src/fraud/
  datasets/      one module per dataset: loader, features, synthetic generator
  config.py      typed access to configs/*.yaml
  data.py        loading, de-duplication, chronological split
  models.py      candidate pipelines (LR, RF, XGBoost, XGBoost + SMOTE)
  evaluate.py    thresholds, cost model, precision@k, bootstrap and paired bootstrap
  validation.py  rolling time-series cross-validation
  inference.py   load artifacts, score and explain transactions
  plots.py       figures written to outputs/
  synthetic.py   CLI writing synthetic data in a dataset's file format
  train.py       training entry point
app.py           Streamlit dashboard
configs/         one config per dataset: paths, split, costs, CV, model hyper-parameters
tests/           pytest suite (unit + end-to-end)
artifacts/<ds>/  metrics.json, demo_transactions.csv (committed); fraud_model.joblib (generated)
outputs/<ds>/    generated figures
archive/         first notebook, kept for reference
```

## Getting started

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[app,dev]"
```

Download the data (see [Datasets](#datasets)): `fraudTrain.csv` and `fraudTest.csv` into
`data/raw/sparkov/`, and/or `creditcard.csv` into `data/raw/`. Then:

```bash
python -m fraud.train --config configs/sparkov.yaml      # default config
python -m fraud.train --config configs/creditcard.yaml   # ~3 min
# options: --no-cv (faster), --review-cost 10, --threshold-method fbeta, --no-shap
pytest --cov                       # tests
ruff check src tests app.py        # lint
streamlit run app.py               # dashboard; pick the dataset in the sidebar
```

No Kaggle account? Try the pipeline on synthetic data:

```bash
python -m fraud.synthetic --dataset sparkov --rows 20000 --out data/synthetic.csv
python -m fraud.train --config configs/sparkov.yaml --data data/synthetic.csv
```

## Known limitations

- `creditcard`: **52 frauds in the test set**, so high uncertainty. Cross-validation helps, but its fold-to-fold
  spread is itself large (savings rate ± 25%).
- `creditcard`: V1–V28 are already an anonymised PCA, so business features cannot be built, and
  the data spans two days, so drift over time cannot be tested.
- `sparkov` is simulated: fraud patterns are cleaner than in real life, so its scores will look
  better than a real bank's would.
- The cost model is deliberately simple: a flat review cost, no chargeback fees, no customer-friction
  cost of false alarms. Tune `costs.review_cost` in `configs/*.yaml` to your own numbers.

## Roadmap

1. ~~Honest, leak-free evaluation~~
2. ~~Package layout, config, tests, CI~~
3. ~~Evaluation framework: time-series CV, cost-based threshold, precision@k~~
4. ~~Multi-dataset support + Sparkov dataset~~
5. Behavioural features (velocity, card history, geography)
6. LightGBM / CatBoost, Optuna tuning, probability calibration
7. Readable reason codes
8. FastAPI scoring service + Docker, MLflow tracking, drift monitoring

Joblib model files execute code when loaded: only load models you trained yourself.
