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

## Results: `sparkov` dataset

Chronological split of 1.85M transactions: train → validation → **test = the last 90 days
(277,859 transactions, 924 frauds worth 483,346 USD)**.

### Card-history features

Real fraud systems look at the card's behaviour, not just the transaction. For every transaction,
`src/fraud/history.py` summarises **only the card's earlier transactions**: counts in the last
1 h / 24 h / 7 days, amount spent in the last 24 h, the card's usual amount (mean and spread),
time since its previous purchase, earlier visits to this merchant and category, and the distance
and speed from the previous purchase. Model features compare the current transaction with that
history (e.g. *amount ÷ card's usual amount*).

Two guarantees are enforced by tests (`tests/test_history.py`):
- **no look-ahead**: a transaction's history is bit-for-bit identical whether later data exists
  or is altered;
- **no labels**: fraud labels are never used (in production they arrive weeks later).

### Tuning, calibration and an expected-value decision rule

- **Tuning** (`python -m fraud.tune`): Optuna searches hyper-parameters inside the training
  window only (fit on its older 80%, score PR-AUC on its most recent 20%), then the winners are
  copied into `configs/sparkov.yaml`. XGBoost: 0.980 → 0.985 on the tuning holdout, and the gain
  carried over to cross-validation (0.970 → 0.982). **LightGBM's did not** (0.985 on the tuning
  holdout, 0.961 in CV): tuning ran on 20% of the legitimate rows for speed, and size-dependent
  settings (7 samples per leaf) behave differently on the full data. Cross-validation caught it.
- **A LightGBM pitfall**: with a large class weight, LightGBM's default minimum hessian per leaf
  (0.001) let leaf values explode on the creditcard data: training stopped after 46 of 400 trees
  with PR-AUC 0.02. The project now uses XGBoost's default (1.0), with a regression test.
- **Calibration**: the selected model's scores go through Platt scaling fitted on validation
  data. It is strictly increasing, so ranking, PR-AUC and SHAP are unchanged, but a score now
  reads as a probability: raw scores were over-confident in the 45–85% range, calibrated ones
  follow the diagonal (`outputs/sparkov/calibration.png`).
- **Decision rule**: with real probabilities, the cost-optimal rule is *review a transaction when
  probability × amount ≥ review cost*. A 2,000 USD purchase at 5% risk gets reviewed, a 10 USD one
  at 30% does not. It is compared on validation with a single best threshold, and it wins there
  (637,234 vs 635,839 USD) and on the test period:

| Decision rule (test period) | Alerts | Precision | Frauds caught | Fraud amount stopped | Net savings |
|---|---|---|---|---|---|
| Single score threshold (best on validation) | 1,200 | 0.75 | 901 / 924 | 99.4% | 474,338 USD (98.1%) |
| **probability × amount ≥ 5 USD** ✓ | 1,276 | 0.69 | 883 / 924 | **99.8%** | **476,033 USD (98.5%)** |

The expected-value rule catches *fewer* frauds but *more money*: it lets through tiny frauds whose
loss is smaller than the cost of reviewing them, and reviews large purchases even at low risk.

### Progress across steps (selected model, held-out test period)

| | Transaction only | + card history | + tuning, calibration, EV rule |
|---|---|---|---|
| Selected model | Random Forest | XGBoost + SMOTE | **XGBoost (calibrated)** |
| PR-AUC (95% CI) | 0.855 (0.835 – 0.875) | 0.968 (0.959 – 0.974) | **0.980 (0.974 – 0.985)** |
| Alerts in 90 days | 1,929 | 1,824 | **1,276** |
| Precision of alerts | 0.43 | 0.50 | **0.69** |
| Fraud amount stopped | 98.4% | 99.8% | **99.8%** |
| Net savings | 466,185 USD (96.4%) | 473,172 USD (97.9%) | **476,033 USD (98.5%)** |
| 10 reviews/day: precision / recall | 0.82 / 0.80 | 0.93 / 0.91 | **0.95 / 0.93** |

From the first to the last column, analysts get a third fewer alerts, and two thirds of them are
real frauds instead of four in ten.

### All models (card history, tuned)

**Time-series cross-validation (4 folds, mean ± std)**

| Model               | PR-AUC        | Savings rate     | Recall        | Precision     |
|---------------------|---------------|------------------|---------------|---------------|
| Logistic Regression | 0.372 ± 0.055 | 90.9% ± 0.5%     | 0.840 ± 0.045 | 0.208 ± 0.029 |
| Random Forest       | 0.942 ± 0.016 | 97.3% ± 0.6%     | 0.956 ± 0.018 | 0.564 ± 0.054 |
| **XGBoost** ✓       | 0.982 ± 0.007 | **98.0% ± 0.3%** | 0.980 ± 0.011 | 0.690 ± 0.078 |
| LightGBM            | 0.961 ± 0.002 | 96.7% ± 0.5%     | 0.968 ± 0.013 | 0.611 ± 0.104 |
| XGBoost + SMOTE     | 0.981 ± 0.007 | 97.9% ± 0.4%     | 0.983 ± 0.011 | 0.638 ± 0.111 |

**Held-out test period** (raw scores, each model's best threshold on validation)

| Model               | PR-AUC | Precision | Recall | Alerts | Net savings | Savings rate |
|---------------------|--------|-----------|--------|--------|-------------|--------------|
| Logistic Regression | 0.280  | 0.175     | 0.810  | 4,274  | 435,795     | 90.2%        |
| Random Forest       | 0.924  | 0.510     | 0.952  | 1,726  | 471,159     | 97.5%        |
| **XGBoost** ✓       | 0.980  | 0.751     | 0.975  | 1,200  | 474,338     | 98.1%        |
| LightGBM            | 0.930  | 0.422     | 0.979  | 2,143  | 469,698     | 97.2%        |
| XGBoost + SMOTE     | 0.979  | 0.761     | 0.975  | 1,184  | 476,377     | 98.6%        |

- XGBoost and XGBoost + SMOTE (same trees, different imbalance handling) are tied: paired
  bootstrap PR-AUC difference +0.000 (95% CI −0.002 to +0.003). XGBoost wins the CV savings and
  needs no resampling, so it is selected.
- XGBoost beats LightGBM (+0.050, CI +0.029 to +0.071) and Random Forest (+0.056).
- Logistic regression can't model the interactions that matter (large amount *and* online
  category *and* night *and* unusual for this card).

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
| LightGBM            | 0.396 ± 0.115 | 39.8% ± 12.7% | 0.593 ± 0.107 |
| XGBoost + SMOTE     | 0.785 ± 0.033 | 47.4% ± 19.5% | 0.728 ± 0.045 |

**Held-out test period** (42,558 transactions, only **52 frauds** worth 6,169 in total):

| Model               | PR-AUC | Precision | Recall | Alerts | Net savings | Savings rate |
|---------------------|--------|-----------|--------|--------|-------------|--------------|
| Logistic Regression | 0.694  | 0.639     | 0.750  | 61     | 3,491       | 56.6%        |
| Random Forest       | 0.770  | 0.830     | 0.750  | 47     | 3,561       | 57.7%        |
| **XGBoost** ✓       | 0.759  | 0.709     | 0.750  | 55     | 3,521       | 57.1%        |
| LightGBM            | 0.520  | 0.180     | 0.731  | 211    | 2,690       | 43.6%        |
| XGBoost + SMOTE     | 0.759  | 0.765     | 0.750  | 51     | 3,541       | 57.4%        |

The selected XGBoost model reviews 55 alerts, catches 39 of 52 frauds and saves **57% of fraud
losses net of review costs**. The threshold chosen on validation lands at the peak of the test
savings curve (`outputs/creditcard/savings_curve.png`).

**What the numbers do and don't say**

- The tree models are statistically tied. A paired bootstrap on the test set gives XGBoost vs
  Random Forest a PR-AUC difference of −0.011 (95% CI −0.028 to +0.003), and the CV standard
  deviations overlap. Picking either is defensible.
- Test PR-AUC 0.759 has a 95% CI of 0.64 – 0.86: 52 frauds is a small sample.
- SMOTE doesn't help. LightGBM, untuned for this small dataset, is the weakest tree model.
- Calibration helps here (Brier score 0.00111 → 0.00043), but the **expected-value rule loses on
  validation** (7,735 vs 7,936 saved), so the single threshold is kept: with 55 validation
  frauds, probabilities are too rough for the amount-weighted rule. The choice is data-driven.
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
  history.py     point-in-time card history (velocity, usual amount, merchant novelty, travel)
  config.py      typed access to configs/*.yaml
  data.py        loading, de-duplication, chronological split
  models.py      candidate pipelines (LR, RF, XGBoost, LightGBM, XGBoost + SMOTE)
  evaluate.py    thresholds, cost model, precision@k, bootstrap and paired bootstrap
  validation.py  rolling time-series cross-validation
  calibration.py Platt calibration, reliability and calibration error
  tune.py        Optuna hyper-parameter search inside the training window
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
python -m fraud.tune --config configs/sparkov.yaml --model XGBoost --trials 40   # ~5 min
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
5. ~~Behavioural features (velocity, card history, geography)~~
6. ~~LightGBM, Optuna tuning, probability calibration, expected-value decision rule~~
7. Readable reason codes
8. FastAPI scoring service + Docker, MLflow tracking, drift monitoring

Joblib model files execute code when loaded: only load models you trained yourself.
