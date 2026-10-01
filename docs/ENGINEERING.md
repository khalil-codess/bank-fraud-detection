# Engineering notes

How the system is built and why. [Back to the README](../README.md)

## Real-time scoring API

`src/fraud/api.py` serves the Sparkov model with FastAPI. `POST /score` takes one transaction,
computes its card-history features from the card's earlier transactions, applies the decision
rule chosen at training time, returns the reasons, and then records the transaction in the card's
history.

```bash
docker compose up --build        # API on :8000 (docs at /docs), dashboard on :8502
```

The request:

```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d '{
  "card_id": 3517814635263522, "timestamp": "2021-01-02T02:13:00", "amount": 1250.00,
  "merchant": "Never Seen Ltd", "category": "shopping_net", "gender": "M", "dob": "1941-10-16",
  "lat": 37.5802, "long": -80.5248, "city_pop": 2443, "merch_lat": 38.1802, "merch_long": -79.8248}'
```

The response:

```json
{
  "decision": "review",
  "fraud_probability": 0.0201,
  "expected_loss": 25.09,
  "rule": "expected_value",
  "reasons": [
    {"text": "Amount 1,250.00 USD", "impact": 4.42, "protected": false},
    {"text": "Amount is 19.7x this card's average (63.59 USD)", "impact": 1.72, "protected": false},
    {"text": "0 other transactions on this card in the previous hour, 0 in 24 h, 29 in 7 days",
     "impact": 0.22, "protected": false}
  ],
  "card_transactions_before": 1462
}
```

A real card from the dataset (1,462 earlier transactions, usually 63.59 USD), a new 1,250 USD
online purchase at 2 a.m. The fraud probability is only 2%, but 2% of 1,250 USD is 25.09 USD,
more than the 5 USD review cost, so it goes to review.

**Training/serving consistency.** Training computes card history for a whole table at once
(`history.py`). The API keeps a small state per card and updates it one transaction at a time
(`state.py`). Both use the same running sums in the same order, and tests require them to be
exactly equal. `python -m fraud.benchmark` warms the state with everything before the test
period, then replays real test transactions one by one through the service:

| 2,000 replayed test transactions | |
|---|---|
| Difference from offline batch scores | **0.0** (identical probabilities, 100% same decisions) |
| Latency p50 / p95 / p99 | **44 / 69 / 71 ms** (features + model + SHAP reasons, without HTTP) |

Profiling took this from 106 ms to 44 ms: features are now computed once per request instead of
three times, the feature frame is built in one go (adding 34 columns one by one cost ~1 ms
each), and the SHAP explainer is built once and cached. A warm-up at startup keeps the first
request from paying for the explainer (1.3 s → 53 ms).

**Production notes.**
- The image pins the exact library versions the model was trained with
  (`requirements-lock.txt`), because pickled models are only safe with those versions.
- At startup, card history is rebuilt from `data/raw/sparkov` (999 cards, ~30 s). A real service
  would persist it in a store such as Redis.
- Requests are processed one at a time under a lock, so score-then-record never interleaves for
  a card. Scaling out would need per-card locking in the shared store.
- A transaction older than the card's latest one is rejected (HTTP 409).

## Design decisions: fixes to the first version

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

## Known limitations

- `creditcard`: **52 frauds in the test set**, so high uncertainty. Cross-validation helps, but its fold-to-fold
  spread is itself large (savings rate ± 25%).
- `creditcard`: V1–V28 are already an anonymised PCA, so business features cannot be built, and
  the data spans two days, so drift over time cannot be tested.
- `sparkov` is simulated: fraud patterns are cleaner than in real life, so its scores will look
  better than a real bank's would.
- The cost model is deliberately simple: a flat review cost, no chargeback fees, no customer-friction
  cost of false alarms. Tune `costs.review_cost` in `configs/*.yaml` to your own numbers.
