"""Train, compare and save fraud models.

Usage: python -m fraud.train [--config configs/sparkov.yaml] [--data PATH] [--no-cv] [--no-shap]

Method:
  - exact duplicate rows are removed
  - chronological split: train | validation | test
  - models are compared by rolling time-series cross-validation inside train+validation
  - the final model is fitted on train; its decision threshold is chosen on validation
    (by default, the threshold that maximises savings under the cost model)
  - the test set is evaluated once, with bootstrap confidence intervals
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from fraud import __version__, plots
from fraud.calibration import brier_score, expected_calibration_error, fit_calibration
from fraud.config import Config, load_config
from fraud.data import load_transactions, time_split
from fraud.datasets import get_dataset
from fraud.evaluate import at_k, bootstrap_ci, decide, evaluate, paired_bootstrap
from fraud.models import TREE_MODELS, build_models
from fraud.validation import cross_validate, pick_threshold

log = logging.getLogger("fraud.train")


def run(cfg: Config) -> dict:
    """Full training run; returns the metrics written to metrics.json."""
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    spec = get_dataset(cfg.dataset)
    df = load_transactions(spec, cfg.data_path)
    plots.plot_eda(df, cfg.outputs_dir / "eda_distribution.png")
    train, val, test = time_split(df, cfg.val_frac, cfg.test_frac)
    for name, part in (("train", train), ("val", val), ("test", test)):
        log.info("%-5s %8s rows, %4d frauds", name, f"{len(part):,}", int(part["Class"].sum()))

    # ── 1. Model comparison by time-series CV (the test set is not touched) ──
    cv = cross_validate(pd.concat([train, val]), cfg) if cfg.cv_folds > 0 else {}
    if cv:
        _log_cv(cv, cfg.selection_metric)

    # ── 2. Fit every model on train, threshold on validation, score test ─────
    X_train, y_train = spec.add_features(train), train["Class"]
    X_val, y_val = spec.add_features(val), val["Class"]
    X_test, y_test = spec.add_features(test), test["Class"]

    models = build_models(y_train, cfg.models, spec.scale_cols, seed=cfg.seed)
    report, test_probas = {}, {}
    for name, model in models.items():
        log.info("Training %s on the full training window", name)
        model.fit(X_train, y_train)
        p_val = model.predict_proba(X_val)[:, 1]
        threshold = pick_threshold(y_val, p_val, val["Amount"], cfg)
        test_probas[name] = model.predict_proba(X_test)[:, 1]
        report[name] = {
            "val": evaluate(y_val, p_val, threshold, val["Amount"], cfg.review_cost),
            "test": evaluate(y_test, test_probas[name], threshold, test["Amount"], cfg.review_cost),
        }

    # ── 3. Select (CV if available, else validation) and report on test ─────
    if cv and all(cv[n] for n in report):
        best = max(report, key=lambda n: cv[n][cfg.selection_metric]["mean"])
        selected_by = f"cv mean {cfg.selection_metric}"
    else:
        best = max(report, key=lambda n: report[n]["val"]["pr_auc"])
        selected_by = "validation pr_auc"
    best_model, threshold = models[best], report[best]["val"]["threshold"]
    p_best = test_probas[best]
    ci = bootstrap_ci(y_test, p_best)
    log.info("Selected model (%s): %s; test PR-AUC %.3f [95%% CI %.3f-%.3f]",
             selected_by, best, report[best]["test"]["pr_auc"], ci[0], ci[1])

    span_days = max((test["Time"].max() - test["Time"].min()) / 86_400, 1 / 24)
    budgets = {str(b): at_k(y_test, p_best, round(b * span_days)) for b in cfg.alert_budgets_per_day}
    comparisons = {other: paired_bootstrap(y_test, p_best, test_probas[other])
                   for other in report if other != best}

    # ── 3b. Calibrate the selected model, then pick the decision rule (validation only) ──
    p_val_raw = best_model.predict_proba(X_val)[:, 1]
    final_model = fit_calibration(best_model, p_val_raw, y_val)
    q_val, q_test = final_model.calibrate(p_val_raw), final_model.calibrate(p_best)
    calibration = {"slope": final_model.slope, "intercept": final_model.intercept}
    for name, p in (("raw", p_best), ("calibrated", q_test)):
        calibration[f"{name}_brier"] = brier_score(y_test, p)
        calibration[f"{name}_ece"] = expected_calibration_error(y_test, p)

    policies = {"threshold": {"rule": "threshold", "review_cost": cfg.review_cost,
                              "threshold": pick_threshold(y_val, q_val, val["Amount"], cfg)}}
    if cfg.threshold_method == "cost":
        policies["expected_value"] = {"rule": "expected_value", "review_cost": cfg.review_cost}
    parts = (("val", y_val, q_val, val["Amount"]), ("test", y_test, q_test, test["Amount"]))
    decision = {}
    for name, pol in policies.items():
        decision[name] = {"policy": pol}
        for part, y, q, amount in parts:
            decision[name][part] = evaluate(y, q, amount=amount, review_cost=cfg.review_cost,
                                            alerts=decide(q, amount, pol))
    rule = max(decision, key=lambda r: decision[r]["val"]["savings"])
    policy, final = policies[rule], decision[rule]["test"]
    final_alerts = decide(q_test, test["Amount"], policy)

    # ── 4. Figures ──────────────────────────────────────────────────────────
    plots.plot_pr_curves(y_test, test_probas, {n: r["test"]["pr_auc"] for n, r in report.items()},
                         cfg.outputs_dir / "pr_curves.png")
    plots.plot_confusion(y_test, final_alerts.astype(int), f"{best}\ndecision rule: {rule}",
                         cfg.outputs_dir / "confusion_matrix.png")
    plots.plot_savings_curve(y_test, q_test, test["Amount"], cfg.review_cost, int(final_alerts.sum()),
                             f"{best} ({rule} rule)", cfg.outputs_dir / "savings_curve.png")
    plots.plot_reliability(y_test, {"Raw score": p_best, "Calibrated": q_test},
                           cfg.outputs_dir / "calibration.png")
    if cv:
        plots.plot_cv(cv, cfg.outputs_dir / "cv_results.png")
    if cfg.shap and isinstance(best_model.named_steps["clf"], TREE_MODELS):
        log.info("Computing SHAP explanations")
        plots.plot_shap(best_model, test, spec, cfg.outputs_dir, cfg.shap_sample, cfg.seed)

    # ── 5. Artifacts ────────────────────────────────────────────────────────
    joblib.dump(final_model, cfg.artifacts_dir / "fraud_model.joblib")
    # Real test transactions for the dashboard demo: every fraud + up to 300 legitimate ones
    legit = test[test["Class"] == 0]
    demo = pd.concat([test[test["Class"] == 1], legit.sample(min(300, len(legit)), random_state=0)])
    demo[spec.raw_cols + ["Class"]].sort_values("Time").to_csv(
        cfg.artifacts_dir / "demo_transactions.csv", index=False)

    metrics = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": __version__,
        "dataset": cfg.dataset,
        "selected_model": best,
        "selected_by": selected_by,
        "policy": policy,               # decision rule used by scoring and the dashboard
        "final": final,                 # selected model + calibration + policy, on the test set
        "decision": decision,           # every rule: validation (used to choose) and test (report)
        "calibration": calibration,
        "threshold": threshold,         # raw-score threshold used in the model comparison
        "threshold_method": cfg.threshold_method,
        "beta": cfg.beta,
        "review_cost": cfg.review_cost,
        "features": spec.features,
        "data": {
            "rows_raw": df.attrs["rows_raw"], "rows_dedup": len(df),
            **{name: {"rows": len(part), "frauds": int(part["Class"].sum())}
               for name, part in (("train", train), ("val", val), ("test", test))},
            "test_span_days": span_days,
        },
        "test_pr_auc_ci95": ci,
        "alert_budgets": budgets,
        "paired_bootstrap_pr_auc": comparisons,
        "cv": cv,
        "models": report,
    }
    (cfg.artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _log_summary(report, best, budgets, comparisons)
    _log_decision(decision, rule, calibration)
    return metrics


def _log_decision(decision: dict, rule: str, calibration: dict) -> None:
    c = calibration
    lines = [f"Calibration on test: Brier {c['raw_brier']:.5f} -> {c['calibrated_brier']:.5f}, "
             f"ECE {c['raw_ece']:.4f} -> {c['calibrated_ece']:.4f}",
             f"{'Decision rule':16s} {'val savings':>12s} {'test savings':>13s} {'rate':>6s} "
             f"{'alerts':>7s} {'prec':>5s} {'recall':>6s}"]
    for name, d in decision.items():
        t = d["test"]
        lines.append(f"{name:16s} {d['val']['savings']:12,.0f} {t['savings']:13,.0f} "
                     f"{t['savings_rate']:6.1%} {t['n_alerts']:7d} {t['precision']:5.2f} {t['recall']:6.3f}"
                     f"{'  <- chosen on validation' if name == rule else ''}")
    log.info("Calibrated decision\n%s", "\n".join(lines))


def _log_cv(cv: dict, selection_metric: str) -> None:
    lines = [f"{'Model (CV mean ± std)':22s} {'PR-AUC':>15s} {'Savings rate':>15s} {'Recall':>15s}"]
    for name, r in cv.items():
        if r:
            lines.append(f"{name:22s} " + " ".join(
                f"{r[m]['mean']:8.3f} ± {r[m]['std']:.3f}" for m in ("pr_auc", "savings_rate", "recall")))
    log.info("Cross-validation (selection by %s)\n%s", selection_metric, "\n".join(lines))


def _log_summary(report: dict, best: str, budgets: dict, comparisons: dict) -> None:
    lines = [f"{'Model (test set)':22s} {'PR-AUC':>7s} {'Prec':>6s} {'Recall':>7s} "
             f"{'Alerts':>7s} {'Savings':>10s} {'Rate':>6s}"]
    for name, r in report.items():
        t = r["test"]
        lines.append(f"{name:22s} {t['pr_auc']:7.3f} {t['precision']:6.3f} {t['recall']:7.3f} "
                     f"{t['n_alerts']:7d} {t['savings']:10,.0f} {t['savings_rate']:6.1%}"
                     f"{'  <- selected' if name == best else ''}")
    lines.append("\nAlert budget (per day) -> precision / recall of the selected model:")
    lines += [f"  {b:>4s}/day (top {r['k']}): {r['precision']:.2f} / {r['recall']:.2f}"
              for b, r in budgets.items()]
    lines.append("\nPaired bootstrap, PR-AUC of the selected model minus:")
    lines += [f"  {o:22s} {c['diff']:+.3f}  95% CI [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  "
              f"P(better) = {c['p_a_better']:.2f}" for o, c in comparisons.items()]
    log.info("Results\n%s", "\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/sparkov.yaml")
    ap.add_argument("--data", type=Path, help="override paths.data")
    ap.add_argument("--artifacts", type=Path, help="override paths.artifacts")
    ap.add_argument("--outputs", type=Path, help="override paths.outputs")
    ap.add_argument("--threshold-method", choices=["cost", "fbeta"])
    ap.add_argument("--beta", type=float, help="F-beta for --threshold-method fbeta")
    ap.add_argument("--review-cost", type=float, help="cost of investigating one alert")
    ap.add_argument("--no-cv", action="store_true", help="skip cross-validation (faster)")
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config).with_overrides(
        data_path=args.data, artifacts_dir=args.artifacts, outputs_dir=args.outputs,
        threshold_method=args.threshold_method, beta=args.beta, review_cost=args.review_cost,
        cv_folds=0 if args.no_cv else None, shap=False if args.no_shap else None)
    run(cfg)


if __name__ == "__main__":
    main()
