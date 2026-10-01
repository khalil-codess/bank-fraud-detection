"""Train, compare and save fraud models.

Usage: python -m fraud.train [--config config.yaml] [--data PATH] [--beta 2] [--no-shap]

Method:
  - exact duplicate rows are removed
  - chronological train / validation / test split (no information from the future)
  - the model AND the decision threshold are chosen on validation only
  - the test set is evaluated once, with a bootstrap confidence interval
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from fraud import __version__, plots
from fraud.config import Config, load_config
from fraud.data import load_transactions, time_split
from fraud.evaluate import bootstrap_ci, evaluate, select_threshold
from fraud.features import FEATURES, RAW_COLS, add_features
from fraud.models import TREE_MODELS, build_models

log = logging.getLogger("fraud.train")


def run(cfg: Config) -> dict:
    """Full training run; returns the metrics written to metrics.json."""
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    df = load_transactions(cfg.data_path)
    plots.plot_eda(df, cfg.outputs_dir / "eda_distribution.png")
    train, val, test = time_split(df, cfg.val_frac, cfg.test_frac)
    for name, part in (("train", train), ("val", val), ("test", test)):
        log.info("%-5s %8s rows, %4d frauds", name, f"{len(part):,}", int(part["Class"].sum()))

    X_train, y_train = add_features(train), train["Class"]
    X_val, y_val = add_features(val), val["Class"]
    X_test, y_test = add_features(test), test["Class"]

    models = build_models(y_train, cfg.models, seed=cfg.seed)
    report, test_probas = {}, {}
    for name, model in models.items():
        log.info("Training %s", name)
        model.fit(X_train, y_train)
        p_val = model.predict_proba(X_val)[:, 1]
        threshold = select_threshold(y_val, p_val, beta=cfg.beta)
        test_probas[name] = model.predict_proba(X_test)[:, 1]
        report[name] = {"val": evaluate(y_val, p_val, threshold),
                        "test": evaluate(y_test, test_probas[name], threshold)}
        t = report[name]["test"]
        log.info("  val PR-AUC %.3f | threshold %.3f | test PR-AUC %.3f P %.3f R %.3f F1 %.3f",
                 report[name]["val"]["pr_auc"], threshold, t["pr_auc"],
                 t["precision"], t["recall"], t["f1"])

    best = max(report, key=lambda n: report[n]["val"]["pr_auc"])
    best_model, threshold = models[best], report[best]["val"]["threshold"]
    ci = bootstrap_ci(y_test, test_probas[best])
    log.info("Selected model (best validation PR-AUC): %s; test PR-AUC %.3f [95%% CI %.3f-%.3f]",
             best, report[best]["test"]["pr_auc"], ci[0], ci[1])

    plots.plot_pr_curves(y_test, test_probas, {n: r["test"]["pr_auc"] for n, r in report.items()},
                         cfg.outputs_dir / "pr_curves.png")
    plots.plot_confusion(y_test, (test_probas[best] >= threshold).astype(int),
                         f"{best}\nthreshold = {threshold:.3f}", cfg.outputs_dir / "confusion_matrix.png")
    if cfg.shap and isinstance(best_model.named_steps["clf"], TREE_MODELS):
        log.info("Computing SHAP explanations")
        plots.plot_shap(best_model, test, cfg.outputs_dir, cfg.shap_sample, cfg.seed)

    joblib.dump(best_model, cfg.artifacts_dir / "fraud_model.joblib")
    # Real test transactions for the dashboard demo: every fraud + up to 300 legitimate ones
    legit = test[test["Class"] == 0]
    demo = pd.concat([test[test["Class"] == 1], legit.sample(min(300, len(legit)), random_state=0)])
    demo[RAW_COLS + ["Class"]].sort_values("Time").to_csv(
        cfg.artifacts_dir / "demo_transactions.csv", index=False)

    metrics = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": __version__,
        "selected_model": best,
        "threshold": threshold,
        "beta": cfg.beta,
        "features": FEATURES,
        "data": {
            "rows_raw": df.attrs["rows_raw"], "rows_dedup": len(df),
            **{name: {"rows": len(part), "frauds": int(part["Class"].sum())}
               for name, part in (("train", train), ("val", val), ("test", test))},
        },
        "test_pr_auc_ci95": ci,
        "models": report,
    }
    (cfg.artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _print_summary(report, best)
    return metrics


def _print_summary(report: dict, best: str) -> None:
    lines = [f"{'Model (test set)':22s} {'PR-AUC':>7s} {'ROC-AUC':>8s} {'Prec':>6s} "
             f"{'Recall':>7s} {'F1':>6s}"]
    for name, r in report.items():
        t = r["test"]
        lines.append(f"{name:22s} {t['pr_auc']:7.3f} {t['roc_auc']:8.3f} {t['precision']:6.3f} "
                     f"{t['recall']:7.3f} {t['f1']:6.3f}{'  <- selected' if name == best else ''}")
    log.info("Results\n%s", "\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", type=Path, help="override paths.data")
    ap.add_argument("--artifacts", type=Path, help="override paths.artifacts")
    ap.add_argument("--outputs", type=Path, help="override paths.outputs")
    ap.add_argument("--beta", type=float, help="F-beta used to pick the threshold")
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config).with_overrides(
        data_path=args.data, artifacts_dir=args.artifacts, outputs_dir=args.outputs,
        beta=args.beta, shap=False if args.no_shap else None)
    run(cfg)


if __name__ == "__main__":
    main()
