"""Figures written to the outputs directory (non-interactive backend, no plt.show)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay  # noqa: E402

from fraud.calibration import expected_calibration_error, reliability  # noqa: E402
from fraud.evaluate import savings_curve  # noqa: E402
from fraud.inference import explain  # noqa: E402

COLORS = {0: "#4C9BE8", 1: "#E85C4C"}
LABELS = {0: "Legitimate", 1: "Fraud"}


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eda(df, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    counts = df["Class"].value_counts().sort_index()
    long_span = df["Time"].max() - df["Time"].min() > 7 * 86_400
    time_unit, time_label = (86_400, "Days") if long_span else (3600, "Hours")
    axes[0].pie(counts, labels=[LABELS[c] for c in counts.index], autopct="%1.2f%%",
                colors=[COLORS[c] for c in counts.index], startangle=90)
    axes[0].set_title("Class distribution")
    for cls in (0, 1):
        sub = df[df["Class"] == cls]
        axes[1].hist(sub["Amount"], bins=50, alpha=0.7, label=LABELS[cls], color=COLORS[cls])
        axes[2].hist(sub["Time"] / time_unit, bins=48, alpha=0.7, label=LABELS[cls],
                     color=COLORS[cls], density=True)
    axes[1].set(yscale="log", title="Transaction amount", xlabel="Amount")
    axes[2].set(title="Time of transaction (density)", xlabel=f"{time_label} since dataset start")
    axes[1].legend()
    axes[2].legend()
    fig.tight_layout()
    _save(fig, path)


def plot_pr_curves(y_test, probas: dict, scores: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in probas.items():
        PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax,
                                                name=f"{name} (AP={scores[name]:.2f})")
    ax.axhline(y_test.mean(), color="k", ls="--", alpha=0.4, label="Chance")
    ax.set_title("Precision-recall curves (chronological test set)")
    ax.legend(fontsize=8, loc="lower left")  # loc="best" is very slow on large curves
    _save(fig, path)


def plot_confusion(y_test, pred, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=[LABELS[0], LABELS[1]],
                                            ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    _save(fig, path)


def plot_savings_curve(y_test, proba, amount, review_cost, chosen_alerts: int, title, path: Path) -> None:
    """Net savings if the k most suspicious transactions are reviewed; the marker is the number of
    alerts raised by the decision rule chosen on validation."""
    _, n_alerts, savings = savings_curve(y_test, proba, amount, review_cost)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(n_alerts, savings, color=COLORS[0], label="Review the top-k scores")
    if chosen_alerts:
        ax.axvline(chosen_alerts, color=COLORS[1], ls="--", label=f"Chosen rule: {chosen_alerts} alerts")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xscale("log")
    best = max(float(np.max(savings)), 1.0)
    ax.set_ylim(-best, best * 1.2)  # flagging everything loses a fortune; zoom on the useful range
    ax.set(xlabel="Number of alerts reviewed (most suspicious first)",
           ylabel="Net savings (fraud caught - review cost)",
           title=f"{title}: test-set savings (review cost = {review_cost:g} per alert)")
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_reliability(y_test, probas: dict, path: Path, n_bins: int = 10) -> None:
    """Predicted probability vs observed fraud rate; a calibrated model follows the diagonal."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    for (name, proba), color in zip(probas.items(), (COLORS[1], COLORS[0]), strict=False):
        rows = reliability(y_test, proba, n_bins)
        pred, obs, counts = (np.array(v) for v in zip(*rows, strict=True))
        ece = expected_calibration_error(y_test, proba, n_bins)
        ax.plot(pred, obs, "o-", color=color, label=f"{name} (ECE {ece:.4f})")
        for x, y, c in zip(pred, obs, counts, strict=True):
            ax.annotate(f"{c:,}", (x, y), fontsize=6, alpha=0.6, xytext=(3, -8), textcoords="offset points")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted fraud probability",
           ylabel="Observed fraud rate", title="Calibration on the test set (labels: transactions per bin)")
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_cv(cv: dict, path: Path) -> None:
    """Mean ± std across time-series CV folds for each model."""
    names = [n for n, r in cv.items() if r]
    metrics = [("pr_auc", "PR-AUC"), ("savings_rate", "Savings rate"), ("recall", "Recall")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4), sharey=False)
    for ax, (key, label) in zip(axes, metrics, strict=True):
        means = [cv[n][key]["mean"] for n in names]
        stds = [cv[n][key]["std"] for n in names]
        ax.barh(names, means, xerr=stds, color=COLORS[0], alpha=0.85, capsize=4)
        ax.set(title=label, xlim=(min(0, min(means) - 0.1), 1))
        ax.invert_yaxis()
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle(f"Time-series cross-validation ({len(cv[names[0]]['folds'])} folds, mean ± std)")
    fig.tight_layout()
    _save(fig, path)


def plot_shap(model, test_df, spec, out_dir: Path, sample_size: int = 500, seed: int = 0) -> None:
    """Global SHAP summary plus the explanation of one real fraud from the test set."""
    import shap

    sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=seed)
    plt.figure()
    shap.summary_plot(explain(model, sample, spec), show=False, plot_size=(10, 8))
    plt.title("SHAP: feature impact on the fraud score")
    plt.tight_layout()
    _save(plt.gcf(), out_dir / "shap_summary.png")

    frauds = test_df[test_df["Class"] == 1]
    if len(frauds):
        plt.figure()
        shap.waterfall_plot(explain(model, frauds.iloc[[0]], spec)[0], show=False, max_display=12)
        plt.tight_layout()
        _save(plt.gcf(), out_dir / "shap_waterfall_fraud.png")
