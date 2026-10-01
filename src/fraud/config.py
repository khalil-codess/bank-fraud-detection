"""Typed access to config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    dataset: str = "creditcard"         # see fraud.datasets
    seed: int = 42
    data_path: Path = Path("data/raw/creditcard.csv")
    artifacts_dir: Path = Path("artifacts/creditcard")
    outputs_dir: Path = Path("outputs/creditcard")
    val_frac: float = 0.15
    test_frac: float = 0.15
    threshold_method: str = "cost"          # "cost" (maximise savings) or "fbeta"
    beta: float = 1.0
    review_cost: float = 5.0
    cv_folds: int = 4                       # 0 disables cross-validation
    cv_fold_frac: float = 0.1
    selection_metric: str = "savings_rate"  # CV mean used to pick the model (or "pr_auc")
    alert_budgets_per_day: tuple = (50, 100, 200)
    shap: bool = True
    shap_sample: int = 500
    models: dict = field(default_factory=dict)

    def __post_init__(self):
        from fraud.datasets import get_dataset

        get_dataset(self.dataset)  # raises on an unknown name
        if self.threshold_method not in ("cost", "fbeta"):
            raise ValueError(f"threshold.method must be 'cost' or 'fbeta', not {self.threshold_method!r}")

    def with_overrides(self, **overrides) -> Config:
        """Return a copy with every non-None override applied."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


def load_config(path: str | Path = "configs/sparkov.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    paths, split = raw.get("paths", {}), raw.get("split", {})
    threshold, ev = raw.get("threshold", {}), raw.get("evaluation", {})
    explain = raw.get("explain", {})
    d = Config()
    dataset = raw.get("dataset", d.dataset)
    return Config(
        dataset=dataset,
        seed=raw.get("seed", d.seed),
        data_path=Path(paths.get("data", d.data_path)),
        artifacts_dir=Path(paths.get("artifacts", f"artifacts/{dataset}")),
        outputs_dir=Path(paths.get("outputs", f"outputs/{dataset}")),
        val_frac=split.get("val_frac", d.val_frac),
        test_frac=split.get("test_frac", d.test_frac),
        threshold_method=threshold.get("method", d.threshold_method),
        beta=threshold.get("beta", d.beta),
        review_cost=raw.get("costs", {}).get("review_cost", d.review_cost),
        cv_folds=ev.get("cv_folds", d.cv_folds),
        cv_fold_frac=ev.get("cv_fold_frac", d.cv_fold_frac),
        selection_metric=ev.get("selection_metric", d.selection_metric),
        alert_budgets_per_day=tuple(ev.get("alert_budgets_per_day", d.alert_budgets_per_day)),
        shap=explain.get("shap", d.shap),
        shap_sample=explain.get("sample_size", d.shap_sample),
        models=raw.get("models", {}),
    )
