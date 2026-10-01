"""Typed access to config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    seed: int = 42
    data_path: Path = Path("data/raw/creditcard.csv")
    artifacts_dir: Path = Path("artifacts")
    outputs_dir: Path = Path("outputs")
    val_frac: float = 0.15
    test_frac: float = 0.15
    beta: float = 1.0
    shap: bool = True
    shap_sample: int = 500
    models: dict = field(default_factory=dict)

    def with_overrides(self, **overrides) -> Config:
        """Return a copy with every non-None override applied."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


def load_config(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    paths, split = raw.get("paths", {}), raw.get("split", {})
    explain = raw.get("explain", {})
    default = Config()
    return Config(
        seed=raw.get("seed", default.seed),
        data_path=Path(paths.get("data", default.data_path)),
        artifacts_dir=Path(paths.get("artifacts", default.artifacts_dir)),
        outputs_dir=Path(paths.get("outputs", default.outputs_dir)),
        val_frac=split.get("val_frac", default.val_frac),
        test_frac=split.get("test_frac", default.test_frac),
        beta=raw.get("threshold", {}).get("beta", default.beta),
        shap=explain.get("shap", default.shap),
        shap_sample=explain.get("sample_size", default.shap_sample),
        models=raw.get("models", {}),
    )
