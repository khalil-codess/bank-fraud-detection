"""Smoke test of the Streamlit dashboard against the trained artifacts."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not list(Path("artifacts").glob("*/fraud_model.joblib")),
                                reason="no trained model (python -m fraud.train)")


def test_dashboard_renders_without_errors():
    at = AppTest.from_file("app.py", default_timeout=180).run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Fraud score", "PR-AUC", "Net savings"} <= labels


def test_every_trained_dataset_renders():
    at = AppTest.from_file("app.py", default_timeout=180).run()
    for name in at.sidebar.selectbox[0].options:
        at.sidebar.selectbox[0].set_value(name).run()
        assert not at.exception, name


def test_switching_to_legitimate_example():
    at = AppTest.from_file("app.py", default_timeout=180).run()
    at.radio[0].set_value("Real legitimate").run()
    assert not at.exception
    assert next(m for m in at.metric if m.label == "True label").value == "Legitimate"
