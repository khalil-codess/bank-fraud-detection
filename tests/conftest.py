import pytest

from fraud.config import load_config
from fraud.data import time_split
from fraud.synthetic import make_synthetic


@pytest.fixture
def synthetic_df():
    return make_synthetic()


@pytest.fixture(scope="module")
def splits():
    return time_split(make_synthetic())


@pytest.fixture(scope="module")
def model_params():
    """Model settings from the real config.yaml, so tests cover what training uses."""
    return load_config("config.yaml").models
