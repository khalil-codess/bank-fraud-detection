import pytest

from fraud.config import load_config
from fraud.datasets import DATASETS, get_dataset


def config_for(name: str):
    """The real config file of a dataset, so tests cover what training uses."""
    return load_config(f"configs/{name}.yaml")


@pytest.fixture(params=DATASETS)
def spec(request):
    return get_dataset(request.param)


@pytest.fixture(scope="module", params=DATASETS)
def module_spec(request):
    return get_dataset(request.param)


@pytest.fixture(scope="module")
def module_config(module_spec):
    return config_for(module_spec.name)
