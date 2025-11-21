import os
import sys
from pathlib import Path

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression

# Ensure tests use the same checkpoint cache directory as iltm
# This prevents duplicate downloads and respects ILTM_CKPT_DIR if set
if "ILTM_CKPT_DIR" not in os.environ:
    # Use the same logic as iltm.model_checkpoints._get_platform_cache_dir
    # to avoid circular imports
    if sys.platform == "win32":
        appdata = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if appdata:
            ckpt_dir = Path(appdata) / "iltm"
        else:
            ckpt_dir = Path.home() / ".iltm"
    elif sys.platform == "darwin":
        ckpt_dir = Path.home() / "Library" / "Caches" / "iltm"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            ckpt_dir = Path(xdg_cache) / "iltm"
        else:
            ckpt_dir = Path.home() / ".cache" / "iltm"
    os.environ["ILTM_CKPT_DIR"] = str(ckpt_dir)


@pytest.fixture
def small_classification_data():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    return X, y


@pytest.fixture
def small_multiclass_data():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    return X, y


@pytest.fixture
def small_regression_data():
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=8,
        random_state=42
    )
    return X, y


@pytest.fixture
def tiny_classification_data():
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    return X, y


@pytest.fixture
def tiny_regression_data():
    X, y = make_regression(
        n_samples=50,
        n_features=5,
        n_informative=4,
        random_state=42
    )
    return X, y

