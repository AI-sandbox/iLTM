"""CatBoost sizes its GPU pool from free memory, independently of total VRAM."""

import numpy as np
import pandas as pd
import pytest

import iltm.tree_embedding as tree_embedding
import iltm.utils as utils
from iltm.tree_embedding import TreeEmbedding


@pytest.mark.parametrize("free_mb,total_mb", [(10541, 24142), (10541, 81156), (80000, 81156)])
def test_fraction_does_not_discount_free_memory_twice(monkeypatch, free_mb, total_mb):
    monkeypatch.setattr(utils, "get_gpu_memory_info", lambda *a, **kw: {
        "free_mb": free_mb, "total_mb": total_mb,
    })
    fraction = utils.pick_gpu_ram_part("cuda:0")
    assert fraction == pytest.approx(0.9)
    assert free_mb * fraction == pytest.approx(free_mb * 0.9)


@pytest.mark.parametrize("cap,floor,expected", [(0.8, 0.3, 0.8), (0.95, 0.92, 0.92), (0.95, 0.3, 0.9)])
def test_explicit_fraction_bounds(cap, floor, expected):
    assert utils.pick_gpu_ram_part(cap=cap, floor=floor) == pytest.approx(expected)


def test_fraction_requires_no_cuda_measurement(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("CatBoost itself measures free VRAM")
    monkeypatch.setattr(utils, "get_gpu_memory_info", unexpected)
    assert utils.pick_gpu_ram_part("cuda:0") == pytest.approx(0.9)


@pytest.mark.parametrize("task,constructor", [("classification", "CatBoostClassifier"), ("regression", "CatBoostRegressor")])
def test_fit_releases_cache_before_sizing_pool_and_preserves_tree_settings(monkeypatch, task, constructor):
    events = []
    class CatBoost:
        def __init__(self, **params):
            self.params = params
        def fit(self, *args, **kwargs):
            events.append("fit")
            return self
    def memory_info(*args, **kwargs):
        events.append("measure")
        return {"free_mb": 10541, "total_mb": 24142}
    monkeypatch.setattr(tree_embedding.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(tree_embedding, "clear_cuda_cache", lambda: events.append("release"))
    monkeypatch.setattr(tree_embedding, "get_gpu_memory_info", memory_info)
    monkeypatch.setattr(tree_embedding, constructor, CatBoost)
    monkeypatch.setattr(tree_embedding, "Pool", lambda *a, **kw: object())
    tree = TreeEmbedding(tree_model="CatBoost", cat_features=[], task_type=task,
                         device="cuda:0", select_best_model=False,
                         n_estimators=125, max_depth=5)
    tree._fit_model(pd.DataFrame({"x": [0., 1.]}), np.array([0, 1]))
    assert events[:4] == ["gc", "release", "measure", "fit"]
    assert tree.model.params["gpu_ram_part"] == pytest.approx(0.9)
    assert tree.model.params["task_type"] == "GPU"
    assert tree.model.params["iterations"] == 125
    assert tree.model.params["max_depth"] == 5
