from __future__ import annotations

import weakref

import numpy as np
import pandas as pd
import pytest
import torch

from iltm import iLTMClassifier, tree_embedding
from iltm.iltm_model import iLTM
from iltm.tree_embedding import TreeEmbedding


@pytest.mark.parametrize(
    "rows,cols,select_best,external,should_cache",
    [
        (1024, 10_000, True, True, True),
        (1025, 10_000, True, True, False),
        (1024, 9_999, True, True, False),
        (32, 10_000, True, False, False),
        (32, 10_000, False, False, True),
    ],
)
def test_pool_reuse_eligibility(monkeypatch, rows, cols, select_best, external, should_cache):
    X = pd.DataFrame(np.broadcast_to(np.float32(1), (rows, cols)))
    y = np.arange(rows) % 2
    tree = TreeEmbedding("CatBoost", [], "classification", device="cpu", select_best_model=select_best)
    cache = {}

    class ReachedPoolConstruction(Exception):
        pass

    def check(X, y, X_val, y_val, actual_cache):
        assert actual_cache is (cache if should_cache else None)
        raise ReachedPoolConstruction

    monkeypatch.setattr(tree, "_catboost_fit_pools", check)
    valid = (X.iloc[:2], y[:2]) if external else None
    with pytest.raises(ReachedPoolConstruction):
        tree._fit_model(X, y, valid, catboost_fit_cache=cache)


def test_wide_mixed_ensemble_reuses_pools_and_preserves_predictions(tmp_path, monkeypatch):
    arch = dict(n_dims=4, hn_n_layers=3, hn_hidden_size=8, rf_size=12, n_classes_limit=5)
    cb, xgb = [str(tmp_path / name) for name in ("cbrconcat.pth", "xgbrconcat.pth")]
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(19)
        weights = iLTM(**arch).state_dict()
        torch.save(weights, cb)
        torch.save(weights, xgb)
    monkeypatch.setattr(iLTMClassifier, "_model_cache", {})
    model = iLTMClassifier(
        checkpoint=cb, checkpoint_mix=(cb, xgb), n_ensemble=3, device="cpu", n_jobs=1,
        tree_n_estimators=3, tree_max_depth=2, finetuning=False, retrieval_alpha_adaptive=False,
        cat_features=[0], auto_stop_on_low_cpu_memory=False, **arch,
    )
    rng = np.random.default_rng(57)
    X = pd.DataFrame(rng.normal(size=(40, 10_000)).astype(np.float32))
    categories = [2.5, 0.0, None, 1.5] * 8 + [9.5] * 8
    X[0] = pd.Categorical(categories)
    y = np.arange(len(X)) % 2
    train, valid = X.iloc[:32], X.iloc[32:]
    pools = []
    make_pool = tree_embedding.Pool

    def record(*args, **kwargs):
        pool = make_pool(*args, **kwargs)
        if kwargs.get("label") is not None:
            pools.append(weakref.ref(pool))
        return pool

    monkeypatch.setattr(tree_embedding, "Pool", record)

    def fit_and_predict():
        model.fit(train, y[:32], eval_set=(valid, y[32:]))
        cats = [t for t in model.tr_ if t.tree_model == "CatBoost"]
        assert len(cats) == 2 and cats[0].seed != cats[1].seed
        assert all(ref() is None for ref in pools)
        return [
            (t.model.get_best_iteration(), t.orig_feature_indices_to_keep_.tolist(), t.transform(valid).tobytes())
            for t in cats
        ], model.predict_proba(valid)

    members, predictions = fit_and_predict()
    assert len(pools) == 2
    pools.clear()
    normal = TreeEmbedding._catboost_fit_pools
    monkeypatch.setattr(TreeEmbedding, "_catboost_fit_pools", lambda self, X, y, Xv, yv, cache: normal(self, X, y, Xv, yv, None))
    reference_members, reference_predictions = fit_and_predict()
    assert len(pools) == 4
    assert members == reference_members
    np.testing.assert_array_equal(predictions, reference_predictions)
