import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import train_test_split

import iltm.inference_interface as inference_interface
from iltm import iLTMClassifier, iLTMRegressor
from iltm.tree_embedding import TreeEmbedding


def test_tree_internal_eval_is_default_and_routing_is_sklearn_configurable():
    for model_cls in (iLTMClassifier, iLTMRegressor):
        default_model = model_cls(checkpoint=None, device="cpu")
        external_model = model_cls(
            checkpoint=None,
            device="cpu",
            tree_use_external_eval_set=True,
        )

        assert default_model.tree_use_external_eval_set is False
        assert default_model.get_params()["tree_use_external_eval_set"] is False
        assert external_model.tree_use_external_eval_set is True
        assert external_model.get_params()["tree_use_external_eval_set"] is True


def test_external_eval_is_not_routed_to_tree_when_disabled(monkeypatch):
    captured = {}

    class RecordingTreeEmbedding:
        n_orig_features_to_keep_ = None

        def __init__(self, **kwargs):
            captured["select_best_model"] = kwargs["select_best_model"]

        def fit_tree(self, X, y, eval_set=None, concat_with_orig_features=True):
            captured["tree_eval_set"] = eval_set

        def transform(self, X):
            return np.zeros((len(X), 1), dtype=np.float32)

    class ModelConfig:
        pca_sampling = "zeropad"
        n_dims = 1
        clip_data_value = 1.0

    monkeypatch.setattr(inference_interface, "TreeEmbedding", RecordingTreeEmbedding)

    model = iLTMRegressor(
        checkpoint=None,
        device="cpu",
        n_ensemble=1,
        tree_embedding=True,
        tree_for_each_predictor=False,
        tree_data_split="all",
        tree_select_best_model=True,
        tree_use_external_eval_set=False,
        auto_stop_on_low_cpu_memory=False,
    )
    model.checkpoint = "unused"
    monkeypatch.setattr(model, "_auto_tune_for_memory", lambda: None)
    monkeypatch.setattr(model, "_initialize_model", lambda: ModelConfig())
    monkeypatch.setattr(
        model,
        "_preprocess_fitting_data",
        lambda X, y, is_classification: (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            object(),
        ),
    )
    monkeypatch.setattr(
        model,
        "_preprocess_test_data",
        lambda X, preprocessor: torch.as_tensor(np.asarray(X), dtype=torch.float32),
    )
    monkeypatch.setattr(
        model,
        "_generate_predictor",
        lambda *args, **kwargs: {"timed_out": False},
    )
    monkeypatch.setattr(model, "_release_training_model", lambda: None)

    X = np.arange(24, dtype=np.float32).reshape(12, 2)
    y = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    external_eval_set = (X[:4], y[:4])
    model._fit_common(X, y, eval_set=external_eval_set, n_outputs=1)

    assert captured["select_best_model"] is True
    assert captured["tree_eval_set"] is None


@pytest.mark.parametrize("tree_model", ["XGBoost_hist", "CatBoost"])
def test_internal_eval_stops_early_and_limits_leaf_embeddings(tree_model):
    n_samples = 80
    seed = 42
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, n_samples)})
    y = (3 * X["x"]).to_numpy(copy=True)
    _, validation_indices = train_test_split(
        np.arange(n_samples),
        test_size=0.2,
        random_state=seed,
    )
    y[validation_indices] *= 0.6

    tree = TreeEmbedding(
        tree_model=tree_model,
        cat_features=[],
        task_type="regression",
        seed=seed,
        device="cpu",
        n_estimators=120,
        lr=0.3,
        max_depth=2,
        select_best_model=True,
        eval_size=0.2,
    )
    tree.fit_tree(X, y, eval_set=None)

    if tree_model == "CatBoost":
        best_iteration = tree.model.get_best_iteration()
        validation_metrics = tree.model.get_evals_result()["validation"]
        trained_iterations = len(next(iter(validation_metrics.values())))
    else:
        best_iteration = tree.model.best_iteration
        trained_iterations = tree.model.num_boosted_rounds()

    leaves = tree.transform(X, onehot_encode=False)

    assert 0 < best_iteration < trained_iterations - 1
    assert trained_iterations < tree.n_estimators
    assert leaves.shape == (len(X), best_iteration + 1)
