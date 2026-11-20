import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import train_test_split

from iltm import iLTMClassifier, iLTMRegressor, inference_interface, tree_embedding
from iltm.tree_embedding import TreeEmbedding


def test_tree_external_eval_is_default_and_routing_is_sklearn_configurable():
    for model_cls in (iLTMClassifier, iLTMRegressor):
        default_model = model_cls(checkpoint=None, device="cpu")
        internal_model = model_cls(
            checkpoint=None,
            device="cpu",
            tree_use_external_eval_set=False,
        )

        assert default_model.get_params()["tree_use_external_eval_set"] is True
        assert internal_model.get_params()["tree_use_external_eval_set"] is False


@pytest.mark.parametrize("tree_for_each_predictor", [False, True])
@pytest.mark.parametrize(
    ("options", "has_eval_set", "uses_external_eval"),
    [({}, True, True), ({"tree_use_external_eval_set": False}, True, False), ({}, False, False)],
)
def test_tree_validation_routing(monkeypatch, tree_for_each_predictor, options, has_eval_set, uses_external_eval):
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
        tree_for_each_predictor=tree_for_each_predictor,
        tree_data_split="all",
        tree_select_best_model=True,
        auto_stop_on_low_cpu_memory=False,
        **options,
    )
    model.checkpoint = "unused"
    monkeypatch.setattr(model, "_auto_tune_for_memory", lambda: None)
    monkeypatch.setattr(model, "_initialize_model", lambda: ModelConfig())
    monkeypatch.setattr(
        model,
        "_preprocess_fitting_data",
        lambda X, y, **kwargs: (
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
    external_eval_set = (X[:4], y[:4]) if has_eval_set else None
    model._fit_common(X, y, eval_set=external_eval_set, n_outputs=1)

    assert captured["select_best_model"] is True
    assert captured["tree_eval_set"] is (external_eval_set if uses_external_eval else None)


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


@pytest.mark.parametrize("tree_model", ["XGBoost_hist", "CatBoost"])
@pytest.mark.parametrize(
    ("labels", "eval_size", "n_validation", "external"),
    [
        pytest.param([0, 1, 2, 3], 0.2, 0, False, id="all-singletons"),
        pytest.param([0, 0, 1, 2], 0.2, 1, False, id="one-repeat"),
        pytest.param([0] * 18 + [1, 1, 2], 0.99, 18, False, id="large-holdout"),
        pytest.param([0, 1, 2, 3], 0.2, 4, True, id="external-holdout"),
    ],
)
def test_classification_holdout_keeps_training_classes_and_disjoint_rows(
    monkeypatch, tree_model, labels, eval_size, n_validation, external,
):
    # Record which labeled rows reach the real backend. Row identity is independent of the pandas index.
    batches = []
    backend, name = (tree_embedding.xgb, "DMatrix") if tree_model == "XGBoost_hist" else (tree_embedding, "Pool")
    constructor = getattr(backend, name)

    def record_rows(data, label=None, **kwargs):
        if label is not None:
            batches.append(data["row_id"].to_numpy().tolist())
        return constructor(data, label=label, **kwargs)

    monkeypatch.setattr(backend, name, record_rows)
    y = np.asarray(labels)
    row_ids = np.arange(len(y))
    X = pd.DataFrame({"row_id": row_ids}, index=row_ids // 2)
    external_set = (X + len(y), y) if external else None
    tree = TreeEmbedding(
        tree_model=tree_model, cat_features=[], task_type="classification", seed=42, device="cpu",
        n_estimators=4, max_depth=2, n_jobs=1, eval_size=eval_size, select_best_model=True,
    )
    tree.fit_tree(X, y, eval_set=external_set)

    assert len(batches) == (2 if n_validation else 1)
    train = batches[0]
    val = batches[1] if n_validation else []
    assert len(val) == n_validation
    assert set(train).isdisjoint(val)
    expected_rows = np.arange(2 * len(y)) if external else row_ids
    assert sorted(train + val) == expected_rows.tolist()
    assert set(y[train]) == set(y)
    best = tree.model.get_best_iteration() if tree_model == "CatBoost" else getattr(tree.model, "best_iteration", None)
    if n_validation:
        assert best is not None and best >= 0
    else:
        assert best is None or best < 0
        assert tree.transform(X, onehot_encode=False).shape == (len(X), tree.n_estimators)
