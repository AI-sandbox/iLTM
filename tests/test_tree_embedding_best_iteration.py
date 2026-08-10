import numpy as np
import pandas as pd
import pytest

from iltm.tree_embedding import TreeEmbedding


@pytest.mark.parametrize("tree_model", ["XGBoost_hist", "CatBoost"])
def test_leaf_embeddings_end_at_best_iteration(tree_model):
    X_train = pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, 48),
        }
    )
    y_train = (3 * X_train["x"]).to_numpy()
    X_valid = pd.DataFrame(
        {
            "x": np.linspace(-0.95, 0.95, 24),
        }
    )
    y_valid = (0.6 * 3 * X_valid["x"]).to_numpy()

    tree = TreeEmbedding(
        tree_model=tree_model,
        cat_features=[],
        task_type="regression",
        device="cpu",
        n_estimators=12,
        lr=0.3,
        max_depth=2,
        select_best_model=True,
    )
    tree.fit_tree(X_train, y_train, eval_set=(X_valid, y_valid))

    if tree_model == "CatBoost":
        best_iteration = tree.model.get_best_iteration()
    else:
        best_iteration = tree.model.best_iteration

    assert 0 < best_iteration < tree.n_estimators - 1
    leaves = tree.transform(X_valid, onehot_encode=False)
    assert leaves.shape == (len(X_valid), best_iteration + 1)
