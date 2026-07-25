import numpy as np
import pandas as pd
import pytest

from iltm.tree_embedding import TreeEmbedding


@pytest.mark.parametrize("task_type", ["classification", "regression"])
@pytest.mark.parametrize("select_best_model", [False, True])
def test_gradient_boosting_validation_parameters(task_type, select_best_model):
    X = pd.DataFrame(
        {
            "linear": np.arange(40, dtype=float),
            "cyclic": np.arange(40, dtype=float) % 5,
        }
    )
    if task_type == "classification":
        y = np.arange(40) % 2
    else:
        y = np.linspace(-1.0, 1.0, 40)
    embedding = TreeEmbedding(
        tree_model="GB",
        cat_features=[],
        task_type=task_type,
        n_estimators=3,
        select_best_model=select_best_model,
        eval_size=0.25,
        device="cpu",
    )

    embedding._fit_model(X, y)

    if select_best_model:
        assert embedding.model.validation_fraction == 0.25
        assert embedding.model.n_iter_no_change == 50
    else:
        assert embedding.model.validation_fraction == 0.1
        assert embedding.model.n_iter_no_change is None
