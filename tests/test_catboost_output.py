import numpy as np
import pandas as pd

import iltm.tree_embedding as tree_embedding
from iltm.tree_embedding import TreeEmbedding


def test_catboost_disables_file_output(monkeypatch):
    captured_params = {}

    class RecordingCatBoost:
        def __init__(self, **params):
            captured_params.update(params)

        def fit(self, train_pool):
            return self

    monkeypatch.setattr(tree_embedding, "CatBoostClassifier", RecordingCatBoost)
    monkeypatch.setattr(tree_embedding, "Pool", lambda *args, **kwargs: object())

    tree = TreeEmbedding(
        tree_model="CatBoost",
        cat_features=[],
        task_type="classification",
        device="cpu",
        select_best_model=False,
    )
    tree._fit_model(
        pd.DataFrame({"feature": [0.0, 1.0]}),
        np.array([0, 1]),
        eval_set=None,
    )

    assert captured_params["allow_writing_files"] is False
