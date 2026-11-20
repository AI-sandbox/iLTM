import numpy as np
import pandas as pd
import pytest
import torch

from iltm import iLTMClassifier


@pytest.mark.parametrize("voting", ["soft", "hard"])
def test_single_class_fit_predicts_the_observed_label(tmp_path, voting):
    checkpoint = tmp_path / "cbrconcat.pth"
    checkpoint.touch()
    classifier = iLTMClassifier(checkpoint=str(checkpoint), device="cpu", voting=voting)
    X = pd.DataFrame({"category": ["a", "b", "a"], "number": [1.0, np.nan, 0.0]})

    assert classifier.fit(X, ["rare"] * len(X), eval_set=(X, ["common"] * len(X))) is classifier
    np.testing.assert_array_equal(classifier.classes_, ["rare"])
    np.testing.assert_array_equal(classifier.predict(X), ["rare"] * len(X))
    np.testing.assert_array_equal(classifier.predict_proba(X), np.ones((len(X), 1)))
    assert classifier.n_ensemble_ == 0

    with pytest.raises(ValueError, match="2 features"):
        classifier.predict(X[["category"]])


def test_single_class_state_is_cleared_on_refit(monkeypatch):
    classifier = iLTMClassifier(checkpoint=None, device="cpu", retrieval_alpha_adaptive=False)
    X = np.eye(2)
    classifier.fit(X, [7, 7])

    monkeypatch.setattr(classifier, "_fit_common", lambda *args, **kwargs: classifier)
    monkeypatch.setattr(classifier, "_predict_ensemble", lambda *args, **kwargs: torch.eye(2))
    classifier.fit(X, [7, 9])

    np.testing.assert_array_equal(classifier.predict(X), [7, 9])
    np.testing.assert_array_equal(classifier.predict_proba(X), np.eye(2))


def test_classifier_eval_labels_use_training_class_indices(monkeypatch):
    classifier = iLTMClassifier(
        checkpoint=None,
        device="cpu",
        retrieval_alpha_adaptive=False,
    )
    captured = {}

    def capture_fit_common(X, y, **kwargs):
        captured.update(kwargs)
        return classifier

    monkeypatch.setattr(classifier, "_fit_common", capture_fit_common)
    X_train = np.zeros((6, 2), dtype=np.float32)
    y_train = np.array([10, 20, 30, 10, 20, 30])
    X_val = pd.DataFrame({"x": [100, 200, 300], "z": [1, 2, 3]})
    y_val = np.array([30, 999, 20])

    classifier.fit(X_train, y_train, eval_set=(X_val, y_val))

    eval_X_proc, eval_y_proc = captured["eval_set"]
    np.testing.assert_array_equal(eval_X_proc["x"].to_numpy(), np.array([100, 300]))
    np.testing.assert_array_equal(eval_y_proc, np.array([2, 1]))


def test_classifier_ignores_eval_set_when_all_classes_are_unseen(monkeypatch):
    classifier = iLTMClassifier(checkpoint=None, device="cpu")
    captured = {}

    def capture_fit_common(X, y, **kwargs):
        captured.update(kwargs)
        return classifier

    monkeypatch.setattr(classifier, "_fit_common", capture_fit_common)
    classifier.fit(
        np.zeros((4, 2), dtype=np.float32),
        np.array([0, 1, 0, 1]),
        eval_set=(np.ones((2, 2), dtype=np.float32), np.array([8, 9])),
    )

    assert captured["eval_set"] is None
