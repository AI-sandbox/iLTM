import numpy as np
import pandas as pd

from iltm import iLTMClassifier


def test_classifier_eval_labels_use_training_class_indices(monkeypatch):
    classifier = iLTMClassifier(checkpoint=None, device="cpu")
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
