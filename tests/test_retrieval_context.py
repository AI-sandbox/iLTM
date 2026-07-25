import numpy as np
import torch
from sklearn.model_selection import train_test_split

import iltm.utils as utils
from iltm import iLTMClassifier


def _identity_forward(
    X,
    model_cfg,
    rf,
    pca,
    norm,
    main_network,
    device,
    use_amp=False,
    training_finetuning=False,
    finetuning_dropout=0.0,
):
    return X, X


def test_chunked_retrieval_context_matches_unchunked(monkeypatch):
    monkeypatch.setattr(
        utils,
        "forward_main_network_with_preprocessing",
        _identity_forward,
    )
    context = torch.randn(19, 5)
    targets = torch.linspace(-2, 2, 19)
    kwargs = dict(
        n_classes=1,
        model_cfg={},
        rf=None,
        pca=None,
        norm=None,
        main_network=None,
        device=torch.device("cpu"),
    )
    unchunked = utils.prepare_retrieval_context(
        context,
        targets,
        **kwargs,
    )
    chunked = utils.prepare_retrieval_context(
        context,
        targets,
        batch_size=7,
        **kwargs,
    )
    torch.testing.assert_close(chunked[0], unchunked[0], rtol=0, atol=0)
    torch.testing.assert_close(chunked[1], unchunked[1], rtol=0, atol=0)


def test_prepared_retrieval_context_is_batch_invariant(monkeypatch):
    monkeypatch.setattr(
        utils,
        "forward_main_network_with_preprocessing",
        _identity_forward,
    )
    context = torch.randn(16, 4)
    targets = torch.linspace(0, 1, 16)
    queries = torch.randn(5, 4)
    kwargs = dict(
        n_classes=1,
        batch_size=16,
        model_cfg={},
        rf=None,
        pca=None,
        norm=None,
        main_network=None,
        device=torch.device("cpu"),
    )
    prepared = utils.prepare_retrieval_context(
        context,
        targets,
        n_classes=1,
        model_cfg={},
        rf=None,
        pca=None,
        norm=None,
        main_network=None,
        device=torch.device("cpu"),
    )
    together = utils.retrieval(
        context,
        targets,
        queries,
        distance_type="euclidean",
        temperature=1.0,
        prepared_context=prepared,
        **kwargs,
    )
    separately = torch.cat(
        [
            utils.retrieval(
                context,
                targets,
                query[None],
                distance_type="euclidean",
                temperature=1.0,
                prepared_context=prepared,
                **kwargs,
            )
            for query in queries
        ]
    )
    torch.testing.assert_close(together, separately, rtol=1e-6, atol=1e-7)


def test_retrieval_predictions_are_repeatable(small_classification_data):
    X, y = small_classification_data
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    model = iLTMClassifier(
        checkpoint="r128bn",
        device="cpu",
        n_ensemble=1,
        batch_size=16,
        finetuning=False,
        do_retrieval=True,
    )
    model.fit(X_train, y_train)

    together = model.predict_proba(X_test[:8])
    repeated = model.predict_proba(X_test[:8])
    separately = np.concatenate(
        [model.predict_proba(X_test[index : index + 1]) for index in range(8)]
    )
    np.testing.assert_allclose(together, repeated, rtol=0, atol=0)
    np.testing.assert_allclose(together, separately, rtol=1e-4, atol=1e-5)
