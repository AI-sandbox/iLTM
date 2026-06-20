from iltm.tree_embedding import TreeEmbedding


def make_tree_embedding(device):
    return TreeEmbedding(
        tree_model="XGBoost_hist",
        cat_features=[],
        task_type="regression",
        device=device,
    )


def test_cuda_device_maps_to_tree_gpu_backends():
    tree_embedding = make_tree_embedding("cuda:0")

    assert tree_embedding._is_gpu_device()
    assert tree_embedding._xgboost_device() == "cuda:0"
    assert tree_embedding._catboost_task_type() == "GPU"
    assert tree_embedding._catboost_devices() == "0"


def test_cuda_without_ordinal_maps_to_default_tree_gpu_backends():
    tree_embedding = make_tree_embedding("cuda")

    assert tree_embedding._is_gpu_device()
    assert tree_embedding._xgboost_device() == "cuda"
    assert tree_embedding._catboost_task_type() == "GPU"
    assert tree_embedding._catboost_devices() == "0"


def test_legacy_gpu_alias_still_maps_to_tree_gpu_backends():
    tree_embedding = make_tree_embedding("gpu")

    assert tree_embedding._is_gpu_device()
    assert tree_embedding._xgboost_device() == "cuda"
    assert tree_embedding._catboost_task_type() == "GPU"
    assert tree_embedding._catboost_devices() == "0"


def test_cpu_device_maps_to_tree_cpu_backends():
    tree_embedding = make_tree_embedding("cpu")

    assert not tree_embedding._is_gpu_device()
    assert tree_embedding._xgboost_device() == "cpu"
    assert tree_embedding._catboost_task_type() == "CPU"
    assert tree_embedding._catboost_devices() == ""
