import pytest
import tempfile
import os
from pathlib import Path

from iltm.model_checkpoints import (
    resolve_model_checkpoint,
    get_model_checkpoint_config,
    _get_default_ckpt_dir
)


class TestCheckpointResolution:
    
    def test_resolve_known_checkpoint_names(self):
        known_checkpoints = [
            "xgbrconcat", "cbrconcat"
        ]
        
        for name in known_checkpoints:
            config = get_model_checkpoint_config(name)
            assert 'checkpoint' in config
            assert isinstance(config['checkpoint'], str)
    
    def test_resolve_unknown_checkpoint_name(self):
        with pytest.raises(ValueError, match="Unknown model name suffix"):
            get_model_checkpoint_config("unknown_model_name")
    
    def test_resolve_existing_file(self):
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            config = resolve_model_checkpoint(tmp_path)
            assert config['checkpoint'] == tmp_path
        finally:
            os.unlink(tmp_path)
    
    def test_resolve_directory_with_pth_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pth_file = os.path.join(tmpdir, "model.pth")
            Path(pth_file).touch()
            
            config = resolve_model_checkpoint(tmpdir)
            assert 'checkpoint' in config
            assert config['checkpoint'].endswith('.pth')

    @pytest.mark.parametrize("name", [
        "xgbrconcat", "cbrconcat", "r128bn", "rnobn", "xgb", "catb", "rtr", "rtrcb",
    ])
    @pytest.mark.parametrize("as_directory", [False, True])
    def test_local_location_preserves_named_checkpoint_settings(
        self, monkeypatch, tmp_path, name, as_directory,
    ):
        # The directory name must not override the identity in the filename.
        directory = tmp_path / "cbrconcat" / "rtr"
        directory.mkdir(parents=True)
        path = directory / f"{name}.pth"
        path.touch()
        monkeypatch.setattr(
            "iltm.model_checkpoints._ensure_checkpoint",
            lambda *_args: str(path),
        )
        expected = get_model_checkpoint_config(name)

        def fail_download(*_args, **_kwargs):
            pytest.fail("Resolving an existing local file must not download weights")

        monkeypatch.setattr("iltm.model_checkpoints._ensure_checkpoint", fail_download)
        supplied_path = str(directory if as_directory else path)
        actual = resolve_model_checkpoint(supplied_path)
        assert actual == expected
        if not as_directory:
            assert actual["checkpoint"] is supplied_path
    
    def test_get_default_ckpt_dir(self):
        ckpt_dir = _get_default_ckpt_dir()
        assert isinstance(ckpt_dir, str)
        assert len(ckpt_dir) > 0
        assert os.path.exists(ckpt_dir)
    
    def test_checkpoint_config_contents(self):
        config = get_model_checkpoint_config("xgbrconcat")
        
        assert 'checkpoint' in config
        assert 'preprocessing' in config
        assert 'tree_embedding' in config
        assert 'tree_model' in config
        assert 'concat_tree_with_orig_features' in config
        
        assert config['preprocessing'] == 'realmlp_td_s_v0'
        assert config['tree_embedding'] is True
        assert config['tree_model'] == 'XGBoost_hist'
        assert config['concat_tree_with_orig_features'] is True

    def test_only_concat_checkpoints_include_original_features(self, monkeypatch):
        monkeypatch.setattr(
            "iltm.model_checkpoints._ensure_checkpoint",
            lambda _repo_id, filename, _ckpt_dir: filename,
        )

        checkpoint_names = (
            "r128bn",
            "rnobn",
            "catb",
            "xgb",
            "rtr",
            "rtrcb",
            "cbrconcat",
            "xgbrconcat",
        )
        for checkpoint_name in checkpoint_names:
            config = get_model_checkpoint_config(checkpoint_name)
            assert config["concat_tree_with_orig_features"] is (
                "concat" in checkpoint_name
            )
