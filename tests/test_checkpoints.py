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

