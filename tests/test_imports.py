import pytest


class TestImports:
    
    def test_import_main_module(self):
        import iltm
        assert iltm is not None
    
    def test_import_classifier(self):
        from iltm import iLTMClassifier
        assert iLTMClassifier is not None
    
    def test_import_regressor(self):
        from iltm import iLTMRegressor
        assert iLTMRegressor is not None
    
    def test_import_version(self):
        from iltm import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)
    
    def test_all_exports(self):
        from iltm import __all__
        assert 'iLTMClassifier' in __all__
        assert 'iLTMRegressor' in __all__
        assert '__version__' in __all__
    
    def test_submodule_imports(self):
        from iltm import inference_interface
        from iltm import iltm_model
        from iltm import utils
        from iltm import tree_embedding
        from iltm import model_checkpoints
        
        assert inference_interface is not None
        assert iltm_model is not None
        assert utils is not None
        assert tree_embedding is not None
        assert model_checkpoints is not None

