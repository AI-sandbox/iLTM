import logging

from iltm import iLTMRegressor


def test_estimator_construction_preserves_root_logging():
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    iltm_logger = logging.getLogger("iltm")
    original_iltm_level = iltm_logger.level
    application_handler = logging.NullHandler()
    root_logger.addHandler(application_handler)
    root_logger.setLevel(logging.CRITICAL)

    try:
        iLTMRegressor(
            checkpoint=None,
            device="cpu",
            logging_level=logging.INFO,
        )

        assert root_logger.handlers == original_handlers + [application_handler]
        assert root_logger.level == logging.CRITICAL
        assert iltm_logger.level == logging.INFO
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
        iltm_logger.setLevel(original_iltm_level)
