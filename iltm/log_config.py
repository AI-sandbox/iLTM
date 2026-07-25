import logging


def setup_logging(logging_level=logging.DEBUG):
    """Set the iLTM log level without modifying application-wide logging."""
    logging.getLogger("iltm").setLevel(logging_level)
