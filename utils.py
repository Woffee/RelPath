import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logger():
    """
    Sets up the logger with a file handler.
    :return: Configured logger.
    """
    # Generate a timestamped log file name
    log_path = "logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # Create a file handler that logs debug and higher level messages
    handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=2)
    handler.setLevel(logging.DEBUG)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handlers to the logger
    if not logger.hasHandlers():
        logger.addHandler(handler)

    return logger


# Set up the logger
logger = setup_logger()


def log_message(message, level='info'):
    """
    Logs a message to the log file.
    :param message: The message to log.
    :param level: The level of the log (debug, info, warning, error, critical).
    """
    if level == 'debug':
        logger.debug(message)
    elif level == 'info':
        logger.info(message)
    elif level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)
    elif level == 'critical':
        logger.critical(message)
    else:
        logger.info(message)
