import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")  # Create a dedicated logs directory
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Creates a logger with a consistent configuration and timestamped filename.

    Args:
        name (str): The name of the logger.
        log_level (int, optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: The configured logger.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:  # Avoid duplicate handlers
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        log_file = os.path.join(LOG_DIR, f"{timestamp}.log")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(log_level)

    return logger