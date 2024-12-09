import logging
import os
from pathlib import Path
from functools import wraps
import time

def setup_logger(name, log_file, level=logging.INFO, log_format=None):
    """Configure a named logger with file handler
    
    Args:
        name (str): Logger name (usually __name__)
        log_file (str): Path to log file
        level (int): Logging level
        log_format (str): Optional custom format
    """
    # Create logs directory if needed
    log_dir = Path("logging")
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create file handler
    file_handler = logging.FileHandler(log_dir / log_file, encoding="utf-8")
    
    # Set format
    if log_format is None:
        log_format = "%(asctime)s:%(name)s:%(levelname)s:%(funcName)s:%(message)s"
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    return logger

def log_execution_time(func):
    """
    Decorator that logs only function execution times to a dedicated log file in the logging folder.
    """
    # Create logging directory if it doesn't exist
    log_dir = 'logging'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a specific logger for execution time
    time_logger = logging.getLogger('execution_timer')
    time_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    time_logger.handlers = []
    
    # Create file handler with path in logging directory
    log_file_path = os.path.join(log_dir, 'execution_time.log')
    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d at %H:%M:%S'
    )
    handler.setFormatter(formatter)
    time_logger.addHandler(handler)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        time_logger.info(
            f"Function '{func.__name__}' executed in {execution_time:.4f} seconds"
        )
        return result
    
    return wrapper