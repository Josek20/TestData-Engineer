import logging
from functools import wraps
from time import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"An error occurred in function '{func.__name__}': {str(e)}")
            raise
    return wrapper


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        execution_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper
