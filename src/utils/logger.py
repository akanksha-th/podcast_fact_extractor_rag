"""
Logging System
"""

import logging, sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """ Custom formatter to add colors to log levels 
    """
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m'  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)
    

def setup_logging(
        name: str = "podcast_rag",
        log_level: str = "INFO",
        log_file: Optional[Path] = None,
        console_output: bool = True
        ) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    if logger.handlers:
        return logger
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt = "%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt = "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(name: str) -> logging.Logger:
    try:
        from config import settings
        log_file = settings.get_log_file_path()
        log_level = settings.log_level
    except ImportError:
        log_file = None
        log_level = "INFO"

    return setup_logging(
        name = name, 
        log_level = log_level,
        log_file = log_file,
        console_output = True
        )


class LoggerContext:
    """ Context manager for logging operations"""
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting operation: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type is not None:
            self.logger.error(
                f"Operation {self.operation} failed after {duration:.2f} seconds",
                exc_info=(exc_type, exc_value, exc_tb)
            )
            return False
        
        self.logger.info(f"Finished operation: {self.operation}")
        return True
    

if __name__ == "__main__":
    logger = get_logger("test_logger")

    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")

    with LoggerContext(logger, "processing data"):
        import time
        time.sleep(1)
        logger.info("Doing some work ...")

    try:
        raise ValueError("Example error.")
    except Exception as e:
        logger.error("An error occurred", exc_info=True)

    print("\nLogging demo complete. Check the logs/ directory for detailed logs.")    