import logging
from pathlib import Path

def setup_logger(log_file: Path):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s"
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger