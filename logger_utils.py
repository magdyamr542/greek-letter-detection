import logging
from typing import Optional


def getLogger(file_path: Optional[str] = None) -> logging.Logger:
    # create logger
    logging.basicConfig(filemode="w")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # console handler
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # file handler
    if file_path:
        fh = logging.FileHandler(file_path, mode="w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
