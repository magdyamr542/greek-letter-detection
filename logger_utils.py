import logging


def getLogger(file_path: str):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    # file handler
    fh = logging.FileHandler(file_path)
    logger.addHandler(fh)

    return logger
