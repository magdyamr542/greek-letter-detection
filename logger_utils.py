import logging


def getLogger(file_path: str):
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
    fh = logging.FileHandler(file_path, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
