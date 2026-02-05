import logging
import os
import time


def get_logger(name="unnamed", log_dir: str = None) -> logging.Logger:
    """Get a logger

    Args:
        name (str, optional): the name of the logger. Defaults to "unnamed".
        log_dir (str, optional): the directory to save the logs. Defaults to None.

    Returns:
        logging.Logger: the logger
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s::%(name)s::%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_and_create_new_log_dir(root="./logs", prefix="", suffix="") -> str:
    """Get and create a new log directory

    Args:
        root (str, optional): the root directory to save the logs. Defaults to "./logs".
        prefix (str, optional): the prefix of the log directory. Defaults to "".
        suffix (str, optional): the suffix of the log directory. Defaults to "".

    Returns:
        str: the new log directory
    """
    filename = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    if prefix != "":
        filename = prefix + "_" + filename
    if suffix != "":
        filename = filename + "_" + suffix
    log_dir = os.path.join(root, filename)
    os.makedirs(log_dir)
    return log_dir


if __name__ == "__main__":
    log_dir = get_and_create_new_log_dir(root="./logs", prefix="test", suffix="")
    logger = get_logger(name="yyy", log_dir=log_dir)
    logger.info("archlinux")
    logger.debug("ubuntu")
