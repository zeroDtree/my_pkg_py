import logging
import time
import os


def get_logger(name="unnamed", log_dir=None):
    logger = logging.getLogger(name)
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


def get_and_create_new_log_dir(root="./logs", prefix="", tag=""):
    fn = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    if prefix != "":
        fn = prefix + "_" + fn
    if tag != "":
        fn = fn + "_" + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


if __name__ == "__main__":
    log_dir = get_and_create_new_log_dir(root="./logs", prefix="test", tag="")
    logger = get_logger(log_dir=log_dir)
    logger.info("archlinux")
    logger.debug("ubuntu")
