import os
import logging
from datetime import datetime


def setup_logger(log_dir="./logs", name="crc"):
    """同时输出到 console 和文件的 logger。"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.propagate = False
    return logger
