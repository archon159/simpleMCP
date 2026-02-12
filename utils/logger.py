import logging

def create_logger(
    filename='cur.log',
):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger