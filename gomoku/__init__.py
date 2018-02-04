import logging.handlers
import os

handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "gomoku.log"))
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
gomoku = logging.getLogger()
gomoku.setLevel(os.environ.get("LOGLEVEL", "INFO"))
gomoku.addHandler(handler)
