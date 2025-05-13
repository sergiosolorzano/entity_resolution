import logging
import traceback

class Tne_Logger:

    def __init__(self, name="tne_log"):
        self.name = name
        self._logger = logging.getLogger(name)
        # self.logger = logging.getLogger(self.name)
        self._logger.setLevel(logging.DEBUG)

        if not self._logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('[%(levelname)s]: %(message)s')
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)