#encoding=utf-8
import os, sys
import logging

class Logger:
    def __init__(self, config, sess=None):
       self.sess = sess
       self.config = config
       self.summary_placeholders = {}
       self.summary_ops = {}
       self.logger = self.set_logger()

    def set_logger(self):
        logger = logging.getLogger("tensorflow")
        if len(logger.handlers) == 1:
            logger.handlers = []
            logger.setLevel(logging.INFO)

            formatter = logging.Formatter(
                "%(asctime)s - [%(filename)s:%(lineno)d] - %(name)s - %(levelname)s - %(message)s")

            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)

            fh = logging.FileHandler('tensorflow.log')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)

            logger.addHandler(ch)
            logger.addHandler(fh)

        return logger
