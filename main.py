# encoding=utf-8
from __future__ import print_function
import traceback
import tensorflow as tf
import sys
from config import process_config
from logger import Logger
from dirs import create_dirs
from data_generator import DataGenerator
from model import DnnModel
from trainer import Trainer

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 0, "batch size")
flags.DEFINE_string("config_file", '../config/sess_slim_train.json', "config file in json format")
flags.DEFINE_string("current_dt", "s", "current_dt")
FLAGS = flags.FLAGS

def main(_):
    config = process_config(FLAGS.config_file)

    logger = Logger(config)
    create_dirs(config, FLAGS)

    data_generator = DataGenerator(logger, FLAGS, config)
    data_generator.build_data_spec()

    model = DnnModel(config, logger)

    trainer = Trainer(model=model, data=data_generator, config=config, logger=logger, flags=FLAGS)
    logger.logger.info("start")
    logger.logger.info("trainer.job_name=%s", trainer.job_name)
    try:
        if config['task'] == 'train':
            trainer.train()
        elif config['task'] == 'evaluate':
            trainer.evaluate_sklearn()
        else:
            logger.logger.info('task error')
    except Exception as e:
        exc_info = traceback.format_exc(sys.exc_info())
        msg = 'creating session exception:%s\n%s' % (e, exc_info)
        tmp = 'Run called even after should_stop requested.'
        should_stop = type(e) == RuntimeError and str(e) == tmp
        if should_stop:
            logger.logger.warn(msg)
        else:
            logger.logger.error(msg)
        exit_code = 0 if should_stop else 1
        sys.exit(exit_code)

if __name__ == "__main__":
    tf.app.run()

