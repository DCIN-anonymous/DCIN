# encoding=utf-8
from __future__ import print_function

import re

import pandas as pd
from datetime import datetime
import time
import random
import json
import os
import tensorflow as tf
import numpy as np

from stats import calc_pred_result


class Trainer(object):
    def __init__(self, model, data, config, logger, flags):
        # 为单机或分布式初始化环境
        self.model = model
        self.model_fn = model.model_fn
        self.data = data
        self.config = config
        self.logger = logger.logger
        self.flags = flags

        self.init_environment()
        self.init_sess_config()
        random.seed(datetime.now())
        t = int(time.time())
        tf.set_random_seed(t)

        self.estimator = self.custom_estimator()  # Estimator初始化

        self.shut_ratio = config['shut_ratio']
        self.slow_worker_delay_ratio = config['slow_worker_delay_ratio']

    def init_environment(self):
        if int(self.config['is_dist']) == 0:
            self.job_name = 'chief'
            self.task_index = 0
        else:
            self.job_name = self.flags.job_name
            self.task_index = self.flags.task_index

            self.ps_hosts = self.flags.ps_hosts.split(",")
            self.worker_hosts = self.flags.worker_hosts.split(",")
            self.chief_hosts = self.flags.chief_hosts.split(",")
            self.evaluator_hosts = self.flags.evaluator_hosts.split(",")
            if len(self.evaluator_hosts) > 0 and self.evaluator_hosts[0] != '' :
                self.cluster = {'chief': self.chief_hosts, "ps": self.ps_hosts,
                                "worker": self.worker_hosts, "evaluator": self.evaluator_hosts}
                self.has_evaluators = True
            else:
                self.cluster = {'chief': self.chief_hosts, "ps": self.ps_hosts,
                                "worker": self.worker_hosts}
                self.has_evaluators = False

            os.environ['TF_CONFIG'] = json.dumps(
                {'cluster': self.cluster,
                 'task': {'type': self.job_name, 'index':
                     self.task_index}})

            self.is_dispredict = int(self.config['is_dispredict'])

    def init_sess_config(self):
        parameter_servers_core = int(self.config['parameter_servers_core'])
        worker_core = int(self.config['worker_core'])
        chief_core = int(self.config['chief_core'])
        vcore = 5
        if self.job_name == 'chief':
            df = ['/job:chief', '/job:ps']
            vcore = chief_core
        elif self.job_name == 'worker':
            df = ["/job:%s/task:%d" % (self.job_name, self.task_index), "/job:ps"]
            vcore = worker_core
        elif self.job_name == 'ps':
            df = ['/job:ps', '/job:worker', '/job:master']
            vcore = parameter_servers_core
        elif self.job_name == 'evaluator':
            df = ["/job:%s/task:%d" % (self.job_name, self.task_index), "/job:ps"]
        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False,
                                        inter_op_parallelism_threads=vcore,
                                        intra_op_parallelism_threads=vcore,
                                        device_filters=df)
        session_config.gpu_options.allow_growth = True

        if int(self.config['debug']) != 1:
            tf.disable_chief_training(shut_ratio=float(self.config['shut_ratio']),
                                      slow_worker_delay_ratio=float(self.config['slow_worker_delay_ratio']))

        tf.enable_persistent_metric()
        if self.is_dispredict == 1 and 'train' in self.config['task'] and self.has_evaluators:
            self.train_config = tf.estimator.RunConfig(
                save_summary_steps=int(self.config['save_summary_steps']),
                save_checkpoints_secs=int(self.config['save_checkpoints_secs']),
                model_dir=self.config['model_ckpt_dir'],
                session_config=session_config,
                keep_checkpoint_max=int(self.config['keep_checkpoint_max']),
                eval_mode='train_and_dist_eval'
            )
        else:
            self.train_config = tf.estimator.RunConfig(
                save_summary_steps=int(self.config['save_summary_steps']),
                save_checkpoints_secs=int(self.config['save_checkpoints_secs']),
                model_dir=self.config['model_ckpt_dir'],
                session_config=session_config,
                keep_checkpoint_max=int(self.config['keep_checkpoint_max']),
            )

    def custom_estimator(self):
        use_pipeline = int(self.config.get('use_pipeline', '0')) > 0
        if use_pipeline:  # use pipeline
            tf.enable_pipeline(data_limit_size=10, data_limit_wait_ms=100)

        if len(pretrain_model_dir) > 10:
            ws = tf.estimator.WarmStartSettings(
                ckpt_to_initialize_from=pretrain_model_dir,
                vars_to_warm_start=['base_hashtable'],
            )
            return tf.estimator.Estimator(
                model_fn=self.model_fn,  # First-class function
                config=self.train_config,  # RunConfig
                warm_start_from=ws
            )
        else:
            return tf.estimator.Estimator(
                model_fn=self.model_fn,  # First-class function
                config=self.train_config  # RunConfig
            )

    def train(self):
        if int(self.config['is_dist']) == 1:
            train_spec, eval_spec = self.data.train_spec, self.data.eval_spec
            tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)
        else:
            self.estimator.train(self.data.train_input_fn)

    def evaluate_sklearn(self):
        features, labels = self.data.eval_input_fn()
        predictions = self.estimator.model_fn(features, labels, tf.estimator.ModeKeys.PREDICT,
                                              self.train_config).predictions
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.config['log_dir'])
            saver.restore(sess, ckpt.model_checkpoint_path)
            prediction_values = {"rectified_pctr": [], "ori_pctr": []}
            label_values = []
            idx = 0
            while True:
                try:
                    preds, lbls = sess.run([predictions, labels])
                    for key, value in preds.items():
                        prediction_values[key].extend(value)
                    label_values.extend(lbls)
                    idx = idx + 1
                    if idx % 100 == 0:
                except tf.errors.OutOfRangeError:
                    break
            statistics = calc_pred_result(label_values, prediction_values)
            statistics['current_dt'] = self.config['current_dt']
            self.save_eval_result(statistics)

    def evaluate(self):
        statistics = self.estimator.evaluate(input_fn=self.data.eval_input_fn)
        self.save_eval_result(statistics)

