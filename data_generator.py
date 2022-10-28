# coding=utf-8
import os, sys
import hashlib
import random
import tensorflow as tf
import functools
import math


class DataGenerator:
    def __init__(self, logger, flags, config):
        self.logger = logger.logger
        self.flags = flags
        self.config = config

        self.file_names = []

        self.train_spec = None
        self.eval_spec = None

    def get_input_filenames(self, data_dir):
        file_names = []
        file_list = tf.gfile.ListDirectory(data_dir)
        for current_file_name in file_list:
            file_path = os.path.join(data_dir, current_file_name)
            file_names.append(file_path)
        self.logger.info('all files: %s' % file_names)
        random.shuffle(file_names)
        return file_names

    def build_data_spec(self):

        if (int(self.config['is_dist']) == 0 or self.flags.job_name in ['worker', 'evaluator']) and int(
                self.config['is_data_dispatch']) == 0:
            self.get_input_filenames('inputs')
        else:
            # ps 节点
            trainset_file_names = []
            validset_file_names = []

        trainset_file_names = self.file_names
        validset_file_names = self.file_names
        self.logger.info('trainset_file_names %s', trainset_file_names)

        train_config = self.config.copy()

        hooks = []
        if int(self.config['debug']) and self.flags.job_name == "worker" and self.flags.task_index == 0:
            profile_hook = tf.train.ProfilerHook(save_steps=1000, output_dir=self.config['log_dir'],
                                                 show_memory=False) 
            hooks.append(profile_hook)

        self.train_input_fn = functools.partial(self.input_fn, trainset_file_names, train_config)
        self.train_spec = tf.estimator.TrainSpec(self.train_input_fn, hooks=hooks)

        self.logger.info('validset_file_names %s', validset_file_names)

        self.eval_input_fn = functools.partial(self.input_fn, validset_file_names, self.config)
        self.eval_spec = tf.estimator.EvalSpec(self.eval_input_fn, throttle_secs=int(4500))

    # 本地生成的tfrecord格式
    def input_fn(self, file_names, config):

        self.logger.info('file_names: %s', file_names)
        self.logger.info('batch_size: %s', config['batch_size'])
        self.logger.info('predict_trick_cate_fea: %s', config['predict_trick_cate_fea'])

        def parse_fn(serialized_example):
            example_protocal = {
                "cf": tf.FixedLenFeature([int(config['cate_fea_num'])], dtype=tf.int64),
                "nf": tf.FixedLenFeature([int(config['dense_fea_num'])], dtype=tf.float32),
                "isclick": tf.FixedLenFeature([1], dtype=tf.float32)
            }
            example_protocal["poi_id_int64"] = tf.FixedLenFeature([1], dtype=tf.int64)
            example_protocal["page_click_poi_list"] = tf.FixedLenFeature([20], dtype=tf.int64)
            example_protocal["page_view_poi_list"] = tf.FixedLenFeature([160], dtype=tf.int64)
            example_protocal["prerank_poi_list_int64_v2"] = tf.FixedLenFeature([50], dtype=tf.int64)

            parsed_features = tf.parse_example(
                serialized=serialized_example,
                features=example_protocal
            )
            return parsed_features, parsed_features['isclick']

        files = tf.data.Dataset.list_files(file_names, shuffle=False)
        dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=8))
        dataset = dataset.shuffle(buffer_size=int(config['batch_size']), reshuffle_each_iteration=True)
        dataset = dataset.repeat(int(config['epoch']))
        dataset = dataset.batch(int(config['batch_size']))
        dataset = dataset.map(parse_fn, num_parallel_calls=8)
        dataset = dataset.prefetch(buffer_size=1)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

