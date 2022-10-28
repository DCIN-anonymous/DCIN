#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import metrics

BIGMODEL_EMBED_TABLE = "bigmodel_embed_table"
BIGMODEL_EMBED_HASH_TABLE = "bigmodel_embed_hash_table"


def get_mutable_dense_hashtable(key_dtype,
                                value_dtype,
                                shape,
                                initializer=None,
                                empty_key=-1,
                                shard_num=1,
                                initial_num_buckets=None,
                                shared_name=None,
                                default_value=None,
                                fusion_optimizer_var=False,
                                export_optimizer_var=False,
                                table_impl_type="tbb",
                                name="get_mutable_dense_hashtable",
                                checkpoint=True,
                                enable_bigmodel=False):
    if enable_bigmodel:
        name = name
        embed_dim = shape.as_list()[-1]
        embed_table = tf.get_variable(
            name,
            shape=[2, embed_dim],
            trainable=False,
            collections=[BIGMODEL_EMBED_TABLE],
            initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001),
        )

        tf.add_to_collection(BIGMODEL_EMBED_TABLE, embed_table)

        return embed_table
    else:
        embed_table = tf.contrib.lookup.get_mutable_dense_hashtable(
            key_dtype=key_dtype,
            value_dtype=value_dtype,
            shape=shape,
            initializer=initializer,
            empty_key=empty_key,
            shard_num=shard_num,
            initial_num_buckets=initial_num_buckets,
            shared_name=shared_name,
            default_value=default_value,
            fusion_optimizer_var=fusion_optimizer_var,
            export_optimizer_var=export_optimizer_var,
            table_impl_type=table_impl_type,
            name=name,
            checkpoint=checkpoint
        )

        tf.add_to_collection(BIGMODEL_EMBED_HASH_TABLE, embed_table)
        return embed_table


def embedding_lookup_hashtable(emb_tables,
                               ids,
                               is_training=None,
                               name=None,
                               threshold=0,
                               serving_default_value=None,
                               enable_bigmodel=False,
                               dump_count_table_to_ckpt=False):
    if enable_bigmodel:
        emb = tf.nn.embedding_lookup(emb_tables, ids)
        return emb
    else:
        emb = tf.nn.embedding_lookup_hashtable_v2(emb_tables,
                                                  ids,
                                                  is_training=is_training,
                                                  threshold=threshold,
                                                  serving_default_value=serving_default_value,
                                                  dump_count_table_to_ckpt=dump_count_table_to_ckpt)
        return emb

