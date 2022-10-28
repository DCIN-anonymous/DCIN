# encoding=utf-8
import json
import math
import stats

from tensorflow.python.ops import array_ops
import tensorflow as tf
from tensorflow.contrib.opt import HashAdamOptimizer, LazyAdamOptimizer, HashAdagradOptimizer
import tensorflow.contrib as contrib


class DnnModel:
    def __init__(self, config, logger):
        self.logger = logger.logger
        self.config = config
        self.model_fn = self.model_fn

        self._ps_num = int(self.config['ps_num'])
        self._partitioner_set = {'fix', 'var', 'min_max'}
        _partitioner = self.config.get('partitioner', 'fix')
        self._partitioner = _partitioner if _partitioner in self._partitioner_set else 'fix'
        self._max_shard_bytes = int(self.config.get('max_shard_bytes', 512 << 10))
        self._min_slice_size = int(self.config.get('min_slice_size', 32 << 10))

        self.kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        self.bias_initializer = tf.zeros_initializer()
        self.split_sizes = [0,0]

    def get_partitioner(self, partitioner_type=None):
        partitioner_type = partitioner_type if partitioner_type in self._partitioner_set else self._partitioner
        if partitioner_type == 'fix':
            return tf.fixed_size_partitioner(self._ps_num)
        elif partitioner_type == 'var':
            return tf.variable_axis_size_partitioner(max_shard_bytes=self._max_shard_bytes, max_shards=self._ps_num)
        elif partitioner_type == 'min_max':
            return tf.min_max_variable_partitioner(max_partitions=self._ps_num, min_slice_size=self._min_slice_size)
        else:
            raise RuntimeError('unknown partitioner type "%s"' % partitioner_type)

    def table_lookup(self, table, is_training, list_ids, threshold, v_name, flatten, embedding_size,
                     dump_count_table_to_ckpt=False, model_export=-999):
        model_graph = int(self.config['model_graph'])
        model_export = int(self.config['model_export'])
        if model_export < 0:
            model_export = int(self.config['model_export'])

        def _do_lookup(ids):
            if model_export:
                return stats.embedding_lookup_hashtable(
                    emb_tables=table,
                    ids=ids, is_training=is_training,
                    threshold=threshold,
                    dump_count_table_to_ckpt=dump_count_table_to_ckpt,
                    serving_default_value=array_ops.zeros(embedding_size, tf.float32),
                    enable_bigmodel=model_export
                )
            else:
                if model_graph:
                    _embed = stats.embedding_lookup_hashtable(
                        emb_tables=table,
                        ids=ids,
                        is_training=is_training,
                        threshold=threshold,
                        dump_count_table_to_ckpt=dump_count_table_to_ckpt,
                        serving_default_value=array_ops.zeros(embedding_size, tf.float32),
                        enable_bigmodel=model_export
                    )
                    return _embed
                else:
                    _embed = stats.embedding_lookup_hashtable(
                        emb_tables=table,
                        ids=_unique_ids,
                        is_training=is_training,
                        dump_count_table_to_ckpt=dump_count_table_to_ckpt,
                        threshold=threshold,
                        serving_default_value=array_ops.zeros(
                            embedding_size, tf.float32),
                    )
                    return _embed

        if not isinstance(list_ids, list):
            list_ids = [list_ids]

        len_list = len(list_ids)
        if model_graph:
            list_embed = []
            for ids in list_ids:
                num = ids.shape[1].value
                uniq_ids = tf.reshape(ids, [-1])
                _embed = _do_lookup(uniq_ids)
                if flatten:
                    _embed = tf.reshape(_embed, [-1, num * embedding_size])
                else:
                    _embed = tf.reshape(_embed, [-1, num, embedding_size])
                list_embed.append(_embed)
            message = 'merged table_lookup:\n\t%s' % v_name
            for i in range(len_list):
                message += '\n\t%d\n\t\t%s\n\t\t%s' % (i, list_ids[i], list_embed[i])
            return list_embed

        else:
            list_of_ids = [tf.cast(ids, tf.int64) for ids in list_ids]
            list_of_size = [tf.size(ids) * embedding_size for ids in list_of_ids]
            list_of_num = [ids.shape[1].value for ids in list_of_ids]
            _concat_ids = tf.concat([tf.reshape(ids, [-1]) for ids in list_of_ids], axis=0)
            _embed = _do_lookup(_concat_ids)
            list_embed = tf.split(tf.reshape(_embed, [-1]), list_of_size, axis=0)
            if flatten:
                list_embed = [tf.reshape(list_embed[i], [-1, list_of_num[i] * embedding_size]) for i in range(len_list)]
            else:
                list_embed = [tf.reshape(list_embed[i], [-1, list_of_num[i], embedding_size]) for i in range(len_list)]
            message = 'merged table_lookup:\n\t%s' % v_name
            for i in range(len_list):
                message += '\n\t%d\n\t\t%s\n\t\t%s' % (i, list_ids[i], list_embed[i])
            return list_embed

    
    def model_fn(self, features, labels, mode):
        is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False
        
        processed_features = self.process_fea(features, is_training)

        logits = self.inference(processed_features, "top_mlp_ctr", is_training)
        pctr = tf.nn.sigmoid(logits, name="predict")

        ctr_label = features['isclick']
        predictions = {
            "pctr": pctr,
            "labels": tf.reshape(ctr_label, [-1, 1])
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        labels = tf.cast(labels, tf.float32)
        labels = tf.reshape(labels, [-1, 1])
        logits = tf.reshape(logits, [-1, 1])
        loss = self.apply_loss(logits, labels)

        auc = tf.metrics.auc(labels=labels, predictions=pctr)
        mae = tf.metrics.mean_absolute_error(labels=labels, predictions=pctr)
        mean_pctr = tf.metrics.mean(pctr)
        mean_ctr = tf.metrics.mean(labels)

        tf.summary.scalar('train-auc', auc[1])
        tf.summary.scalar('train-loss', loss)
        tf.summary.scalar('train-mae', mae[1])
        tf.summary.scalar('train-ctr', mean_ctr[1])
        tf.summary.scalar('train-pctr', mean_pctr[1])

        train_op = self.apply_optimizer(loss, float(self.config['learning_rate']),
                                        float(self.config['lazyAdam_learning_rate']), self.config['optimizer'])

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    def process_fea(self, features, is_training):
        base_embed_dim = int(self.config['embed_dim'])
        cate_fea_num = int(self.config['cate_fea_num'])
        dense_fea_num = int(self.config['dense_fea_num'])

        din_deep_layers = map(int, self.config['din_deep_layers'].split('_'))
        din_activation = self.config['din_activation']
        embed_fea_num = int(self.config['embed_fea_num'])
        cate_fea_col = features['cf']
        numerical_fea_col = features['nf']
        batch_size = int(self.config['batch_size'])
        ps_num = int(self.config['ps_num'])
        model_export = int(self.config['model_export'])

        r = math.sqrt(6 / base_embed_dim)
        can_input_dim = 8
        can_weight_dim = can_input_dim * can_input_dim
        can_bias_dim = can_input_dim
        can_mlp_layer = 2
        can_mlp_emb_dim = (can_weight_dim + can_bias_dim) * can_mlp_layer
        
        base_hashtable = stats.get_mutable_dense_hashtable(
            key_dtype=tf.int64,
            value_dtype=tf.float32,
            shape=tf.TensorShape(base_embed_dim),
            name='base_hashtable',
            empty_key=tf.int64.max,
            initializer=tf.random_uniform_initializer(minval=-r, maxval=r),
            shard_num=ps_num,
            fusion_optimizer_var=fusion,
            enable_bigmodel=model_export,
            export_optimizer_var=False
        )
        base_hashtable_threshold = 5

        can_target_hashtable = stats.get_mutable_dense_hashtable(
            key_dtype=tf.int64,
            value_dtype=tf.float32,
            shape=tf.TensorShape(can_mlp_emb_dim),
            name='can_target_hashtable',
            empty_key=tf.int64.max,
            initializer=tf.random_uniform_initializer(minval=-math.sqrt(6 / float(can_mlp_emb_dim)),
                                                      maxval=math.sqrt(6 / float(can_mlp_emb_dim))),
            shard_num=ps_num,
            fusion_optimizer_var=fusion,
            enable_bigmodel=model_export,
            export_optimizer_var=False
        )
        can_target_hashtable_threshold = 5

        can_user_hashtable = stats.get_mutable_dense_hashtable(
            key_dtype=tf.int64,
            value_dtype=tf.float32,
            shape=tf.TensorShape(can_input_dim),
            name='can_user_hashtable',
            empty_key=tf.int64.max,
            initializer=tf.random_uniform_initializer(minval=-r, maxval=r),
            shard_num=ps_num,
            fusion_optimizer_var=fusion,
            enable_bigmodel=model_export,
            export_optimizer_var=False
        )
        can_user_hashtable_threshold = 5

        with tf.name_scope('input'):
            embed_base_ids = [
                tf.concat([tf.cast(cate_fea_col, tf.int64)),
            ]

            embed_cate_fea_col = self.table_lookup(base_hashtable, is_training, embed_base_ids, base_hashtable_threshold,
                                    "embed_base_ids", flatten=True, embedding_size=base_embed_dim)

            # din_table_ids = [
            # din_output = self.din_layer(

            total_cate_fea_num = cate_fea_num
            reshaped_embed_cate_fea_col = tf.reshape(embed_cate_fea_col, [-1, total_cate_fea_num * base_embed_dim])

            normalized_numerical_fea_col = tf.layers.batch_normalization(inputs=numerical_fea_col, training=is_training,
                                                                         name='num_fea_batch_norm')
            #can_cur_poi_ids = [
            #can_poi_list_table_ids = [

            # Explicit Decision-Making Context Modeling
            page_num=int(self.config['page_num'])
            page_len=int(self.config['page_len'])
            # table lookup
            page_click_poi_list = features['page_click_poi_list'][:, :page_num]
            page_view_poi_list = features['page_view_poi_list'][:, :page_num*page_len]
            click_page_table_ids = [
                tf.cast(page_click_poi_list, tf.int64),
                tf.cast(page_view_poi_list, tf.int64)
            ]
            page_click_poi_list_emb, page_view_poi_list_emb = self.table_lookup(
                                                            base_hashtable, is_training, click_page_table_ids, base_hashtable_threshold,
                                                             "click_page_poi_list_emb", flatten=False, embedding_size=base_embed_dim)
            page_click_poi_list_emb = tf.reshape(page_click_poi_list_emb, [-1, page_num, 1, base_embed_dim])
            page_view_poi_list_emb = tf.reshape(page_view_poi_list_emb, [-1, page_num, page_len, base_embed_dim])
            augmented_click_embs = self.context_modeling(page_click_poi_list_emb, page_view_poi_list_emb, int(self.config['explicit_k']), 'explicit_modeling')
            augmented_click_embs = tf.reshape(augmented_click_embs, [-1, page_num, base_embed_dim])

            # Implicit Decision-Making Context Modeling
            # target item
            poi_id_int64_fea_col = tf.cast(features['poi_id_int64'], tf.int64)
            poi_id = [tf.cast(poi_id_int64_fea_col, tf.int64),]
            poi_id_int64_emb = self.table_lookup(base_hashtable, is_training, poi_id, base_hashtable_threshold, "poi_emb", flatten=False, embedding_size=base_embed_dim)
            poi_id_int64_emb = tf.reshape(poi_id_int64_emb, [-1, 1, 1, base_embed_dim])
            # pre-ranking candidates
            prerank_num=int(self.config['prerank_poi_num'])
            prerank_poi_list_int64_fea_col = tf.cast(features['prerank_poi_list_int64_v2'], tf.int64)[:, :prerank_num]
            prerank_ids = [prerank_poi_list_int64_fea_col,]
            prerank_poi_list_int64_emb = self.table_lookup(base_hashtable, is_training, prerank_ids, base_hashtable_threshold, 'prerank_ids', flatten=False, embedding_size=base_embed_dim)
            prerank_poi_list_int64_emb = tf.reshape(prerank_poi_list_int64_emb, [-1, 1, prerank_num, base_embed_dim])
            augmented_target_emb = self.context_modeling(poi_id_int64_emb, prerank_poi_list_int64_emb, int(self.config['implicit_k']), 'implicit_modeling')      # b, 1, 1, d
            augmented_target_emb = tf.reshape(augmented_target_emb, [-1, 1, base_embed_dim])

            aiau_output = self.aiau(augmented_target_emb, augmented_click_embs, 'aiau')

            processed_features = tf.concat(
                [
                    reshaped_embed_cate_fea_col,
                    normalized_numerical_fea_col,
#                    din_output,
#                    din_clk_output,
#                    can_click_poi_list_output,
#                    can_order_poi_list_output,
                    aiau_output
                ],
                axis=1)

            return processed_features

    def context_modeling(self, target_emb, context_embs, k, variable_scope):
        '''
            Input
                target_emb:     embedding to be augmented, e.g., clicked item embedding or target item embedding, with shape [b, t, 1, d]
                context_embd:   embeddings used to augment target_emb, e.g., context items in one page or pre-ranking candidates, with shape [b, t, n, d]
                k:              num of relevant items to be kept, integer
            Output
                aug_target_emb: the augmented target embedding, with shape [b, t, 1, d]
        '''
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
            relevance = self.isu(target_emb, context_embs, k)                   # b, t, 1, n
            aug_target_emb = self.riu(target_emb, context_embs, relevance)
        return aug_target_emb

    def isu(self, target_emb, context_embs, k):
        isu_deep_layers = map(int, self.config['isu_deep_layers'].split('_'))
        activation = tf.nn.relu if self.config['isu_activation'] == "relu" else tf.nn.tanh

        b, t, n, d = context_embs.shape.as_list()
        target_embs = tf.tile(target_emb, [1, 1, n, 1])      # [b, t, n, d]
        input_layer = tf.concat([target_embs, context_embs, target_embs - context_embs, target_embs * context_embs], axis=-1)       # [b, t, n, 4*d]
        for i in range(len(isu_deep_layers)):
            deep_layer = tf.layers.dense(input_layer, int(isu_deep_layers[i]), activation=activation,
                                         partitioner=self.get_partitioner('min_max'), name='mlp_%d'%(i))
            input_layer = deep_layer
        isu_output_layer = tf.layers.dense(input_layer, 1, activation=None, name='mlp_out')                                         # [b, t, n, 1]
        isu_output_layer = tf.reshape(isu_output_layer, [-1, t, 1, n])                                                              # [b, t, 1, n]
        relevance, _ = tf.math.top_k(isu_output_layer, k=k, sorted=False)
        kth = tf.reduce_min(relevance, axis=-1, keepdims=True)
        topk = tf.greater_equal(isu_output_layer, kth)
        final_relevance = tf.cast(~topk,dtype=tf.float32)*(-1e10) + isu_output_layer                                                # [b, t, 1, n]
        return final_relevance
   
    def riu(self, target_emb, context_embs, relevance):
        b, t, n, d = context_embs.shape.as_list()
        Q = tf.layers.dense(target_emb, d, name='query')*(d ** (-0.5))          # b, t, 1, d
        K = tf.layers.dense(context_embs, d, name='key')                        # b, t, n, d
        V = tf.layers.dense(context_embs, d, name='value')
        attn = tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2]))
        attn += relevance
        attn = tf.nn.softmax(attn)
        attn_out = tf.matmul(attn, V)
        attn_out = tf.layers.dense(attn_out, d, name='proj')
        attn_out += target_emb
        return attn_out

    def aiau(self, target_emb, click_embs, variable_scope):
        aiau_deep_layers1 = map(int, self.config['aiau_deep_layers1'].split('_'))
        aiau_deep_layers2 = map(int, self.config['aiau_deep_layers2'].split('_'))
        activation = tf.nn.relu if self.config['aiau_activation'] == "relu" else tf.nn.tanh
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
            b, t, d = click_embs.shape.as_list()
            target_embs = tf.tile(target_emb, [1, t, 1])      # [b, t, d]
            input_layer = tf.concat([target_embs, click_embs], axis=-1)       # [b, t, 2*d]
            for i in range(len(aiau_deep_layers1)):
                deep_layer = tf.layers.dense(input_layer, int(aiau_deep_layers1[i]), activation=activation,
                                          partitioner=self.get_partitioner('min_max'),
                                          name='mlp1_%d' % (i))
                input_layer = deep_layer
            aligned_fea = tf.layers.dense(input_layer, d, activation=None, name='mlp1_out')
            Q = tf.layers.dense(aligned_fea, d, name='query')*(d ** (-0.5))       # b, t, d
            K = tf.layers.dense(aligned_fea, d, name='key')
            V = tf.layers.dense(aligned_fea, d, name='value')
            attn = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
            attn = tf.nn.softmax(attn)
            attn_out = tf.matmul(attn, V)
            input_layer = tf.reduce_mean(attn_out, axis=1)           # b, d
            for i in range(len(aiau_deep_layers2)):
                deep_layer = tf.layers.dense(input_layer, int(aiau_deep_layers2[i]), activation=activation,
                                          partitioner=self.get_partitioner('min_max'), name ='mlp2_%d' % (i))
                input_layer = deep_layer
            aiau_output = tf.layers.dense(input_layer, d, activation=None, name='mlp2_out')
            return aiau_output

    def inference(self, input_layer, name_scope, is_training):
        with tf.name_scope("tower_mlp"):
            deep_layers = map(int, self.config['deep_layers'].split('_'))

            for i in range(len(deep_layers)):
                net = tf.layers.dense(inputs=input_layer, units=deep_layers[i],
                                      activation=None,
                                      kernel_initializer=self.kernel_initializer,
                                      bias_initializer=self.bias_initializer,
                                      partitioner=self.get_partitioner('fix'),
                                      name='%s_fc_%d' % (name_scope, i))

                input_layer = tf.layers.batch_normalization(inputs=net, training=is_training,
                                                            name='bn_%d' % (i))

                if self.config['activation'] == "dice":
                    input_layer = self.dice(input_layer, name='dice_%d' % i, is_training=is_training)
                elif self.config['activation'] == "prelu":
                    input_layer = self.parametric_relu(input_layer, "fcn_layer_prelu_%d" % i)
                else:
                    input_layer = tf.nn.relu(input_layer)

            # 最后一层为输出层，w*x+b 得到logits, 所以不需要指定activation
            # 不指定activation时即使用线性激活函数
            net = tf.layers.dense(inputs=input_layer, units=int(self.config['label_dim']),
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  name='%s_output' % (name_scope))

            net = tf.layers.batch_normalization(inputs=net, training=is_training,
                                                name='bn_%d' % (len(deep_layers)))
        return net

    def apply_loss(self, y_logits, y_true):
        with tf.name_scope('loss'):
            cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                                         logits=y_logits,
                                                                         name='xentropy')
            loss = tf.reduce_mean(cross_entropy_loss, name='xentropy_mean')
        return loss

    def apply_optimizer(self, loss, learning_rate=0.001, lazyAdam_learning_rate=0.0003, optimizer='adam'):
        with tf.name_scope('optimizer'):
            if optimizer == 'adagrad_lazyadam':
                hash_opt = HashAdagradOptimizer(learning_rate)
                normal_opt = tf.contrib.opt.LazyAdamOptimizer(lazyAdam_learning_rate)
            elif optimizer == 'adam_lazyadam':
                hash_opt = HashAdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8,
                                             use_locking=False, name="HashAdam", shared_beta_power=False,
                                             use_parallel=True)
                normal_opt = tf.contrib.opt.LazyAdamOptimizer(lazyAdam_learning_rate)
            else:
                hash_opt = HashAdagradOptimizer(learning_rate)
                normal_opt = tf.contrib.opt.LazyAdamOptimizer(lazyAdam_learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                var_list_not_in_hashtable = []
                for v in tf.trainable_variables():
                    if not v.name.startswith('emb_hashtable'):
                        var_list_not_in_hashtable.append(v)

                grads = hash_opt.compute_gradients(loss, var_list=tf.get_collection(stats.BIGMODEL_EMBED_HASH_TABLE))
                normal_grads = normal_opt.compute_gradients(loss, var_list=var_list_not_in_hashtable)
                hash_train_op = hash_opt.apply_gradients(grads)
                normal_train_op = normal_opt.apply_gradients(normal_grads, global_step=tf.train.get_global_step())

                train_op = tf.group(hash_train_op, normal_train_op)

                return train_op


    def din_layer(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col, seq_len_fea_col,
                        embed_dim, seq_fea_num, seq_len, din_deep_layers, din_activation, name_scope,
                        att_type):
        with tf.name_scope("attention_layer_%s" % (att_type)):
            cur_poi_emb_rep = tf.tile(cur_poi_seq_fea_col, [1, seq_len, 1])
            # 将query复制 seq_len 次 None, seq_len, embed_dim
            din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)

            activation = tf.nn.relu if din_activation == "relu" else tf.nn.tanh
            input_layer = din_all

            for i in range(len(din_deep_layers)):
                deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=activation,
                                             partitioner=self.get_partitioner('min_max'),
                                             name=name_scope + 'f_%d_att' % (i))
                input_layer = deep_layer

            din_output_layer = tf.layers.dense(input_layer, 1, activation=None, name=name_scope + 'fout_att')
            din_output_layer = tf.reshape(din_output_layer, [-1, 1, seq_len])

            # Mask
            key_masks = tf.sequence_mask(seq_len_fea_col, seq_len)

            paddings = tf.zeros_like(din_output_layer)
            outputs = tf.where(key_masks, din_output_layer, paddings)

            weighted_outputs = tf.matmul(outputs, hist_poi_seq_fea_col)

            weighted_outputs = tf.reshape(weighted_outputs, [-1, embed_dim * seq_fea_num])
            return weighted_outputs

    # 改成target, hist_sequence
    def can_layer(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col, seq_len_fea_col,
                  can_input_dim, seq_len, can_mlp_layer, din_activation, can_weight_dim, can_bias_dim, is_training, name
                  ):
        cur_poi_emb_rep = tf.tile(cur_poi_seq_fea_col, [1, seq_len, 1])

        cur_poi_emb_rep = tf.reshape(cur_poi_emb_rep, [-1, seq_len, can_mlp_layer, can_weight_dim + can_bias_dim])
        hist_poi_seq_fea_col = tf.reshape(hist_poi_seq_fea_col, [-1, seq_len, 1, can_input_dim])

        cur_poi_weight_rep = tf.slice(cur_poi_emb_rep, [0, 0, 0, 0], [-1, -1, -1, can_weight_dim])
        cur_poi_bias_rep = tf.slice(cur_poi_emb_rep, [0, 0, 0, can_weight_dim], [-1, -1, -1, can_bias_dim])
        cur_poi_weight_rep = tf.reshape(cur_poi_weight_rep, [-1, seq_len, can_mlp_layer, can_input_dim, can_input_dim])
        cur_poi_weight_list = tf.split(cur_poi_weight_rep, can_mlp_layer, axis=2)
        cur_poi_bias_list = tf.split(cur_poi_bias_rep, can_mlp_layer, axis=2)

        input = hist_poi_seq_fea_col
        for layer in range(can_mlp_layer):
            cur_layer_cur_poi_weight = tf.squeeze(cur_poi_weight_list[layer], axis=[2])
                             cur_layer_cur_poi_weight)
            cur_output = tf.matmul(input, cur_layer_cur_poi_weight)
            cur_output = tf.math.add(cur_output, cur_poi_bias_list[layer])
            cur_output = tf.layers.batch_normalization(inputs=cur_output, training=is_training,
                                                        name='can%s_bn_%d' % (name, layer))
            cur_output = tf.nn.tanh(cur_output)
            input = cur_output
        cur_output = tf.reshape(cur_output, [-1, seq_len, can_input_dim])
        sum_output = tf.reduce_sum(cur_output, axis=1)

        dft_seq_len_array = tf.ones_like(seq_len_fea_col) * seq_len
        no_zero_seq_len_fea_col = tf.where(tf.equal(seq_len_fea_col, 0), dft_seq_len_array, seq_len_fea_col)
        no_zero_seq_len_fea_col = tf.cast(no_zero_seq_len_fea_col, tf.float32)
        avg_pooling_output = tf.divide(sum_output, no_zero_seq_len_fea_col)
        return avg_pooling_output

    def dice(self, _x, axis=-1, epsilon=0.000000001, name='', is_training=True):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
        pop_mean = tf.get_variable(name="pop_mean" + name, shape=[1, _x.get_shape().as_list()[-1]],
                                   initializer=tf.constant_initializer(0.0),
                                   trainable=False)
        pop_std = tf.get_variable(name="pop_std" + name, shape=[1, _x.get_shape().as_list()[-1]],
                                  initializer=tf.constant_initializer(1.0),
                                  trainable=False)

        reduction_axes = 0
        broadcast_shape = [1, _x.shape.as_list()[-1]]
        decay = 0.999
        if is_training:
            mean = tf.reduce_mean(_x, axis=reduction_axes)
            brodcast_mean = tf.reshape(mean, broadcast_shape)
            std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
            std = tf.sqrt(std)
            brodcast_std = tf.reshape(std, broadcast_shape)
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + brodcast_mean * (1 - decay))
            train_std = tf.assign(pop_std,
                                  pop_std * decay + brodcast_std * (1 - decay))
            with tf.control_dependencies([train_mean, train_std]):
                x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
        else:
            x_normed = (_x - pop_mean) / (pop_std + epsilon)
        x_p = tf.sigmoid(x_normed)
        return alphas * (1.0 - x_p) * _x + x_p * _x

    def parametric_relu(self, _x, name):
        alphas = tf.get_variable('alpha' + name, [_x.get_shape()[-1]],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg
