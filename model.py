import tensorflow as tf
import tensorflow_probability as tfp


class AWGIM:
    def __init__(self, args, keep_prob, is_training):
        self.dim_latent = args.dim_latent
        self._l2_penalty_weight = args.weight_decay
        self._float_dtype = tf.float32
        self.initializer = tf.initializers.glorot_uniform(dtype=self._float_dtype)
        self.keep_prob = keep_prob
        self.is_training = is_training
        self.num_shot = args.num_shot
        self.num_class = args.num_way
        self.num_query = args.num_query
        self.embedding_dim = 640
        self.random_sample = self.sample
        self.mlp_size = [2*self.dim_latent] * (args.mlp_size-1)
        self.b_size = args.batch_size
        self.shuffle = args.shuffle

    def sample(self, distribution_params, stddev_offset=0.):
        means, unnormalized_stddev = tf.split(distribution_params, 2, axis=-1)
        stddev = tf.exp(unnormalized_stddev)
        stddev -= (1. - stddev_offset)
        stddev = tf.maximum(stddev, 1e-10)
        distribution = tfp.distributions.Normal(loc=means, scale=stddev)
        samples = distribution.sample()
        return tf.cond(self.is_training, false_fn=lambda: (means, means, stddev),
                       true_fn=lambda: (samples, means, stddev))

    def support_loss(self, data, label, cls_weights):
        cls_weights = tf.reshape(cls_weights, (self.b_size, self.num_class*self.num_query, self.num_class, self.num_shot, self.embedding_dim))
        cls_weights = tf.reduce_mean(cls_weights, 3)
        data = tf.reshape(data, (self.b_size, 1, self.num_class*self.num_shot, self.embedding_dim))
        data = tf.tile(data, (1, self.num_query*self.num_class, 1, 1))
        after_dropout = tf.nn.dropout(data, keep_prob=self.keep_prob)
        logits = tf.einsum('bqsp,bqcp->bqsc', after_dropout, cls_weights)

        logits = tf.reshape(logits, (self.b_size, self.num_class*self.num_query, self.num_class*self.num_shot, self.num_class))
        label = tf.tile(tf.reshape(label, (self.b_size, 1, self.num_class*self.num_shot)), (1, self.num_class*self.num_query, 1))
        one_hot_label = tf.one_hot(label, self.num_class)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(one_hot_label), logits=logits, dim=-1))
        pred = tf.nn.softmax(logits)
        accuracy = tf.contrib.metrics.accuracy(tf.argmax(pred, -1, output_type=tf.int32), label)
        entropy = tf.reduce_mean(tf.reduce_sum(-1. * tf.multiply(pred, tf.log(pred + 1e-6)), -1))
        return loss, accuracy, entropy, pred

    def query_loss(self, data, label, cls_weights):
        data = tf.reshape(data, (self.b_size, self.num_class*self.num_query, 1, self.embedding_dim))
        cls_weights = tf.reshape(cls_weights, (self.b_size, self.num_class*self.num_query, self.num_class, self.num_shot, self.embedding_dim))
        cls_weights = tf.reduce_mean(cls_weights, 3)
        if self.shuffle != 0:
            cls_weights = tf.reshape(cls_weights, (self.b_size, self.num_class, self.num_query, self.num_class, self.embedding_dim))
            cls_weights = tf.transpose(cls_weights, (1, 0, 2, 3, 4) if self.shuffle==1 else (2, 1, 0, 3, 4))
            cls_weights = tf.random_shuffle(cls_weights)
            cls_weights = tf.transpose(cls_weights, (1, 0, 2, 3, 4) if self.shuffle==1 else (2, 1, 0, 3, 4))
            cls_weights = tf.reshape(cls_weights, (self.b_size, self.num_class*self.num_query, self.num_class, self.embedding_dim))

        after_dropout = tf.nn.dropout(data, keep_prob=self.keep_prob)
        logits = tf.einsum('bqip,bqcp->bqic', after_dropout, cls_weights)
        logits = tf.squeeze(logits)
        logits = tf.reshape(logits, (self.b_size, self.num_class, self.num_query, self.num_class))
        loss = self.loss_fn(logits, label)
        pred = tf.nn.softmax(logits)
        accuracy = tf.contrib.metrics.accuracy(tf.argmax(pred, -1, output_type=tf.int32), label)
        entropy = tf.reduce_mean(tf.reduce_sum(-1. * tf.multiply(pred, tf.log(pred + 1e-6)), -1))
        return loss, accuracy, entropy, pred

    def loss_fn(self, logits, label):
        one_hot_label = tf.one_hot(label, self.num_class)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(one_hot_label), logits=logits, dim=-1))

    def dot_product_attention(self, q, k, v, normalise):
        d_k = tf.shape(q)[-1]
        scale = tf.sqrt(tf.cast(d_k, tf.float32))
        unnorm_weights = tf.einsum('bjk,bik->bij', k, q) / scale
        if normalise:
            weight_fn = tf.nn.softmax
        else:
            weight_fn = tf.sigmoid
        weights = weight_fn(unnorm_weights)
        rep = tf.einsum('bik,bkj->bij', weights, v)
        return rep

    def mlp(self, input, output_sizes, name):
        output = tf.nn.dropout(input, self.keep_prob)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for i, size in enumerate(output_sizes[:-1]):
                output = tf.nn.relu(tf.layers.dense(output, size, name="layer_{}".format(i), use_bias=False))
            # Last layer without a ReLu
            output = tf.layers.dense(output, output_sizes[-1], name="layer_out", use_bias=False)
        return output

    def multihead_attention(self, q, k, v, name, num_heads=4):
        d_k = q.get_shape().as_list()[-1]
        d_v = v.get_shape().as_list()[-1]
        head_size = d_v / num_heads
        key_initializer = tf.random_normal_initializer(stddev=d_k ** -0.5)
        value_initializer = tf.random_normal_initializer(stddev=d_v ** -0.5)
        rep = tf.constant(0.0)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for h in range(num_heads):
                o = self.dot_product_attention(
                    tf.layers.dense(q, head_size, kernel_initializer=key_initializer, use_bias=False, name='wq%d' % h),
                    tf.layers.dense(k, head_size, kernel_initializer=key_initializer, use_bias=False, name='wk%d' % h),
                    tf.layers.dense(v, head_size, kernel_initializer=key_initializer, use_bias=False, name='wv%d' % h),
                    normalise=True)
                rep += tf.layers.dense(o, d_v, kernel_initializer=value_initializer, use_bias=False, name='wo%d' % h)
            rep += q
            rep = tf.contrib.layers.layer_norm(rep, 2)
            rep += self.mlp(rep, [2*d_v, d_v], name+'ln_mlp')
            rep = tf.contrib.layers.layer_norm(rep, 2)
        return rep

    def forward(self, tr_data, tr_label, val_data, val_label):
        # tr_data is b x c x k x p, tr_lable is b x c x k, val_data is b x c x q x p, val_label is b x c x q
        fan_in = tf.cast(self.embedding_dim, self._float_dtype)
        fan_out = tf.cast(self.num_class, self._float_dtype)
        stddev_offset = tf.sqrt(2. / (fan_out + fan_in))

        # attentive path
        support = self.mlp(tr_data, [self.dim_latent], 'CA_encoder')
        query = self.mlp(val_data, [self.dim_latent], 'CA_encoder')
        key_support = tf.reshape(support, (self.b_size, self.num_class*self.num_shot, self.dim_latent))
        query_query = tf.reshape(query, (self.b_size, self.num_class*self.num_query, self.dim_latent))
        value_context = tf.reshape(support, (self.b_size, self.num_class * self.num_shot, self.dim_latent))
        value_context = self.multihead_attention(value_context, value_context, value_context, 'context_CA')
        value_context = tf.reshape(value_context, (self.b_size, self.num_class, self.num_shot, self.dim_latent))
        value_context = tf.reduce_mean(value_context, 2, True)
        value_context = tf.tile(value_context, (1, 1, self.num_shot, 1))
        value_context = tf.reshape(value_context, (self.b_size, self.num_class*self.num_shot, self.dim_latent))

        query_ca = self.multihead_attention(query_query, key_support, value_context, 'query_CA')
        query_ca_code = tf.tile(tf.reshape(query_ca, (self.b_size, self.num_class * self.num_query, 1, self.dim_latent)),
                                (1, 1, self.num_shot * self.num_class, 1))

        # contextual path
        support = self.mlp(tr_data, [self.dim_latent], 'SA_encoder')
        context = tf.reshape(support, (self.b_size, self.num_class * self.num_shot, self.dim_latent))
        context = self.multihead_attention(context, context, context, 'context_SA')
        context_code = tf.reshape(context, (self.b_size, self.num_class, self.num_shot, self.dim_latent))
        context_code = tf.reduce_mean(context_code, 2, True)
        context_code = tf.tile(context_code, (1, 1, self.num_shot, 1))
        context_code = tf.tile(tf.reshape(context_code, (self.b_size, 1, self.num_class * self.num_shot, self.dim_latent)),
                               (1, self.num_class * self.num_query, 1, 1))
        concat = tf.concat((context_code, query_ca_code), 3)
        # concat is b x cq x ck x d
        decoder_name = 'decoder'
        reconstruct_mlp = self.mlp_size
        weights_dist_params = self.mlp(concat, self.mlp_size + [2*self.embedding_dim], decoder_name+'mlp_weight')
        classifier_weights, mu, sigma = self.random_sample(weights_dist_params, stddev_offset=stddev_offset)
        reconstructed_query = self.mlp(classifier_weights, reconstruct_mlp + [self.dim_latent],
                                       'recontructed_q')
        reconstructed_support = self.mlp(classifier_weights, reconstruct_mlp + [self.dim_latent],
                                         'recontructed_s')

        tr_loss, tr_accuracy, entropy_support, pred_support = self.support_loss(tr_data, tr_label, classifier_weights)
        val_loss, val_accuracy, entropy_query, pred_query = self.query_loss(val_data, val_label, classifier_weights)

        loss_support = tf.reduce_mean(tf.reduce_sum(tf.square(tf.stop_gradient(context_code) - reconstructed_support), -1))
        loss_query = tf.reduce_mean(tf.reduce_sum(tf.square(tf.stop_gradient(query_ca_code) - reconstructed_query), -1))
        return val_loss, val_accuracy, tr_loss, tr_accuracy, loss_support, loss_query

