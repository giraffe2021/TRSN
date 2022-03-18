from tools.augmentations import *
from functools import partial
import os
import datetime
import random
import cv2
import numpy as np, scipy.stats as st
from datasets.mini_imagenet_dataset_v2 import MiniImageNetDataLoader as MiniImageNetDataLoader_v2
from datasets.dataloader import DataLoader
from backbone.backbone import Backbone


class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                            self.learning_rate_base - self.warmup_learning_rate
                    ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


class WarmUpStep(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self, learning_rate_base, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpStep, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):

        learning_rate = self.learning_rate_base

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                            self.learning_rate_base - self.warmup_learning_rate
                    ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return learning_rate


WEIGHT_DECAY = 0.0005

model_name = os.path.basename(__file__).split(".")[0]


class FSLModel(tf.keras.Model):
    def __init__(self, imageshape=(84, 84, 3), num_class=64, name=model_name):
        super(FSLModel, self).__init__(name=name)

        self.num_class = num_class
        self.encoder = Backbone("resnet_12", input_shape=imageshape, pooling=None, use_bias=False).get_model()
        self.encoder.build([None, *imageshape])
        index = 0
        feature_size_h, feature_size_w, feature_dim = [*self.encoder.output.shape[1:]]

        self.attention = Sequential([layers.Conv2D(128,
                                                   kernel_size=(3, 3),
                                                   strides=(1, 1),
                                                   padding="same",
                                                   use_bias=False,
                                                   kernel_initializer='he_normal',
                                                   kernel_regularizer=tf.keras.regularizers.l2(
                                                       WEIGHT_DECAY),
                                                   name="attention_{}".format(index)),
                                     layers.BatchNormalization()])
        self.attention.build([None, feature_size_h, feature_size_w, feature_dim])

        self.self_attention_referenced_conv = Sequential([layers.Conv2D(64,
                                                                        kernel_size=(3, 3),
                                                                        strides=(1, 1),
                                                                        padding="same",
                                                                        kernel_initializer='he_normal',
                                                                        kernel_regularizer=tf.keras.regularizers.l2(
                                                                            WEIGHT_DECAY),
                                                                        name="self_attention_referenced_conv_{}".format(
                                                                            0)),
                                                          layers.BatchNormalization(),
                                                          layers.LeakyReLU(alpha=0.2),
                                                          layers.Conv2D(1,
                                                                        kernel_size=(1, 1),
                                                                        strides=(1, 1),
                                                                        padding="same",
                                                                        kernel_initializer='he_normal',
                                                                        kernel_regularizer=tf.keras.regularizers.l2(
                                                                            WEIGHT_DECAY),
                                                                        name="self_attention_referenced_conv_{}".format(
                                                                            1)),
                                                          layers.BatchNormalization(),
                                                          layers.LeakyReLU(alpha=0.2),
                                                          ],
                                                         name="self_attention_referenced_conv")
        # self.self_attention_referenced_conv.build([None, feature_size_h, feature_size_w, feature_dim * 2])
        self.self_attention_referenced_conv.build([None, feature_size_h, feature_size_w, feature_dim])

        self.class_special_pos = Sequential([layers.Conv2D(64,
                                                           kernel_size=(1, 1),
                                                           strides=(1, 1),
                                                           padding="same",
                                                           kernel_initializer='he_normal',
                                                           kernel_regularizer=tf.keras.regularizers.l2(
                                                               WEIGHT_DECAY),
                                                           name="class_special_pos_{}".format(
                                                               0)),
                                             layers.BatchNormalization(),
                                             layers.LeakyReLU(alpha=0.2),
                                             layers.Conv2D(self.num_class,
                                                           kernel_size=(1, 1),
                                                           strides=(1, 1),
                                                           padding="same",
                                                           kernel_initializer='he_normal',
                                                           kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                                           name="pix_conv_pos_{}".format(index)),
                                             layers.BatchNormalization(),
                                             layers.LeakyReLU(alpha=0.2)], name="class_special_positive")
        self.class_special_pos.build([None, feature_size_h, feature_size_w, feature_dim])
        self.class_special_neg_local = Sequential([layers.Conv2D(1,
                                                                 kernel_size=(1, 1),
                                                                 padding="same",
                                                                 kernel_regularizer=tf.keras.regularizers.l2(
                                                                     WEIGHT_DECAY),
                                                                 name="pix_conv_neg_{}".format(index)),
                                                   layers.BatchNormalization(),
                                                   layers.LeakyReLU(alpha=0.2)], name="class_special_negative_local")
        self.class_special_neg_mid = Sequential([layers.Conv2D(1,
                                                               kernel_size=(3, 3),
                                                               dilation_rate=(2, 2),
                                                               padding="same",
                                                               kernel_regularizer=tf.keras.regularizers.l2(
                                                                   WEIGHT_DECAY),
                                                               name="pix_conv_neg_{}".format(index)),
                                                 layers.BatchNormalization(),
                                                 layers.LeakyReLU(alpha=0.2)], name="class_special_negative_mid")
        self.class_special_neg_global = Sequential([layers.Conv2D(1,
                                                                  kernel_size=(3, 3),
                                                                  dilation_rate=(3, 3),
                                                                  padding="same",
                                                                  kernel_regularizer=tf.keras.regularizers.l2(
                                                                      WEIGHT_DECAY),
                                                                  name="pix_conv_neg_{}".format(index)),
                                                    layers.BatchNormalization(),
                                                    layers.LeakyReLU(alpha=0.2)], name="class_special_negative_global")
        self.class_special_neg_local.build([None, feature_size_h, feature_size_w, feature_dim])
        self.class_special_neg_mid.build([None, feature_size_h, feature_size_w, feature_dim])
        self.class_special_neg_global.build([None, feature_size_h, feature_size_w, feature_dim])

        self.last_max_pooling = tf.keras.layers.MaxPool2D(padding="same", name="last_max_pooling")
        self.last_max_pooling.build([None, feature_size_h, feature_size_w, feature_dim])
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="gap")
        self.gap.build([None, feature_size_h, feature_size_w, feature_dim])
        self.bn = tf.keras.layers.BatchNormalization()
        self.bn.build([None, self.num_class])

        self.clc = tf.keras.layers.Dense(self.num_class,
                                         kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                         name="clc")
        self.clc.build([None, feature_dim])

        self.mrn = Sequential([
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
            layers.Dense(1)
        ], name="mrn")
        self.mrn.build([None, feature_dim * 2])
        self.mrn.summary()

        self.build([None, *imageshape])
        self.summary()
        self.acc = tf.keras.metrics.CategoricalAccuracy(name="acc")
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.salient_loss_metric = tf.keras.metrics.Mean(name="salient_loss")

        self.query_loss_metric = tf.keras.metrics.Mean("query_loss")
        self.mean_query_acc = tf.keras.metrics.Mean(name="mean_query_acc")
        self.mean_query_acc_attention = tf.keras.metrics.Mean(name="mean_query_acc_attention")
        self.query_loss_metric_attenion = tf.keras.metrics.Mean("query_loss_metric_attenion")

        self.mean_query_acc_base = tf.keras.metrics.Mean(name="mean_query_acc_base")
        self.query_loss_metric_base = tf.keras.metrics.Mean("query_loss_metric_base")

    @tf.function
    def get_max(self, x):
        batch, h, w, c = tf.unstack(tf.shape(x))
        batch_index = tf.reshape(tf.range(batch), [batch, 1, 1])
        batch_index = tf.broadcast_to(batch_index, [batch, c, 1])

        x = tf.reshape(x, [batch, h * w, c])
        indices_max = tf.argmax(x, axis=1, output_type=tf.int32)

        indices_max = tf.stack([indices_max // w, indices_max % w], -1)
        indices_max = tf.concat([batch_index, indices_max], -1)

        indices_min = tf.argmin(x, axis=1, output_type=tf.int32)
        indices_min = tf.stack([indices_min // w, indices_min % w], -1)
        indices_min = tf.concat([batch_index, indices_min], -1)

        return indices_max, indices_min

    def call(self, inputs, training=None):
        return None

    def get_local_feature(self, features, training=None):
        _, f_h, f_w, f_c = tf.unstack(tf.shape(features))
        pred_special_pos_logits = self.class_special_pos(features, training=training)
        pred_special_neg_logits_local = self.class_special_neg_local(features, training=training)

        pred_special_neg_logits_mid = self.class_special_neg_mid(features, training=training)
        pred_special_neg_logits_mid = tf.image.resize(pred_special_neg_logits_mid, (f_h, f_w))
        pred_special_neg_logits_global = self.class_special_neg_global(features, training=training)
        pred_special_neg_logits_global = tf.image.resize(pred_special_neg_logits_global, (f_h, f_w))
        pred_special_neg_logits = pred_special_neg_logits_local + pred_special_neg_logits_mid + pred_special_neg_logits_global
        pred_special_neg_logits = tf.broadcast_to(pred_special_neg_logits, tf.shape(pred_special_pos_logits))
        pred_special = tf.stack([pred_special_pos_logits, pred_special_neg_logits], -1)
        pred_special_softmax = tf.nn.softmax(pred_special / 10., -1)
        return pred_special, pred_special_softmax

    def calculate_entropy(self, support_mean_base, query_logits, temper=1., training=None):
        batch = tf.shape(query_logits)[0]
        query_ways = tf.shape(query_logits)[1]
        query_shots = tf.shape(query_logits)[2]
        logits_dim = tf.shape(query_logits)[-1]
        support_ways = tf.shape(support_mean_base)[1]

        query_logits = tf.nn.l2_normalize(query_logits, -1)
        query_logits = tf.reshape(query_logits, [batch, query_ways * query_shots, 1, logits_dim])
        support_mean_base = tf.reshape(support_mean_base, [batch, 1, support_ways, logits_dim])
        query_logits = tf.broadcast_to(query_logits, [batch, query_ways * query_shots, support_ways, logits_dim])
        support_mean_base = tf.broadcast_to(support_mean_base,
                                            [batch, query_ways * query_shots, support_ways, logits_dim])
        logits_pairs = tf.concat([query_logits, support_mean_base], -1)
        logits_pairs = tf.reshape(logits_pairs, [-1, logits_dim * 2])

        sim = self.mrn(logits_pairs, training)
        sim = tf.reshape(sim, [batch, query_ways, query_shots, support_ways])
        sim_softmax = tf.nn.softmax(sim * temper, -1)
        entropy = -1. * sim_softmax * tf.math.log(sim_softmax + tf.keras.backend.epsilon())
        entropy = tf.reduce_sum(entropy, -1)

        return sim_softmax, entropy

    def meta_train_step(self, support, query, training=None):
        support_image, support_label, support_global_label = support
        query_image, query_label, query_global_label = query

        image_shape = tf.unstack(tf.shape(support_image)[-3:])
        dim_shape = tf.shape(support_label)[-1]
        global_dim_shape = tf.shape(support_global_label)[-1]

        batch = tf.shape(support_image)[0]
        ways = tf.shape(support_image)[1]
        shots = tf.shape(support_image)[2]
        query_shots = tf.shape(query_image)[2]

        support_x = tf.reshape(support_image, [-1, *image_shape])
        query_x = tf.reshape(query_image, [-1, *image_shape])

        support_global_label = tf.reshape(support_global_label, [-1, global_dim_shape])
        query_global_label = tf.reshape(query_global_label, [-1, global_dim_shape])

        features_support = self.encoder(support_x, training=training)
        attention_features_support = self.attention(features_support, training=training)
        attention_features_l2_norm_support = tf.nn.l2_normalize(attention_features_support, axis=-1)
        _, f_h, f_w, f_c = tf.unstack(tf.shape(attention_features_l2_norm_support))
        reshape_support_features = tf.reshape(attention_features_l2_norm_support,
                                              [batch * ways, shots, f_h * f_w, f_c])

        features_query = self.encoder(query_x, training=training)
        attention_features_query = self.attention(features_query, training=training)
        attention_features_l2_norm_query = tf.nn.l2_normalize(attention_features_query, axis=-1)

        reshape_query_features = tf.reshape(attention_features_l2_norm_query,
                                            [batch * ways, 1, query_shots * f_h * f_w, f_c])
        similarity_map_referenced = tf.linalg.matmul(reshape_support_features, reshape_query_features, transpose_b=True)
        similarity_map_referenced_core = tf.clip_by_value(similarity_map_referenced, 0., 1)
        similarity_map_referenced_core = tf.reshape(similarity_map_referenced_core,
                                                    [batch * ways * shots, f_h, f_w, query_shots * f_h, f_w])
        reshape_query_features = tf.reshape(reshape_query_features,
                                            [batch * ways, query_shots, f_h * f_w, f_c])

        _, f_h, f_w, f_c = tf.unstack(tf.shape(features_support))

        pred_local_support, pred_local_softmax_support = self.get_local_feature(features_support, training=training)
        pred_local_query, pred_local_softmax_query = self.get_local_feature(features_query, training=training)

        indices_max_support, indices_min_support = self.get_max(pred_local_softmax_support[..., 0])
        logits_weights_referenced = tf.gather_nd(similarity_map_referenced_core, indices_max_support)
        indices_max_query, indices_min_query = self.get_max(pred_local_softmax_query[..., 0])

        logits_weights_query_referenced = tf.reshape(tf.transpose(logits_weights_referenced, [0, 2, 3, 1]),
                                                     [batch * ways, shots, query_shots, f_h, f_w, global_dim_shape])
        query_global_label_referenced = tf.reshape(
            tf.repeat(tf.reshape(query_global_label, [batch, ways, 1, query_shots, global_dim_shape]), shots, 2),
            [batch * ways, shots, query_shots, global_dim_shape])
        logits_weights_query_referenced = tf.reduce_sum(tf.reduce_mean(tf.reshape(query_global_label_referenced,
                                                                                  [batch * ways, shots, query_shots, 1,
                                                                                   1,
                                                                                   global_dim_shape]) * logits_weights_query_referenced,
                                                                       1), -1)
        logits_weights_query_referenced = tf.reshape(logits_weights_query_referenced,
                                                     [batch * ways * query_shots, f_h, f_w, 1])
        logits_weights_query_referenced_stop_gradient = tf.stop_gradient(logits_weights_query_referenced)

        logits_weights_referenced = tf.transpose(logits_weights_referenced, [0, 2, 3, 1])
        logits_weights_referenced = tf.reshape(logits_weights_referenced,
                                               [batch * ways, shots, query_shots * f_h, f_w, self.num_class])

        local_logits_query = pred_local_query[..., 0]
        local_logits_query = tf.reshape(local_logits_query, [batch * ways, 1, query_shots * f_h, f_w, self.num_class])
        weighted_local_logits_query = logits_weights_referenced * local_logits_query
        weighted_local_logits_query = tf.math.divide_no_nan(tf.reduce_sum(weighted_local_logits_query, [-3, -2]),
                                                            tf.reduce_sum(logits_weights_referenced, [-3, -2]))
        weighted_local_logits_query_pred = tf.reshape(weighted_local_logits_query,
                                                      [batch * ways * shots, self.num_class])
        weighted_local_logits_query_pred = self.bn(weighted_local_logits_query_pred, training=training)

        similarity_map_support_self = tf.linalg.matmul(reshape_support_features, reshape_support_features,
                                                       transpose_b=True)
        similarity_map_support_self_score = tf.clip_by_value(similarity_map_support_self, 0., 1)
        similarity_map_support_self_score = tf.reshape(similarity_map_support_self_score,
                                                       [batch * ways * shots, f_h, f_w, f_h, f_w])
        logits_weights_support_self = tf.gather_nd(similarity_map_support_self_score, indices_max_support)
        logits_weights_support = tf.transpose(logits_weights_support_self, [0, 2, 3, 1])

        logits_weights_support_self = tf.reduce_sum(
            tf.reshape(support_global_label,
                       [*tf.unstack(tf.shape(support_global_label)), 1, 1]) * logits_weights_support_self,
            1)
        logits_weights_support_self = tf.expand_dims(logits_weights_support_self, -1)
        logits_weights_support_self_stop_gradient = tf.stop_gradient(logits_weights_support_self)
        local_logits_support = pred_local_support[..., 0]
        weighted_local_logits_support = logits_weights_support * local_logits_support
        weighted_local_logits_support = tf.math.divide_no_nan(tf.reduce_sum(weighted_local_logits_support, [-3, -2]),
                                                              tf.reduce_sum(logits_weights_support, [-3, -2]))
        weighted_local_logits_support_pred = tf.reshape(weighted_local_logits_support,
                                                        [batch * ways * shots, self.num_class])
        weighted_local_logits_support_pred = self.bn(weighted_local_logits_support_pred, training=training)

        similarity_map_query_self = tf.linalg.matmul(reshape_query_features, reshape_query_features, transpose_b=True)
        similarity_map_query_self_core = tf.clip_by_value(similarity_map_query_self, 0., 1)

        similarity_map_query_self_core = tf.reshape(similarity_map_query_self_core,
                                                    [batch * ways * query_shots, f_h, f_w, f_h, f_w])

        logits_weights_query_self = tf.gather_nd(similarity_map_query_self_core, indices_max_query)
        logits_weights_query_self = tf.reduce_sum(
            tf.reshape(query_global_label,
                       [*tf.unstack(tf.shape(query_global_label)), 1, 1]) * logits_weights_query_self,
            1)
        logits_weights_query_self = tf.expand_dims(logits_weights_query_self, -1)
        logits_weights_query_self_stop_gradient = tf.stop_gradient(logits_weights_query_self)

        support_logits = self.gap(self.last_max_pooling(features_support))
        query_logits = self.gap(self.last_max_pooling(features_query))

        support_pred = self.clc(support_logits)
        query_pred = self.clc(query_logits)

        support_logits = tf.reshape(support_logits,
                                    [batch, ways, shots, tf.shape(support_logits)[-1]])
        support_logits_mean = tf.nn.l2_normalize(support_logits, -1)
        support_logits_mean = tf.reduce_mean(support_logits_mean, 2)
        support_logits_mean = tf.reshape(support_logits_mean, [batch, 1, ways, 1, 1, f_c])
        support_logits_broad = tf.broadcast_to(support_logits_mean, [batch, ways * query_shots, ways, f_h, f_w, f_c])
        features_query = tf.reshape(features_query, [batch, ways * query_shots, 1, f_h, f_w, f_c])
        features_query_broad = tf.broadcast_to(features_query, [batch, ways * query_shots, ways, f_h, f_w, f_c])
        # merge_feature_query = tf.concat([tf.nn.l2_normalize(features_query_broad, -1), support_logits_broad], -1)
        merge_feature_query = tf.nn.l2_normalize(features_query_broad, -1) * support_logits_broad
        merge_feature_query = tf.stop_gradient(merge_feature_query)
        merge_feature_query = tf.reshape(merge_feature_query, [-1, f_h, f_w, tf.shape(merge_feature_query)[-1]])
        query_enhanced_mask = self.self_attention_referenced_conv(merge_feature_query, training=training)
        query_enhanced_mask = tf.reshape(query_enhanced_mask, [-1, ways, f_h, f_w])
        query_enhanced_mask = tf.transpose(query_enhanced_mask, [0, 2, 3, 1])
        query_enhanced_mask = tf.nn.softmax(query_enhanced_mask / 10., -1)
        loss_clc = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                tf.concat([support_global_label, query_global_label], 0),
                tf.concat(
                    [support_pred, query_pred],
                    0),
                from_logits=True))

        loss_clc = 0.5 * loss_clc + 0.5 * tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                tf.concat([support_global_label, support_global_label], 0),
                tf.concat(
                    [weighted_local_logits_support_pred, weighted_local_logits_query_pred],
                    0),
                from_logits=True))
        loss_clc = tf.clip_by_value(loss_clc, 0., 10.)

        # pseudo_label = 0.5 * logits_weights_query_self_stop_gradient + 0.5 * logits_weights_query_referenced_stop_gradient
        pseudo_label = tf.minimum(logits_weights_query_self_stop_gradient,
                                  logits_weights_query_referenced_stop_gradient)
        query_label = tf.reshape(query_label, [-1, 1, 1, dim_shape])
        pseudo_label = query_label * pseudo_label + (1 - query_label) * (1 - pseudo_label) / tf.cast(dim_shape,
                                                                                                     tf.float32)
        salient_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(pseudo_label, query_enhanced_mask))

        neg_sample_indices_query = tf.where(tf.equal(0., query_global_label))
        neg_samples_query = tf.gather_nd(tf.transpose(pred_local_softmax_query, [0, 3, 1, 2, 4]),
                                         neg_sample_indices_query)
        neg_score_query = tf.broadcast_to([[0.5, 0.5]], shape=tf.shape(neg_samples_query))

        contrastive_neg_loss = 0.5 * tf.reduce_mean(tfa.losses.SigmoidFocalCrossEntropy()(
            y_true=neg_score_query, y_pred=neg_samples_query))

        neg_sample_indices_support = tf.where(tf.equal(0., support_global_label))
        neg_samples_support = tf.gather_nd(tf.transpose(pred_local_softmax_support, [0, 3, 1, 2, 4]),
                                           neg_sample_indices_support)
        neg_score_support = tf.broadcast_to([[0.5, 0.5]], shape=tf.shape(neg_samples_support))

        contrastive_neg_loss += 0.5 * tf.reduce_mean(tfa.losses.SigmoidFocalCrossEntropy()(
            y_true=neg_score_support, y_pred=neg_samples_support))
        total_loss = loss_clc + contrastive_neg_loss

        return total_loss, salient_loss

    def reset_metrics(self):
        # Resets the state of all the metrics in the model.
        for m in self.metrics:
            m.reset_states()

        self.acc.reset_states()
        self.loss_metric.reset_states()
        self.query_loss_metric.reset_states()
        self.mean_query_acc.reset_states()
        self.mean_query_acc_attention.reset_states()
        self.query_loss_metric_attenion.reset_states()
        self.mean_query_acc_base.reset_states()
        self.query_loss_metric_base.reset_states()
        self.salient_loss_metric.reset_states()
        #
        # self.offset_loss_metric.reset_states()
        # self.contras_loss_metric.reset_states()
        # self.acc_special.reset_states()
        # self.acc_common.reset_states()
        # self.loss_common_metric.reset_states()
        # self.loss_special_metric.reset_states()
        #
        # self.d_loss_metric.reset_states()
        # self.g_loss_metric.reset_states()

    def random_enrode(self, x):
        x = tf.reshape(x, tf.concat([[-1], tf.shape(x)[-3:]], 0))
        batch, h, w, c = tf.unstack(tf.shape(x))
        for _ in range(1):
            features = self.encoder(x, training=False)

            self_attention = self.self_attention_referenced_conv(features, training=False)
            self_attention = tf.image.resize(self_attention, [h, w])
            bg = x * (1. - self_attention)
            bg_mean = tf.reduce_mean(bg, [1, 2], keepdims=True)
            ratio = 0.
            bg_mean = tf.broadcast_to(bg_mean, tf.shape(x)) * self_attention
            keep_ratio = 0.005
            x = x * keep_ratio + (1. - keep_ratio) * tf.clip_by_value(
                x * (1. - self_attention) + x * ratio + bg_mean * (1 - ratio), 0., 1.)

        return x

    # @tf.function
    def train_step_normal(self, data):
        support, query = data
        with tf.GradientTape() as tape:
            total_loss, salient_loss = self.meta_train_step(support, query, True)
        trainable_vars = self.encoder.trainable_weights \
                         + self.attention.trainable_weights \
                         + self.class_special_pos.trainable_weights \
                         + self.class_special_neg_local.trainable_weights \
                         + self.class_special_neg_mid.trainable_weights \
                         + self.class_special_neg_global.trainable_weights \
                         + self.clc.trainable_weights \
                         + self.self_attention_referenced_conv.trainable_weights
        grads = tape.gradient([total_loss, salient_loss], trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.loss_metric.update_state(total_loss)
        self.salient_loss_metric.update_state(salient_loss)

        logs = {
            self.loss_metric.name: self.loss_metric.result(),
            self.salient_loss_metric.name: self.salient_loss_metric.result(),
        }

        return logs

    def random_sample_support(self, s, s_label):
        b, w, k, d = tf.unstack(tf.shape(s))
        begin_end = tf.random.shuffle(tf.range(k + 1))[:2]
        begin, end = tf.reduce_min(begin_end), tf.reduce_max(begin_end)
        # tf.print(begin, end)
        sub_s = s[..., begin:end, :]
        sub_label = s_label[..., begin:end, :]
        mean_s = tf.reduce_mean(sub_s, 2)
        mean_s_label = tf.reduce_mean(sub_label, 2)
        return mean_s, mean_s_label

    def train_step_meta(self, data):
        support, query = data
        support_image, support_label, _ = support
        query_image, query_label, _ = query

        batch = tf.shape(support_image)[0]
        ways = tf.shape(support_image)[1]
        shots = tf.shape(support_image)[2]
        query_shots = tf.shape(query_image)[2]

        support_image = tf.reshape(support_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))

        training = True

        with tf.GradientTape() as tape:
            support_features = self.encoder(support_image, training=training)
            _, f_h, f_w, f_c = tf.unstack(tf.shape(support_features))
            support_logits = self.gap(self.last_max_pooling(support_features))
            support_logits = tf.nn.l2_normalize(support_logits, -1)

            support_logits_merge = tf.reshape(support_logits,
                                              [batch, ways * shots, tf.shape(support_logits)[-1]])
            support_logits_merge = tf.nn.l2_normalize(support_logits_merge, -1)
            support_logits_merge = tf.reduce_mean(support_logits_merge, 1)
            support_logits_merge = tf.reshape(support_logits_merge, [batch, 1, 1, 1, f_c])
            support_logits_broad = tf.broadcast_to(support_logits_merge, [batch, ways * shots, f_h, f_w, f_c])
            support_logits_broad = tf.reshape(support_logits_broad, [-1, f_h, f_w, f_c])
            # merge_feature_support = tf.concat([tf.nn.l2_normalize(support_features, -1), support_logits_broad], -1)
            merge_feature_support = tf.nn.l2_normalize(support_features, -1) * support_logits_broad
            support_self_attention = self.self_attention_referenced_conv(merge_feature_support, training=training)

            support_features = tf.nn.l2_normalize(support_features, -1)
            support_logits_attention = support_features * support_self_attention
            support_logits_attention = tf.math.divide_no_nan(tf.reduce_sum(support_logits_attention, [1, 2]),
                                                             tf.reduce_sum(support_self_attention, [1, 2]))
            support_logits_attention = tf.nn.l2_normalize(support_logits_attention, -1)
            support_logits_fusion = tf.concat([support_logits, support_logits_attention], -1)

            support_logits_fusion = tf.reshape(support_logits_fusion,
                                               [batch, ways, shots, tf.shape(support_logits_fusion)[-1]])
            support_logits_fusion = tf.nn.l2_normalize(support_logits_fusion, -1)

            support_mean_fusion, support_mean_label = self.random_sample_support(support_logits_fusion, support_label)

            new_shape = tf.concat([[-1], tf.shape(query_image)[-3:]], axis=0)
            query_image = tf.reshape(query_image, new_shape)

            query_features = self.encoder(query_image, training=training)
            query_logits = self.gap(self.last_max_pooling(query_features))
            query_logits = tf.nn.l2_normalize(query_logits, -1)
            support_logits_broad = tf.broadcast_to(support_logits_merge, [batch, ways * query_shots, f_h, f_w, f_c])
            support_logits_broad = tf.reshape(support_logits_broad, [-1, f_h, f_w, f_c])
            # merge_feature_query = tf.concat([tf.nn.l2_normalize(query_features, -1), support_logits_broad], -1)
            merge_feature_query = tf.nn.l2_normalize(query_features, -1) * support_logits_broad
            query_self_attention = self.self_attention_referenced_conv(merge_feature_query, training=training)
            query_features = tf.nn.l2_normalize(query_features, -1)
            query_logits_attention = query_features * query_self_attention
            query_logits_attention = tf.math.divide_no_nan(tf.reduce_sum(query_logits_attention, [1, 2]),
                                                           tf.reduce_sum(query_self_attention, [1, 2]))
            query_logits_attention = tf.nn.l2_normalize(query_logits_attention, -1)
            query_logits_fusion = tf.concat([query_logits, query_logits_attention], -1)
            query_logits_fusion = tf.reshape(query_logits_fusion,
                                             [batch, ways, query_shots, tf.shape(query_logits_fusion)[-1]])
            logits_dim = tf.shape(support_logits_fusion)[-1]
            dim_shape = tf.shape(query_label)[-1]

            support_mean_fusion = tf.reshape(support_mean_fusion, [batch, ways, logits_dim])
            support_mean_fusion = tf.nn.l2_normalize(support_mean_fusion, -1)

            support_logits_fusion = tf.reshape(support_logits_fusion,
                                               [batch, -1, tf.shape(support_logits_fusion)[-1]])

            query_logits_fusion = tf.reshape(query_logits_fusion,
                                             [batch, -1, tf.shape(query_logits_fusion)[-1]])

            score_suport = tf.linalg.matmul(tf.nn.l2_normalize(support_mean_label, axis=-1),
                                            tf.nn.l2_normalize(tf.reshape(support_label, [batch, -1, dim_shape]),
                                                               axis=-1),
                                            transpose_b=True)
            score_query = tf.linalg.matmul(tf.nn.l2_normalize(support_mean_label, axis=-1),
                                           tf.nn.l2_normalize(tf.reshape(query_label, [batch, -1, dim_shape]), axis=-1),
                                           transpose_b=True)

            pos_index_support = tf.cast(tf.greater(score_suport, 0.7), tf.float32)
            pos_index_query = tf.cast(tf.greater(score_query, 0.7), tf.float32)
            neg_index_support = tf.cast(tf.less(score_suport, 0.5), tf.float32)
            neg_index_query = tf.cast(tf.less(score_query, 0.5), tf.float32)
            all_samples = tf.concat(
                [support_logits_fusion, query_logits_fusion],
                1)

            pos_mask = tf.concat(
                [pos_index_support, pos_index_query],
                2)
            neg_mask = tf.concat(
                [neg_index_support, neg_index_query],
                2)
            distances = 1. - tf.linalg.matmul(support_mean_fusion, all_samples, transpose_b=True)
            pos_distance = tf.math.divide_no_nan(
                tf.reduce_sum(distances * pos_mask, -1),
                tf.reduce_sum(pos_mask, -1))
            neg_distance = tf.math.divide_no_nan(
                tf.reduce_sum(distances * neg_mask, -1),
                tf.reduce_sum(neg_mask, -1))
            tripel_loss = tf.maximum(0., 1.0 + pos_distance - neg_distance)
            meta_contrast_loss = tf.reduce_mean(tripel_loss)
            avg_loss = meta_contrast_loss

        trainable_vars = self.encoder.trainable_weights \
                         + self.self_attention_referenced_conv.trainable_weights
        grads = tape.gradient([avg_loss], trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.query_loss_metric.update_state(avg_loss)
        logs = {
            self.query_loss_metric.name: self.query_loss_metric.result(),
        }
        return logs

    def run(self, lr=0.001, weights=None, ways=5, shots=5, test_shots=15,
            data_dir_path="/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224"):
        if weights is not None:
            self.load_weights(weights, by_name=True, skip_mismatch=True)

        dataloader = DataLoader(data_dir_path=data_dir_path)

        meta_test_ds, meta_test_name_projector = dataloader.get_dataset(phase='test', way_num=ways, shot_num=shots,
                                                                        episode_test_sample_num=test_shots,
                                                                        episode_num=600,
                                                                        batch=4,
                                                                        augment=False)
        ways = 5
        shots = 5
        train_test_num = 6
        train_batch = 4
        episode_num = 1200
        steps_per_epoch = episode_num // train_batch
        train_epoch = 20
        mix_up = True
        augment = True
        meta_train_ds, meta_train_name_projector = dataloader.get_dataset_V2(phase='train', way_num=ways,
                                                                             shot_num=shots,
                                                                             episode_test_sample_num=train_test_num,
                                                                             episode_num=episode_num,
                                                                             batch=train_batch,
                                                                             augment=augment,
                                                                             mix_up=mix_up,
                                                                             epochs=train_epoch)
        scheduled_lrs = WarmUpStep(
            learning_rate_base=lr,
            warmup_learning_rate=0.0,
            warmup_steps=steps_per_epoch * 3,
        )

        self.compile(tf.keras.optimizers.Adam(scheduled_lrs))
        self.train_step = self.train_step_normal

        self.test_step = self.test_step_meta
        self.predict_step = self.test_step_meta

        # for data in meta_train_ds:
        #     self.train_step(data)
        # for data in meta_test_ds:
        #     self.train_step(data)
        # self.test_step(data)
        monitor_name = "val_mean_query_acc"
        monitor_cmp = "max"
        cur_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = self.name
        log_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_base_path = os.path.join(ckpt_base_path, "{}_ckpts/".format(name))
        print(log_base_path, ckpt_base_path)
        filepath = os.path.join(ckpt_base_path,
                                "model_e{}-l {}.h5".format("{epoch:03d}",
                                                           "{" + "{}:.5f".format(monitor_name) + "}"))
        os.makedirs(ckpt_base_path, exist_ok=True)

        tensorboard_save = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_base_path, '{}_logs/{}'.format(name, cur_date)),
            profile_batch=0, )
        checkpoint_save = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                             verbose=1, monitor=monitor_name,
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode=monitor_cmp)

        callbacks = [
            checkpoint_save,
            tensorboard_save,
            tf.keras.callbacks.ModelCheckpoint(os.path.join(ckpt_base_path, "latest.h5"),
                                               verbose=1, monitor=monitor_name,
                                               save_best_only=False,
                                               save_weights_only=True,
                                               mode=monitor_cmp)
        ]

        self.fit(meta_train_ds.repeat(), epochs=1000,
                 steps_per_epoch=steps_per_epoch,
                 validation_data=meta_test_ds,
                 callbacks=callbacks, initial_epoch=0)

    def fine_tune(self, lr=0.001, weights=None, ways=5, shots=5, test_shots=15,
                  data_dir_path="/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224"):
        if weights is not None:
            self.load_weights(weights, by_name=True, skip_mismatch=True)

        dataloader = MiniImageNetDataLoader_v2(data_dir_path=data_dir_path)

        meta_test_ds, meta_test_name_projector = dataloader.get_dataset(phase='val', way_num=ways, shot_num=shots,
                                                                        episode_test_sample_num=test_shots,
                                                                        episode_num=600,
                                                                        batch=4,
                                                                        augment=False)
        ways = 5
        shots = 5
        train_test_num = 5
        train_batch = 4
        episode_num = 100
        steps_per_epoch = episode_num // train_batch
        train_epoch = 50
        mix_up = True
        meta_train_ds, meta_train_name_projector = dataloader.get_dataset(phase='train', way_num=ways,
                                                                          shot_num=shots,
                                                                          episode_test_sample_num=train_test_num,
                                                                          episode_num=episode_num,
                                                                          batch=train_batch,
                                                                          augment=True,
                                                                          mix_up=mix_up,
                                                                          epochs=train_epoch)
        if mix_up is True:
            steps_per_epoch = steps_per_epoch * 3
        else:
            steps_per_epoch = steps_per_epoch * 2
        TOTAL_STEPS = steps_per_epoch * train_epoch
        scheduled_lrs = WarmUpCosine(
            learning_rate_base=lr,
            total_steps=TOTAL_STEPS,
            warmup_learning_rate=0.0,
            warmup_steps=steps_per_epoch,
        )
        self.compile(tf.keras.optimizers.SGD(scheduled_lrs, momentum=0.9, nesterov=True))
        self.train_step = self.train_step_meta

        self.test_step = self.test_step_meta

        # for data in meta_train_ds:
        #     self.train_step(data)
        # for data in meta_test_ds:
        #     self.test_step_meta(data)
        monitor_name = "val_{}".format("acc")
        monitor_cmp = "max"
        monitor_name = "val_mean_query_acc"
        monitor_cmp = "max"
        cur_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = self.name
        log_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_base_path = os.path.join(ckpt_base_path, "{}_ckpts/".format(name))
        print(log_base_path, ckpt_base_path)
        filepath = os.path.join(ckpt_base_path,
                                "model_e{}-l {}.h5".format("{epoch:03d}",
                                                           "{" + "{}:.5f".format(monitor_name) + "}"))
        os.makedirs(ckpt_base_path, exist_ok=True)

        tensorboard_save = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_base_path, '{}_logs/{}'.format(name, cur_date)),
            profile_batch=0, )
        checkpoint_save = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                             verbose=1, monitor=monitor_name,
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode=monitor_cmp)

        callbacks = [
            checkpoint_save,
            tensorboard_save,
        ]
        self.evaluate(meta_test_ds)
        self.fit(meta_train_ds.repeat(), epochs=train_epoch,
                 steps_per_epoch=steps_per_epoch,
                 validation_data=meta_test_ds,
                 callbacks=callbacks, initial_epoch=0)

    def show(self, weights=None, data_dir_path="/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224",
             phase="test"):
        if weights is not None:
            self.load_weights(weights, by_name=True, skip_mismatch=True)

        dataloader = DataLoader(data_dir_path=data_dir_path)

        meta_test_ds, name_projector = dataloader.get_dataset_V2(phase=phase, way_num=5, shot_num=5,
                                                                 episode_test_sample_num=5,
                                                                 episode_num=600,
                                                                 augment=False,
                                                                 mix_up=False,
                                                                 batch=1)

        self.train_step = self.train_step_normal
        cv2.namedWindow("image", 1)
        cv2.namedWindow("q_show_image", 0)
        cv2.namedWindow("q_image", 0)

        count = 0
        for x, y in meta_test_ds:
            support, query = x, y
            support_image, support_label, support_global_label = support
            query_image, query_label, query_global_label = query
            batch, ways, shots, h, w, ch = tf.unstack(tf.shape(support_image))
            query_shots = tf.shape(query_image)[2]
            # support_image = tf.reshape(support_image, [batch, ways, 1, shots * h, w, ch])
            # batch, ways, shots, h, w, ch = tf.unstack(tf.shape(support_image))
            dim_shape = tf.shape(support_label)[-1]
            training = False

            support_image = tf.reshape(support_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))
            # support_image = tf.map_fn(do_augmentations, images, parallel_iterations=16)
            training = False

            support_features = self.encoder(support_image, training=training)
            _, f_h, f_w, f_c = tf.unstack(tf.shape(support_features))
            support_logits = self.gap(self.last_max_pooling(support_features))
            support_logits = tf.nn.l2_normalize(support_logits, -1)

            support_logits_base = tf.reshape(support_logits,
                                             [batch, ways, shots, tf.shape(support_logits)[-1]])
            support_logits_base = tf.reduce_mean(support_logits_base, 2)
            support_logits_base = tf.nn.l2_normalize(support_logits_base, -1)
            support_logits_mean = tf.reshape(support_logits_base, [batch, 1, ways, 1, 1, f_c])
            support_logits_mean_broad = tf.broadcast_to(support_logits_mean, [batch, ways * shots, ways, f_h, f_w, f_c])
            support_features_broad = tf.reshape(support_features, [batch, ways * shots, 1, f_h, f_w, f_c])
            support_features_broad = tf.broadcast_to(support_features_broad, [batch, ways * shots, ways, f_h, f_w, f_c])
            # merge_feature_support = tf.concat(
            #     [tf.nn.l2_normalize(support_features_broad, -1), support_logits_mean_broad], -1)
            merge_feature_support = tf.nn.l2_normalize(support_features_broad, -1) * support_logits_mean_broad
            merge_feature_support = tf.reshape(merge_feature_support,
                                               [-1, f_h, f_w, tf.shape(merge_feature_support)[-1]])
            support_self_attention = self.self_attention_referenced_conv(merge_feature_support, training=training)
            support_self_attention = tf.reshape(support_self_attention, [-1, ways, f_h, f_w])
            support_self_attention = tf.transpose(support_self_attention, [0, 2, 3, 1])
            support_self_attention = tf.nn.softmax(support_self_attention * 1., -1)

            support_self_attention = support_self_attention * tf.reshape(support_label, [-1, 1, 1, dim_shape])
            support_self_attention = tf.reduce_sum(support_self_attention, -1, keepdims=True)

            support_features = tf.nn.l2_normalize(support_features, -1)
            support_logits_attention = support_features * support_self_attention
            support_logits_attention = tf.math.divide_no_nan(tf.reduce_sum(support_logits_attention, [1, 2]),
                                                             tf.reduce_sum(support_self_attention, [1, 2]))
            support_logits_attention = tf.nn.l2_normalize(support_logits_attention, -1)
            support_logits_fusion = tf.concat([support_logits, support_logits_attention], -1)

            support_logits_fusion = tf.reshape(support_logits_fusion,
                                               [batch, ways, shots, tf.shape(support_logits_fusion)[-1]])
            support_logits_fusion = tf.nn.l2_normalize(support_logits_fusion, -1)

            support_logits_fusion = tf.reshape(support_logits_fusion,
                                               [batch, ways, shots, tf.shape(support_logits_fusion)[-1]])

            support_logits_fusion = tf.nn.l2_normalize(support_logits_fusion, -1)
            x_mean = tf.reduce_mean(support_logits_fusion, 2)
            x_mean = tf.reshape(x_mean, [batch, ways, 1, 1, 1, -1])
            support_image = tf.reshape(support_image,
                                       [batch, ways, shots, support_image.shape[-3], *support_image.shape[-2:]])
            support_self_attention = tf.reshape(support_self_attention, [batch, ways, shots, f_h, f_w, 1])

            def transpose_and_reshape(x):
                b, way, s, h, w, c = tf.unstack(tf.shape(x))
                x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
                x = tf.reshape(x, [b, way * h, s * w, c])
                return x

            support_image = transpose_and_reshape(support_image)
            support_self_attention = transpose_and_reshape(support_self_attention)
            referenced_support_means = tf.reshape(x_mean, [batch, 1, 1, ways, 1, 1, -1])
            # query_image = self.random_enrode(query_image)

            q_referenced_image = tf.reshape(query_image,
                                            [batch, ways, query_shots, 1, query_image.shape[-2],
                                             *query_image.shape[-2:]])
            q_referenced_image = tf.broadcast_to(q_referenced_image,
                                                 [batch, ways, query_shots, ways, query_image.shape[-2],
                                                  *query_image.shape[-2:]])
            q_referenced_image = tf.reshape(q_referenced_image, [batch, ways, query_shots, ways * query_image.shape[-2],
                                                                 *query_image.shape[-2:]])
            q_referenced_image = transpose_and_reshape(q_referenced_image)
            query_image = tf.reshape(query_image, tf.concat([[-1], tf.shape(query_image)[-3:]], 0))

            query_features = self.encoder(query_image, training=training)
            query_logits = self.gap(self.last_max_pooling(query_features))
            query_logits = tf.nn.l2_normalize(query_logits, -1)

            support_logits_broad = tf.broadcast_to(support_logits_mean,
                                                   [batch, ways * query_shots, ways, f_h, f_w, f_c])

            features_query = tf.reshape(query_features, [batch, ways * query_shots, 1, f_h, f_w, f_c])
            features_query_broad = tf.broadcast_to(features_query, [batch, ways * query_shots, ways, f_h, f_w, f_c])
            # merge_feature_query = tf.concat(
            #     [tf.nn.l2_normalize(features_query_broad, -1), support_logits_broad], -1)

            merge_feature_query = tf.nn.l2_normalize(features_query_broad, -1) * support_logits_broad
            merge_feature_query = tf.reshape(merge_feature_query,
                                             [-1, f_h, f_w, tf.shape(merge_feature_query)[-1]])
            query_enhanced_mask = self.self_attention_referenced_conv(merge_feature_query, training=training)
            query_enhanced_mask = tf.reshape(query_enhanced_mask, [-1, ways, f_h, f_w])
            query_self_attention = tf.transpose(query_enhanced_mask, [0, 2, 3, 1])
            query_self_attention = tf.nn.softmax(query_self_attention * 1., -1)
            # query_self_attention = tf.reduce_max(query_self_attention, -1, keepdims=True)

            query_features = tf.nn.l2_normalize(query_features, -1)

            query_features = tf.concat([query_features, query_features], -1)
            _, f_h, f_w, f_c = tf.unstack(tf.shape(query_features))

            reshaped_query_features = tf.reshape(query_features,
                                                 [batch, ways, query_shots, 1, f_h, f_w, f_c])
            reshaped_query_features = tf.broadcast_to(reshaped_query_features,
                                                      [batch, ways, query_shots, ways, f_h, f_w, f_c])
            referenced_attention = -tf.losses.cosine_similarity(
                tf.broadcast_to(referenced_support_means, tf.shape(reshaped_query_features)),
                reshaped_query_features)
            referenced_attention = tf.clip_by_value(referenced_attention, 0., 1.)
            # referenced_attention = tf.cast(tf.greater(referenced_attention, 0.3), tf.float32)
            referenced_attention = tf.expand_dims(referenced_attention, -1)
            referenced_attention = tf.reshape(referenced_attention, [batch, ways, query_shots, ways * f_h, f_w, 1])
            referenced_attention = transpose_and_reshape(referenced_attention)

            query_image = tf.reshape(query_image,
                                     [batch, ways, query_shots, query_image.shape[-2], *query_image.shape[-2:]])
            query_self_attention = tf.reshape(query_self_attention, [batch, ways, query_shots, f_h, f_w, -1])

            query_image = transpose_and_reshape(query_image)
            query_self_attention = transpose_and_reshape(query_self_attention)

            for q_image, r_q_attention in \
                    zip(q_referenced_image,
                        referenced_attention):
                q_image = (q_image[..., ::-1] * 255).numpy().astype(np.uint8)

                r_q_attention = tf.image.resize(r_q_attention * 255, q_image.shape[-3:-1],
                                                method='bilinear').numpy().astype(
                    np.uint8)
                r_q_attention = cv2.applyColorMap(r_q_attention, cv2.COLORMAP_JET)
                r_q_attention = cv2.addWeighted(q_image, 0.5, r_q_attention, 0.5, 0)

                q_show_image = cv2.hconcat([q_image, r_q_attention])
                cv2.imshow("q_image", q_show_image)
                cv2.waitKey(1)

            for image, origin_s_attention, q_image, origin_q_attention in \
                    zip(support_image,
                        support_self_attention,
                        query_image,
                        query_self_attention):
                image = (image[..., ::-1] * 255).numpy().astype(np.uint8)

                origin_s_attention = tf.image.resize(origin_s_attention * 255, image.shape[-3:-1],
                                                     method='bilinear').numpy().astype(
                    np.uint8)
                origin_s_attention = cv2.applyColorMap(origin_s_attention, cv2.COLORMAP_JET)
                origin_s_attention = cv2.addWeighted(image, 0.5, origin_s_attention, 0.5, 0)

                show_image = cv2.hconcat([image, origin_s_attention])
                # show_image = cv2.hconcat([image, origin_s_attention])
                # show_image = cv2.transpose(show_image)

                cv2.imshow("image", show_image)

                q_image = (q_image[..., ::-1] * 255).numpy().astype(np.uint8)
                origin_q_attention_max = tf.reduce_max(origin_q_attention, -1, keepdims=True)
                origin_q_attention_max = tf.image.resize(origin_q_attention_max * 255, q_image.shape[-3:-1],
                                                         method='bilinear')
                origin_q_attention_max = cv2.addWeighted(q_image, 0.5, cv2.applyColorMap(
                    origin_q_attention_max.numpy().astype(np.uint8),
                    cv2.COLORMAP_JET), 0.5,
                                                         0)
                origin_q_attention = tf.image.resize(origin_q_attention * 255, q_image.shape[-3:-1],
                                                     method='bilinear')

                origin_q_attention = tf.split(origin_q_attention, 5, -1)
                origin_q_attention = [
                    cv2.addWeighted(q_image, 0.5, cv2.applyColorMap(a.numpy().astype(np.uint8), cv2.COLORMAP_JET), 0.5,
                                    0)
                    for a in origin_q_attention]
                origin_q_attention = tf.stack(origin_q_attention, 1)
                origin_q_attention = tf.reshape(origin_q_attention, [tf.shape(origin_q_attention)[0], -1, 3],
                                                ).numpy().astype(np.uint8)

                q_show_image = cv2.hconcat([q_image, origin_q_attention_max, origin_q_attention])
                cv2.imshow("q_show_image", q_show_image)
                cv2.waitKey(0)

    # @tf.function
    def test_step_meta(self, data):
        support, query = data
        support_image, support_label, _ = support
        query_image, query_label, _ = query

        batch = tf.shape(support_image)[0]
        ways = tf.shape(support_image)[1]
        shots = tf.shape(support_image)[2]
        query_shots = tf.shape(query_image)[2]
        dim_shape = tf.shape(support_label)[-1]

        support_image = tf.reshape(support_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))

        training = False

        support_features = self.encoder(support_image, training=training)
        _, f_h, f_w, f_c = tf.unstack(tf.shape(support_features))
        support_logits = self.gap(self.last_max_pooling(support_features))
        support_logits = tf.nn.l2_normalize(support_logits, -1)

        support_logits_base = tf.reshape(support_logits,
                                         [batch, ways, shots, tf.shape(support_logits)[-1]])
        support_logits_base = tf.reduce_mean(support_logits_base, 2)
        support_logits_base = tf.nn.l2_normalize(support_logits_base, -1)
        support_logits_mean = tf.reshape(support_logits_base, [batch, 1, ways, 1, 1, f_c])
        support_logits_mean_broad = tf.broadcast_to(support_logits_mean, [batch, ways * shots, ways, f_h, f_w, f_c])
        support_features_broad = tf.reshape(support_features, [batch, ways * shots, 1, f_h, f_w, f_c])
        support_features_broad = tf.broadcast_to(support_features_broad, [batch, ways * shots, ways, f_h, f_w, f_c])
        # merge_feature_support = tf.concat([tf.nn.l2_normalize(support_features_broad, -1), support_logits_mean_broad],
        #                                   -1)
        merge_feature_support = tf.nn.l2_normalize(support_features_broad, -1) * support_logits_mean_broad
        merge_feature_support = tf.reshape(merge_feature_support, [-1, f_h, f_w, tf.shape(merge_feature_support)[-1]])
        support_self_attention = self.self_attention_referenced_conv(merge_feature_support, training=training)
        support_self_attention = tf.reshape(support_self_attention, [-1, ways, f_h, f_w])
        support_self_attention = tf.transpose(support_self_attention, [0, 2, 3, 1])
        support_self_attention = tf.nn.softmax(support_self_attention * 1., -1)

        support_self_attention = support_self_attention * tf.reshape(support_label, [-1, 1, 1, dim_shape])
        support_self_attention = tf.reduce_sum(support_self_attention, -1, keepdims=True)

        support_features = tf.nn.l2_normalize(support_features, -1)
        support_logits_attention = support_features * support_self_attention
        support_logits_attention = tf.math.divide_no_nan(tf.reduce_sum(support_logits_attention, [1, 2]),
                                                         tf.reduce_sum(support_self_attention, [1, 2]))
        support_logits_attention = tf.nn.l2_normalize(support_logits_attention, -1)
        support_logits_fusion = tf.concat([support_logits, support_logits_attention], -1)

        support_logits_fusion = tf.reshape(support_logits_fusion,
                                           [batch, ways, shots, tf.shape(support_logits_fusion)[-1]])
        support_logits_fusion = tf.nn.l2_normalize(support_logits_fusion, -1)
        x_mean = tf.reduce_mean(support_logits_fusion, 2)

        x_mean_base = support_logits_base

        support_logits_attention = tf.reshape(support_logits_attention,
                                              [batch, ways, shots, tf.shape(support_logits_attention)[-1]])
        support_logits_attention = tf.nn.l2_normalize(support_logits_attention, -1)
        x_mean_attention = tf.reduce_mean(support_logits_attention, 2)

        new_shape = tf.concat([[-1], tf.shape(query_image)[-3:]], axis=0)
        query_image = tf.reshape(query_image, new_shape)

        query_features = self.encoder(query_image, training=training)
        query_logits = self.gap(self.last_max_pooling(query_features))
        query_logits = tf.nn.l2_normalize(query_logits, -1)

        logits_dim = tf.shape(x_mean_base)[-1]

        support_mean_base = tf.reshape(x_mean_base, [batch, ways, logits_dim])
        support_mean_base = tf.nn.l2_normalize(support_mean_base, -1)
        reshape_query_logits_base = tf.reshape(query_logits, [batch, ways * query_shots, logits_dim])
        reshape_query_logits_base = tf.nn.l2_normalize(reshape_query_logits_base, -1)
        dist_base = tf.linalg.matmul(reshape_query_logits_base, support_mean_base, transpose_b=True)

        query_pred_base = tf.clip_by_value(dist_base, 0., 1.)

        loss_base = tf.losses.binary_crossentropy(tf.reshape(query_label, [batch, -1, dim_shape, 1]),
                                                  tf.expand_dims(query_pred_base, -1))
        loss_base = tf.reduce_mean(loss_base)

        acc_base = tf.keras.metrics.categorical_accuracy(tf.reshape(query_label, [batch, -1, dim_shape]),
                                                         dist_base)
        acc_base = tf.reduce_mean(acc_base, -1)
        self.mean_query_acc_base.update_state(acc_base)
        self.query_loss_metric_base.update_state(loss_base)

        support_logits_broad = tf.broadcast_to(support_logits_mean, [batch, ways * query_shots, ways, f_h, f_w, f_c])

        # support_logits_attention = tf.reshape(support_logits_attention,
        #                                       [batch, ways, shots, tf.shape(support_logits)[-1]])
        # support_logits_attention_base = tf.reduce_mean(support_logits_attention, 2)
        # support_logits_attention_base = tf.nn.l2_normalize(support_logits_attention_base, -1)
        # support_logits_attention_mean = tf.reshape(support_logits_attention_base, [batch, 1, ways, 1, 1, f_c])
        # support_logits_attention_mean_broad = tf.broadcast_to(support_logits_attention_mean,
        #                                                       [batch, ways * query_shots, ways, f_h, f_w, f_c])

        features_query = tf.reshape(query_features, [batch, ways * query_shots, 1, f_h, f_w, f_c])
        features_query_broad = tf.broadcast_to(features_query, [batch, ways * query_shots, ways, f_h, f_w, f_c])
        # merge_feature_query = tf.concat(
        #     [tf.nn.l2_normalize(features_query_broad, -1), support_logits_broad], -1)
        merge_feature_query = tf.nn.l2_normalize(features_query_broad, -1) * support_logits_broad
        merge_feature_query = tf.reshape(merge_feature_query, [-1, f_h, f_w, tf.shape(merge_feature_query)[-1]])
        query_enhanced_mask = self.self_attention_referenced_conv(merge_feature_query, training=training)
        query_enhanced_mask = tf.reshape(query_enhanced_mask, [-1, ways, f_h, f_w])
        query_self_attention = tf.transpose(query_enhanced_mask, [0, 2, 3, 1])
        query_self_attention = tf.nn.softmax(query_self_attention * 1., -1)
        # query_self_attention = tf.reduce_sum(query_self_attention, -1, keepdims=True)
        # query_self_attention = tf.reduce_max(query_self_attention, -1, keepdims=True)
        query_self_attention = tf.expand_dims(query_self_attention, -1)
        query_features = tf.nn.l2_normalize(query_features, -1)
        query_logits_attention = tf.expand_dims(query_features, -2) * query_self_attention
        query_logits_attention = tf.math.divide_no_nan(tf.reduce_sum(query_logits_attention, [1, 2]),
                                                       tf.reduce_sum(query_self_attention, [1, 2]))
        query_logits_attention = tf.nn.l2_normalize(query_logits_attention, -1)
        query_logits_fusion = tf.concat([tf.repeat(tf.expand_dims(query_logits, -2), ways, -2), query_logits_attention],
                                        -1)

        logits_dim = tf.shape(support_logits_fusion)[-1]

        support_mean = tf.reshape(x_mean, [batch, ways, logits_dim])
        support_mean = tf.nn.l2_normalize(support_mean, -1)
        reshape_query_logits = tf.reshape(query_logits_fusion, [batch, ways * query_shots * ways, logits_dim])
        reshape_query_logits = tf.nn.l2_normalize(reshape_query_logits, -1)
        dist = tf.linalg.matmul(reshape_query_logits, support_mean, transpose_b=True)
        query_pred = tf.clip_by_value(dist, 0., 1.)
        query_pred = tf.reshape(query_pred, [batch, ways * query_shots, ways, -1])
        query_pred = tf.reduce_max(query_pred, -2)

        acc = tf.keras.metrics.categorical_accuracy(tf.reshape(query_label, [batch, -1, dim_shape]), query_pred)
        acc = tf.reduce_mean(acc, -1)
        self.mean_query_acc.update_state(acc)

        logits_dim = tf.shape(x_mean_attention)[-1]

        support_mean_attention = tf.reshape(x_mean_attention, [batch, ways, logits_dim])
        support_mean_attention = tf.nn.l2_normalize(support_mean_attention, -1)
        reshape_query_logits_attention = tf.reshape(query_logits_attention,
                                                    [batch, ways * query_shots * ways, logits_dim])
        reshape_query_logits_attention = tf.nn.l2_normalize(reshape_query_logits_attention, -1)
        dist_attention = tf.linalg.matmul(reshape_query_logits_attention, support_mean_attention, transpose_b=True)
        query_pred_attention = tf.clip_by_value(dist_attention, 0., 1.)
        query_pred_attention = tf.reshape(query_pred_attention, [batch, ways * query_shots, ways, -1])
        query_pred_attention = tf.reduce_max(query_pred_attention, -2)

        acc_attention = tf.keras.metrics.categorical_accuracy(tf.reshape(query_label, [batch, -1, dim_shape]),
                                                              query_pred_attention)
        acc_attention = tf.reduce_mean(acc_attention, -1)
        self.mean_query_acc_attention.update_state(acc_attention)

        logs = {
            self.mean_query_acc.name: self.mean_query_acc.result(),
            self.mean_query_acc_attention.name: self.mean_query_acc_attention.result(),
            self.mean_query_acc_base.name: self.mean_query_acc_base.result(),
            "mean_query_acc_current": tf.reduce_mean(acc, -1),
        }
        return logs

    def test(self, weights=None, ways=5, shots=5, episode_num=10000):
        if weights is not None:
            self.load_weights(weights, by_name=True, skip_mismatch=True)

        data_dir_path = "/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224"
        # data_dir_path = "/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224"

        dataloader = MiniImageNetDataLoader_v2(data_dir_path=data_dir_path)

        meta_test_ds, meta_test_name_projector = dataloader.get_dataset(phase='test', way_num=ways, shot_num=shots,
                                                                        episode_test_sample_num=15,
                                                                        episode_num=episode_num,
                                                                        batch=4,
                                                                        augment=False)

        self.compile(tf.keras.optimizers.Adam(0.0001))
        self.train_step = self.train_step_normal

        self.test_step = self.test_step_meta
        self.predict_step = self.test_step_meta
        self.reset_states()
        # for data in meta_test_ds:
        #     self.test_step(data)

        monitor_name = "val_mean_query_acc"
        monitor_cmp = "max"
        save_best_only = True
        cur_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = self.name
        log_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_base_path = os.path.join(ckpt_base_path, "{}_ckpts/".format(name))
        print(log_base_path, ckpt_base_path)
        filepath = os.path.join(ckpt_base_path,
                                "model_e{}-l {}.h5".format("{epoch:03d}",
                                                           "{" + "{}:.5f".format(monitor_name) + "}"))
        os.makedirs(ckpt_base_path, exist_ok=True)

        tensorboard_save = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_base_path, '{}_logs/{}'.format(name, cur_date)),
            profile_batch=0, )
        checkpoint_save = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                             verbose=1, monitor=monitor_name,
                                                             save_best_only=save_best_only,
                                                             # save_best_only=False,
                                                             mode=monitor_cmp)

        callbacks = [
            checkpoint_save,
            tensorboard_save,
        ]

        self.reset_states()
        # out = self.predict(meta_test_ds, verbose=1)
        # acc = out["mean_query_acc_current"]
        # acc = acc * 100.
        # mean = np.mean(acc)
        # std = np.std(acc)
        # # #
        # pm = 1.96 * (std / np.sqrt(len(acc)))
        # print(mean, std, pm)
        # pm = st.t.interval(0.95, len(acc) - 1, loc=np.mean(acc), scale=st.sem(acc)) - mean
        # print(mean, pm)
        out = self.evaluate(meta_test_ds, return_dict=True)

        # acc = self.predict(meta_test_ds)["mean_query_acc"]
        # # print(acc)
        # mean = np.mean(acc)
        # std = np.std(acc)
        # #
        # pm = 1.96 * (std / np.sqrt(len(acc)))
        # print(mean, std, pm)


multi_gpu = True
seed = 100
random.seed(seed)
if multi_gpu is True:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = FSLModel(imageshape=(84, 84, 3), num_class=64)
else:
    model = FSLModel(imageshape=(84, 84, 3), num_class=64)
# model.run()
# model.run(weights="/data/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name))
model.test(weights="/data/giraffe/0_FSL/{}_ckpts/model_e120-l 0.81242.h5".format(model_name), shots=5)
# model.init("/data/giraffe/0_FSL/TRSN_ckpts/model_e526-l 0.84958.h5", phase="train")
# model.init_and_test("{}.h5".format(model.name), phase="train")
# model.show("/data/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name), phase="test")
# model.test(weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e047-l 0.84964.h5", shots=1)
# model.test(weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e047-l 0.84964.h5", shots=5)
# model.fine_tune(lr=0.0001, weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e526-l 0.84958.h5")
