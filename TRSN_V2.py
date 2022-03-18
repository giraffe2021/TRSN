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
        feature_size_h, feature_size_w, feature_dim = [*self.encoder.output.shape[1:]]
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="gap")
        self.gap.build([None, feature_size_h, feature_size_w, feature_dim])

        self.last_max_pooling = tf.keras.layers.MaxPool2D(padding="same", name="last_max_pooling")
        self.last_max_pooling.build([None, feature_size_h, feature_size_w, feature_dim])
        self.clc = tf.keras.layers.Dense(self.num_class,
                                         kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                         name="clc")
        self.clc.build([None, feature_dim])

        self.clc_hinted = tf.keras.layers.Dense(self.num_class,
                                                kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                                name="clc_hinted")
        self.clc_hinted.build([None, feature_dim])

        self.rotation_head = tf.keras.layers.Dense(1,
                                                   activation="relu",
                                                   kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                                   name="rotation_head")
        self.rotation_head.build([None, feature_dim])

        self.ratio_head = tf.keras.layers.Dense(1,
                                                activation="relu",
                                                kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                                name="ratio_head")
        self.ratio_head.build([None, feature_dim * 2])

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
                                                          layers.Activation("sigmoid")
                                                          ],
                                                         name="self_attention_referenced_conv")
        self.self_attention_referenced_conv.build([None, feature_size_h, feature_size_w, feature_dim])

        self.build([None, *imageshape])
        self.acc = tf.keras.metrics.CategoricalAccuracy(name="acc")
        self.loss_metric = tf.keras.metrics.Mean(name="clc_loss")
        self.rotation_metric = tf.keras.metrics.Mean(name="rotate_loss")
        self.ratio_metric = tf.keras.metrics.Mean(name="ratio_loss")
        self.hinted_loss_metric = tf.keras.metrics.Mean(name="hinted_clc_loss")
        self.er_loss_metric = tf.keras.metrics.Mean(name="er_loss")
        self.cluster_loss_metric = tf.keras.metrics.Mean(name="cluster_loss")

        self.query_loss_metric = tf.keras.metrics.Mean("query_loss")
        self.mean_query_acc = tf.keras.metrics.Mean(name="mq_acc")

        self.mean_query_acc_base = tf.keras.metrics.Mean(name="mq_acc_base")

        self.build([None, *imageshape])
        self.summary()

    def call(self, inputs, training=None):
        return None

    def reset_metrics(self):
        # Resets the state of all the metrics in the model.
        for m in self.metrics:
            m.reset_states()

        self.acc.reset_states()
        self.loss_metric.reset_states()
        self.hinted_loss_metric.reset_states()
        self.rotation_metric.reset_states()
        self.ratio_metric.reset_states()
        self.er_loss_metric.reset_states()
        self.cluster_loss_metric.reset_states()

        self.query_loss_metric.reset_states()
        self.mean_query_acc.reset_states()

        self.mean_query_acc_base.reset_states()

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
        pred_special_softmax = tf.nn.softmax(pred_special, -1)
        return pred_special, pred_special_softmax

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

        support_logits_merge = tf.reshape(support_logits,
                                          [batch, ways * shots, tf.shape(support_logits)[-1]])
        support_logits_merge = tf.nn.l2_normalize(support_logits_merge, -1)
        support_logits_merge = tf.reduce_mean(support_logits_merge, 1)
        support_logits_merge = tf.reshape(support_logits_merge, [batch, 1, 1, 1, f_c])
        support_logits_merge = tf.broadcast_to(support_logits_merge, [batch, ways * query_shots, f_h, f_w, f_c])
        support_logits_merge = tf.reshape(support_logits_merge, [-1, f_h, f_w, f_c])
        merge_feature_query = tf.concat([features_query, support_logits_merge], -1)
        merge_feature_query = tf.stop_gradient(merge_feature_query)
        self_attention_query = self.self_attention_referenced_conv(merge_feature_query, training=training)

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

        pseudo_label = 0.5 * logits_weights_query_self_stop_gradient + 0.5 * logits_weights_query_referenced_stop_gradient
        # pseudo_label = logits_weights_query_referenced_stop_gradient
        salient_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(pseudo_label, self_attention_query))

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

    def baseline_task_train_step(self, data, training=True):
        support, query = data
        support_image, _, support_global_label = support
        query_image, _, query_global_label = query

        image_shape = tf.unstack(tf.shape(support_image)[-3:])
        global_dim_shape = tf.shape(support_global_label)[-1]

        batch = tf.shape(support_image)[0]
        ways = tf.shape(support_image)[1]
        shots = tf.shape(support_image)[2]
        query_shots = tf.shape(query_image)[2]

        support_x = tf.reshape(support_image, [-1, *image_shape])
        query_x = tf.reshape(query_image, [-1, *image_shape])

        support_global_label = tf.reshape(support_global_label, [-1, global_dim_shape])
        query_global_label = tf.reshape(query_global_label, [-1, global_dim_shape])

        y = tf.concat([support_global_label, query_global_label], 0)
        support_logits = self.gap(self.encoder(support_x, training=training))
        query_logits = self.gap(self.encoder(query_x, training=training))
        logits = tf.concat([support_logits, query_logits], 0)
        pred = self.clc(logits)
        loss_clc = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y, pred, from_logits=True))
        support_logits = tf.reshape(support_logits, [batch, ways, shots, -1])
        query_logits = tf.reshape(query_logits, [batch, ways, query_shots, -1])
        return loss_clc, support_logits, query_logits

    # @tf.function
    def train_step_normal(self, data):
        x, y = data
        image, x_rotated_1, x_rotated_2, x_resized_1, x_resized_2 = x
        y, rotate_label_1, rotate_label_2, resize_label_1, resize_label_2 = y
        training = True
        batch = tf.shape(image)[0]

        x = tf.concat([image, x_rotated_1, x_rotated_2, x_resized_1, x_resized_2], 0)
        y_all = tf.concat([y, y, y, y, y], 0)
        with tf.GradientTape() as tape:
            features = self.encoder(x, training=training)
            f_b, f_h, f_w, f_c = tf.unstack(tf.shape(features))
            logits = self.gap(self.last_max_pooling(features))
            pred = self.clc(logits, training=training)

            logits_base, logits_rotated_1, logits_rotated_2, logits_resized_1, logits_resized_2 \
                = tf.split(logits, [batch, batch, batch, batch, batch], 0)
            clc_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_all, pred, from_logits=True))

            all_protos = tf.reduce_mean(logits, 0)
            all_protos = tf.reshape(all_protos, [1, 1, f_c])
            reshaped_features = tf.reshape(features, [f_b, f_h * f_w, f_c])
            similarity_map = tf.reduce_sum(
                tf.nn.l2_normalize(reshaped_features, -1) * tf.nn.l2_normalize(all_protos, -1), -1)
            similarity_map = tf.reshape(similarity_map, [f_b, f_h, f_w, 1])
            hinted_features = features + similarity_map
            hinted_attention = self.self_attention_referenced_conv(hinted_features, training=training)

            hinted_logits = features * hinted_attention
            hinted_logits = tf.math.divide_no_nan(tf.reduce_sum(hinted_logits, [1, 2]),
                                                  tf.reduce_sum(hinted_attention, [1, 2]))
            hinted_pred = self.clc_hinted(hinted_logits, training=training)
            hinted_clc_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y_all, hinted_pred, from_logits=True))

            rotate_pred = self.rotation_head(tf.concat([logits_base, logits_rotated_1, logits_rotated_2], 0),
                                             training=training)

            rotated_label = tf.concat([tf.zeros_like(rotate_label_1),
                                       rotate_label_1,
                                       rotate_label_2], 0)
            # rotated_label = tf.squeeze(rotated_label, 1)
            rotate_loss = tf.reduce_mean(
                tf.keras.losses.mse(rotated_label, rotate_pred))

            size_logits_1 = tf.concat([logits_base, logits_resized_1], -1)
            size_logits_2 = tf.concat([logits_base, logits_resized_2], -1)
            size_logits_self = tf.concat([logits_base, logits_base], -1)

            size_logits = tf.concat([size_logits_1,
                                     size_logits_2,
                                     size_logits_self], 0)
            size_pred = self.ratio_head(size_logits)

            size_label = tf.concat([resize_label_1,
                                    resize_label_2,
                                    tf.ones_like(resize_label_1)], 0)
            # #
            ratio_loss = tf.reduce_mean(
                tf.keras.losses.mse(size_pred, size_label))

        trainable_vars = self.encoder.trainable_weights + self.clc.trainable_weights \
                         + self.rotation_head.trainable_weights \
                         + self.ratio_head.trainable_weights \
                         + self.clc_hinted.trainable_weights \
                         + self.self_attention_referenced_conv.trainable_weights
        grads = tape.gradient([clc_loss, hinted_clc_loss, rotate_loss, ratio_loss], trainable_vars)
        # grads = tape.gradient([clc_loss], trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.loss_metric.update_state(clc_loss)
        self.hinted_loss_metric.update_state(hinted_clc_loss)
        self.rotation_metric.update_state(rotate_loss)
        self.ratio_metric.update_state(ratio_loss)
        # self.er_loss_metric.update_state(e_loss)

        logs = {
            self.loss_metric.name: self.loss_metric.result(),
            self.hinted_loss_metric.name: self.hinted_loss_metric.result(),
            self.rotation_metric.name: self.rotation_metric.result(),
            self.ratio_metric.name: self.ratio_metric.result(),
            # self.er_loss_metric.name: self.er_loss_metric.result(),
        }

        return logs

    def run(self, lr=0.001, weights=None, ways=5, shots=5, test_shots=15,
            data_dir_path="/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224"):
        if weights is not None:
            self.load_weights(weights, by_name=True, skip_mismatch=True)

        dataloader = DataLoader(data_dir_path=data_dir_path)

        meta_test_ds, meta_test_name_projector = dataloader.get_dataset_V3(phase='test', way_num=ways, shot_num=shots,
                                                                           episode_test_sample_num=test_shots,
                                                                           episode_num=600,
                                                                           batch=4,
                                                                           augment=False)

        meta_train_ds, steps_per_epoch = dataloader.get_all_dataset(phase='train', batch=32,
                                                                    augment=True, contrastive=True)
        total_epoch = 600
        scheduled_lrs = WarmUpStep(
            learning_rate_base=lr,
            warmup_learning_rate=0.0,
            warmup_steps=steps_per_epoch * 3,
        )

        # scheduled_lrs = WarmUpCosine(
        #     learning_rate_base=0.001,
        #     total_steps=total_epoch * steps_per_epoch,
        #     warmup_learning_rate=0.00001,
        #     warmup_steps=steps_per_epoch * 3,
        # )

        # self.compile(tfa.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY, beta_1=0.937))
        self.compile(tf.optimizers.Adam(scheduled_lrs))
        # self.compile(tf.keras.optimizers.SGD(scheduled_lrs, momentum=0.937, nesterov=True))
        self.train_step = self.train_step_normal

        self.test_step = self.test_step_meta
        self.predict_step = self.test_step_meta

        # for data in meta_train_ds:
        #     self.train_step(data)
        # for data in meta_test_ds:
        #     self.test_step(data)
        # self.test_step(data)
        monitor_name = "val_mq_acc"
        monitor_name = "val_mq_acc_base"
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

        self.fit(meta_train_ds, epochs=total_epoch,
                 validation_data=meta_test_ds,
                 callbacks=callbacks, initial_epoch=0)
        # self.fit(meta_train_ds.repeat(), epochs=total_epoch,
        #          steps_per_epoch=steps_per_epoch,
        #          validation_data=meta_test_ds,
        #          callbacks=callbacks, initial_epoch=0)

    def fine_tune(self, lr=0.001, weights=None, ways=5, shots=5, test_shots=15,
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
        train_test_num = 15
        train_batch = 4
        episode_num = 1200
        steps_per_epoch = episode_num // train_batch
        train_epoch = 20
        mix_up = True
        augment = True
        total_epoch = 100
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

        scheduled_lrs = WarmUpCosine(
            learning_rate_base=lr,
            total_steps=total_epoch * steps_per_epoch,
            warmup_learning_rate=0.0,
            warmup_steps=steps_per_epoch * 3,
        )
        self.compile(tf.keras.optimizers.SGD(scheduled_lrs, momentum=0.9, nesterov=True))
        # self.compile(tf.keras.optimizers.Adam(scheduled_lrs))
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
        self.fit(meta_train_ds.repeat(), epochs=total_epoch,
                 steps_per_epoch=steps_per_epoch,
                 validation_data=meta_test_ds,
                 callbacks=callbacks, initial_epoch=0)

    @tf.function
    def get_boundingBox(self, input, thresh=0.5):
        h, w, _ = input.shape
        locations = tf.where(tf.greater(input, thresh))

        def process(locations):
            locations = tf.cast(locations, tf.float32)
            left_x = tf.minimum(tf.reduce_min(locations[..., 1]), w - 2.)
            top_y = tf.minimum(tf.reduce_min(locations[..., 0]), h - 2.)
            right_x = tf.maximum(tf.reduce_max(locations[..., 1]), left_x + 1.)
            bottom_y = tf.maximum(tf.reduce_max(locations[..., 0]), top_y + 1.)
            return tf.stack([left_x, top_y, right_x, bottom_y]) / tf.stack([w - 1., h - 1., w - 1., h - 1.])

        out = tf.cond(tf.shape(locations)[0] == 0, lambda: tf.cast([0., 0., 1., 1.], tf.float32),
                      lambda: process(locations))
        return out

    @tf.function
    def get_croped_images(self, input, out_size=[84, 84]):
        image, box = input
        h, w, _ = image.shape
        left_x, top_y, right_x, bottom_y = tf.unstack(box)
        begin_x = tf.cast(w * left_x, tf.int64)
        begin_y = tf.cast(h * top_y, tf.int64)
        end_x = tf.cast(w * right_x, tf.int64)
        end_y = tf.cast(h * bottom_y, tf.int64)
        croped_image = image[begin_y:end_y, begin_x:end_x, ...]
        croped_image = tf.image.resize(croped_image, out_size)
        return croped_image

    def show(self, weights=None, data_dir_path="/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224",
             phase="test"):
        if weights is not None:
            self.load_weights(weights, by_name=True, skip_mismatch=True)

        dataloader = MiniImageNetDataLoader_v2(data_dir_path=data_dir_path)

        meta_test_ds, name_projector = dataloader.get_dataset(phase=phase, way_num=5, shot_num=5,
                                                              episode_test_sample_num=5,
                                                              episode_num=600,
                                                              augment=False,
                                                              mix_up=False,
                                                              batch=1)

        cv2.namedWindow("image", 1)
        cv2.namedWindow("q_show_image", 1)
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

            training = False

            support_image = tf.reshape(support_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))
            # support_image = tf.map_fn(do_augmentations, images, parallel_iterations=16)
            training = False

            support_features = self.encoder(support_image, training=training)
            _, f_h, f_w, f_c = tf.unstack(tf.shape(support_features))
            support_logits = self.gap(self.last_max_pooling(support_features))
            support_logits = tf.nn.l2_normalize(support_logits, -1)

            support_logits_merge = tf.reshape(support_logits,
                                              [batch, ways * shots, tf.shape(support_logits)[-1]])
            support_logits_merge = tf.nn.l2_normalize(support_logits_merge, -1)
            support_logits_merge = tf.reduce_mean(support_logits_merge, 1)
            support_logits_merge = tf.nn.l2_normalize(support_logits_merge, -1)
            support_logits_merge = tf.reshape(support_logits_merge, [batch, 1, 1, 1, f_c])
            support_logits_broad = tf.broadcast_to(support_logits_merge, [batch, ways * shots, f_h, f_w, f_c])
            support_logits_broad = tf.reshape(support_logits_broad, [-1, f_h, f_w, f_c])
            merge_feature_support = tf.reduce_sum(tf.nn.l2_normalize(support_features, -1) *
                                                  support_logits_broad, -1, keepdims=True) + support_features
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
            _, q_f_h, q_f_w, q_f_c = tf.unstack(tf.shape(query_features))
            query_logits = tf.nn.l2_normalize(query_logits, -1)
            support_logits_broad = tf.broadcast_to(support_logits_merge,
                                                   [batch, ways * query_shots, f_h, q_f_w, f_c])
            support_logits_broad = tf.reshape(support_logits_broad, [-1, f_h, q_f_w, f_c])
            merge_feature_query = tf.reduce_sum(tf.nn.l2_normalize(query_features, -1) *
                                                support_logits_broad, -1, keepdims=True) + query_features

            query_self_attention = self.self_attention_referenced_conv(merge_feature_query, training=training)
            query_features = tf.nn.l2_normalize(query_features, -1)
            query_logits_attention = query_features * query_self_attention
            query_logits_attention = tf.math.divide_no_nan(tf.reduce_sum(query_logits_attention, [1, 2]),
                                                           tf.reduce_sum(query_self_attention, [1, 2]))
            query_logits_attention = tf.nn.l2_normalize(query_logits_attention, -1)
            query_logits = tf.concat([query_logits, query_logits_attention], -1)

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
                                     [batch, ways, query_shots, query_image.shape[-3], *query_image.shape[-2:]])
            query_self_attention = tf.reshape(query_self_attention, [batch, ways, query_shots, f_h, f_w, 1])

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

            origin_q_attention = tf.image.resize(origin_q_attention * 255, q_image.shape[-3:-1],
                                                 method='bilinear').numpy().astype(
                np.uint8)
            origin_q_attention = cv2.applyColorMap(origin_q_attention, cv2.COLORMAP_JET)
            origin_q_attention = cv2.addWeighted(q_image, 0.5, origin_q_attention, 0.5, 0)

            q_show_image = cv2.hconcat([q_image, origin_q_attention])
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

        support_image = tf.reshape(support_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))

        training = False

        support_features = self.encoder(support_image, training=training)
        _, f_h, f_w, f_c = tf.unstack(tf.shape(support_features))
        support_logits = self.gap(self.last_max_pooling(support_features))
        support_logits = tf.nn.l2_normalize(support_logits, -1)

        support_logits_base = tf.reshape(support_logits,
                                         [batch, ways, shots, tf.shape(support_logits)[-1]])
        support_logits_base = tf.nn.l2_normalize(support_logits_base, -1)
        x_mean_base = tf.reduce_mean(support_logits_base, 2)

        new_shape = tf.concat([[-1], tf.shape(query_image)[-3:]], axis=0)
        query_image = tf.reshape(query_image, new_shape)

        query_features = self.encoder(query_image, training=training)
        query_logits = self.gap(self.last_max_pooling(query_features))
        query_logits = tf.nn.l2_normalize(query_logits, -1)

        logits_dim = tf.shape(x_mean_base)[-1]
        dim_shape = tf.shape(query_label)[-1]

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
        logs = {
            self.mean_query_acc_base.name: self.mean_query_acc_base.result(),
        }
        return logs


multi_gpu = True
seed = 100
random.seed(seed)
if multi_gpu is True:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = FSLModel(imageshape=(84, 84, 3), num_class=64)
else:
    model = FSLModel(imageshape=(84, 84, 3), num_class=64)
model.run(weights="/data/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name))
model.fine_tune(weights="/data/giraffe/0_FSL/Normal_ckpts/model_e166-l 0.82573.h5",
                lr=0.005,
                )
# model.run(weights="/data/giraffe/0_FSL/TRRN_2_ckpts/model_e273-l 0.82473.h5")
# model.run(data_dir_path="/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224")
# model.run(data_dir_path="/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224")
# model.run(data_dir_path="/data/giraffe/0_FSL/data/CUB_200_2011/CUB_200_2011/processed_images_224_crop")
# model.run(weights="/data2/giraffe/0_FSL/TRRN_2_ckpts/model_e001-l 0.82144.h5")
# model.run(weights="/data/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name),
#           data_dir_path="/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224")
# model.run(weights="/data2/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name))
# model.test(weights="/data2/giraffe/0_FSL/TRRN_2_ckpts/model_e273-l 0.82473.h5", shots=5)
# model.init("/data/giraffe/0_FSL/TRSN_ckpts/model_e526-l 0.84958.h5", phase="train")
# model.init_and_test("{}.h5".format(model.name), phase="train")
# random.seed(seed)
model.show("/data2/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name), phase="test")

# model.test(weights="/data/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name), shots=5, episode_num=600)
# model.test(weights=None, shots=5, episode_num=600)
# model.test(weights="/data2/giraffe/0_FSL/{}_ckpts//model_e786-l 0.84222.h5".format(model_name), shots=5,
#            episode_num=1000)
# model.test(weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e047-l 0.84964.h5", shots=5)
# model.fine_tune(lr=0.0001, weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e526-l 0.84958.h5")
