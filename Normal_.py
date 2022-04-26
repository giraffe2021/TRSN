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
        self.clc = tf.keras.layers.Dense(self.num_class,
                                         kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                         name="clc")
        self.clc.build([None, feature_dim])

        self.last_max_pooling = tf.keras.layers.MaxPool2D(padding="same", name="last_max_pooling")
        self.last_max_pooling.build([None, feature_size_h, feature_size_w, feature_dim])

        self.build([None, *imageshape])

        self.acc = tf.keras.metrics.CategoricalAccuracy(name="acc")
        self.loss_metric = tf.keras.metrics.Mean(name="clc_loss")
        self.rotation_metric = tf.keras.metrics.Mean(name="rotate_loss")
        self.ratio_metric = tf.keras.metrics.Mean(name="ratio_loss")
        self.contrastive_loss_metric = tf.keras.metrics.Mean(name="contrastive_loss")
        self.intra_metric = tf.keras.metrics.Mean(name="intra_loss")
        self.inter_metric = tf.keras.metrics.Mean(name="inter_loss")

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
        self.contrastive_loss_metric.reset_states()
        self.rotation_metric.reset_states()
        self.ratio_metric.reset_states()
        self.intra_metric.reset_states()
        self.inter_metric.reset_states()

        self.query_loss_metric.reset_states()
        self.mean_query_acc.reset_states()

        self.mean_query_acc_base.reset_states()

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
        return loss_clc

    # @tf.function
    def train_step_normal(self, data):

        x, y = data
        training = True
        with tf.GradientTape() as tape:
            features = self.encoder(x, training=training)
            logits = self.gap(self.last_max_pooling(features))
            pred = self.clc(logits, training=training)
            clc_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y, pred, from_logits=True))
        trainable_vars = self.encoder.trainable_weights + self.clc.trainable_weights
        grads = tape.gradient([clc_loss], trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.loss_metric.update_state(clc_loss)

        logs = {
            self.loss_metric.name: self.loss_metric.result(),
        }

        return logs

    @tf.function
    def reparameterize_batch(self, means, var, sample_num=20):
        means = tf.repeat(means, sample_num, axis=0)
        var = tf.repeat(var, sample_num, axis=0)

        eps = tf.random.normal(shape=tf.shape(means))
        sampled_logits = eps * tf.sqrt(var) + means
        sampled_logits = tf.reshape(sampled_logits, [-1, tf.shape(sampled_logits)[-1]])
        return sampled_logits

    @tf.function
    def get_off_diag(self, c):
        """get_off_diag method.

        Makes the diagonals of the cross correlation matrix zeros.
        This is used in the off-diagonal portion of the loss function,
        where we take the squares of the off-diagonal values and sum them.

        Arguments:
            c: A tf.tensor that represents the cross correlation
              matrix

        Returns:
            Returns a tf.tensor which represents the cross correlation
            matrix with its diagonals as zeros.
        """

        zero_diag = tf.zeros(tf.shape(c)[-1])
        return tf.linalg.set_diag(c, zero_diag)

    def train_step_self_supervised(self, data):
        x, y = data
        image, x_rotated_1, x_rotated_2, x_resized_1, x_resized_2 = x
        y, rotate_label_1, rotate_label_2, resize_label_1, resize_label_2 = y
        training = True
        x = tf.concat([image, x_rotated_1, x_rotated_2, x_resized_1, x_resized_2], 0)
        y_all = tf.concat([y, y, y, y, y], 0)
        batch = tf.shape(image)[0]
        with tf.GradientTape() as tape:
            features = self.encoder(x, training=training)
            logits = self.gap(self.last_max_pooling(features))
            pred = self.clc(logits, training=training)
            clc_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_all, pred, from_logits=True))
            logits_base, logits_rotated_1, logits_rotated_2, logits_resized_1, logits_resized_2 \
                = tf.split(logits, [batch, batch, batch, batch, batch], 0)

            logits = tf.stack([logits_base, logits_rotated_1, logits_rotated_2, logits_resized_1, logits_resized_2], 0)

            proto = tf.reduce_mean(logits, 0, keepdims=True)
            sim = tf.keras.losses.cosine_similarity(logits, proto, -1)
            contrastive_loss = tf.reduce_mean(sim)

        trainable_vars = self.encoder.trainable_weights + self.clc.trainable_weights
        grads = tape.gradient([clc_loss], trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.loss_metric.update_state(clc_loss)
        self.contrastive_loss_metric.update_state(contrastive_loss)

        logs = {
            self.loss_metric.name: self.loss_metric.result(),
            self.contrastive_loss_metric.name: self.contrastive_loss_metric.result(),
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

            support_logits = tf.reshape(support_logits,
                                        [batch, ways, shots, tf.shape(support_logits)[-1]])
            support_mean, support_mean_label = self.random_sample_support(support_logits, support_label)

            new_shape = tf.concat([[-1], tf.shape(query_image)[-3:]], axis=0)
            query_image = tf.reshape(query_image, new_shape)

            query_features = self.encoder(query_image, training=training)
            query_logits = self.gap(self.last_max_pooling(query_features))
            query_logits = tf.nn.l2_normalize(query_logits, -1)

            logits_dim = tf.shape(support_mean)[-1]
            dim_shape = tf.shape(query_label)[-1]

            support_mean = tf.reshape(support_mean, [batch, ways, logits_dim])
            support_mean = tf.nn.l2_normalize(support_mean, -1)

            query_logits = tf.reshape(query_logits,
                                      [batch, -1, tf.shape(query_logits)[-1]])

            query_logits = tf.nn.l2_normalize(query_logits, -1)
            sim = tf.linalg.matmul(query_logits, support_mean, transpose_b=True)

            sim = tf.reshape(sim, [batch, ways, query_shots, -1])
            sim = tf.nn.softmax(sim * 20, -1)
            meta_contrast_loss = tf.keras.losses.categorical_crossentropy(
                tf.reshape(query_label, [-1, tf.shape(query_label)[-1]]),
                tf.reshape(sim, [-1, tf.shape(sim)[-1]]))
            meta_contrast_loss = tf.reduce_mean(meta_contrast_loss)

            avg_loss = meta_contrast_loss

        trainable_vars = self.encoder.trainable_weights
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

        meta_test_ds, meta_test_name_projector = dataloader.get_dataset_V3(phase='test', way_num=ways, shot_num=shots,
                                                                           episode_test_sample_num=test_shots,
                                                                           episode_num=600,
                                                                           batch=4,
                                                                           augment=False)

        contrastive = True
        batch = 64
        total_epoch = 600

        if contrastive is not True:
            batch *= 5
            total_epoch *= 5

        meta_train_ds, steps_per_epoch = dataloader.get_all_dataset(phase='train', batch=batch,
                                                                    augment=True, contrastive=contrastive)
        steps_per_epoch = steps_per_epoch // batch
        scheduled_lrs = WarmUpStep(
            learning_rate_base=lr,
            warmup_learning_rate=0.0,
            warmup_steps=steps_per_epoch * 3,
        )

        self.compile(tf.optimizers.Adam(scheduled_lrs))
        if contrastive is True:
            self.train_step = self.train_step_self_supervised
        else:
            self.train_step = self.train_step_normal

        self.test_step = self.test_step_meta
        self.predict_step = self.test_step_meta

        # for data in meta_train_ds:
        #     self.train_step(data)
        # for data in meta_test_ds:
        #     self.test_step_meta(data)
        # self.test_step(data)
        monitor_name = "val_mq_acc"
        monitor_name = "val_mq_acc_base"
        # monitor_name = "val_query_loss"
        monitor_cmp = "max"
        # monitor_cmp = "min"
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
        train_test_num = 5
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
        sim = tf.linalg.matmul(reshape_query_logits_base, support_mean_base, transpose_b=True)

        sim = tf.reshape(sim, [batch, ways, query_shots, -1])
        intra_dist = tf.math.divide_no_nan(tf.reduce_sum((-sim + 1.) * query_label, -1),
                                           tf.reduce_sum(query_label, -1))
        intra_dist = tf.reduce_mean(intra_dist)
        inter_dist = tf.math.divide_no_nan(tf.reduce_sum((-sim + 1.) * (1. - query_label), -1),
                                           tf.reduce_sum((1. - query_label), -1))
        inter_dist = tf.reduce_mean(inter_dist)

        sim = tf.nn.softmax(sim * 20, -1)

        meta_contrast_loss = tf.keras.losses.categorical_crossentropy(
            tf.reshape(query_label, [-1, tf.shape(query_label)[-1]]),
            tf.reshape(sim, [-1, tf.shape(sim)[-1]]))
        meta_contrast_loss = tf.reduce_mean(meta_contrast_loss)

        acc_base = tf.keras.metrics.categorical_accuracy(tf.reshape(query_label, [batch, -1, dim_shape]),
                                                         tf.reshape(sim, [batch, -1, tf.shape(sim)[-1]]))
        acc_base = tf.reduce_mean(acc_base, -1)
        self.mean_query_acc_base.update_state(acc_base)
        self.query_loss_metric.update_state(meta_contrast_loss)
        self.inter_metric.update_state(inter_dist)
        self.intra_metric.update_state(intra_dist)
        logs = {
            self.mean_query_acc_base.name: self.mean_query_acc_base.result(),
            self.query_loss_metric.name: self.query_loss_metric.result(),
            self.intra_metric.name: self.intra_metric.result(),
            self.inter_metric.name: self.inter_metric.result(),
        }
        return logs

    def test_step_meta_2(self, data):
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
        support_mean_base = tf.nn.l2_normalize(x_mean_base, -1)

        logits_dim = tf.shape(support_mean_base)[-1]
        dim_shape = tf.shape(query_label)[-1]

        x_1_1_features = tf.reshape(support_features, [batch, -1, logits_dim])
        x_3_3_features = tf.nn.avg_pool2d(support_features, (3, 3), 2, padding="VALID")
        x_3_3_features = tf.reshape(x_3_3_features, [batch, -1, logits_dim])
        x_5_5_features = tf.nn.avg_pool2d(support_features, (5, 5), 3, padding="VALID")
        x_5_5_features = tf.reshape(x_5_5_features, [batch, -1, logits_dim])
        features_collections = tf.concat([x_1_1_features, x_3_3_features, x_5_5_features], 1)
        features_collections = tf.nn.l2_normalize(features_collections, -1)

        def get_soft_kmeans_protos(logits, logits_base, protos, shots):
            dist = tf.linalg.matmul(logits, protos, transpose_b=True)
            logits = tf.expand_dims(logits, -2)
            weights = tf.expand_dims(tf.nn.softmax(dist * 20, -1), -1)
            weighted_logits_fusion = logits * weights
            sum_weights = tf.reduce_sum(weights, 1)
            sum_weights = sum_weights + tf.ones_like(sum_weights) * tf.cast(shots, tf.float32)
            protos_sum = tf.reduce_sum(logits_base, 2) + tf.reduce_sum(
                weighted_logits_fusion, 1)
            new_protos = tf.math.divide_no_nan(protos_sum, sum_weights)
            new_protos = tf.nn.l2_normalize(new_protos, -1)

            return new_protos

        new_protos = get_soft_kmeans_protos(features_collections, support_logits_base, support_mean_base, shots)
        new_shape = tf.concat([[-1], tf.shape(query_image)[-3:]], axis=0)
        query_image = tf.reshape(query_image, new_shape)

        query_features = self.encoder(query_image, training=training)
        query_logits = self.gap(self.last_max_pooling(query_features))
        query_logits = tf.nn.l2_normalize(query_logits, -1)

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

    def test(self, weights=None, ways=5, shots=5, episode_num=10000):
        if weights is not None:
            self.load_weights(weights, by_name=True, skip_mismatch=True)

        data_dir_path = "/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224"
        # data_dir_path = "/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224_high_level"
        # data_dir_path = "/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224"

        dataloader = DataLoader(data_dir_path=data_dir_path)

        meta_test_ds, meta_test_name_projector = dataloader.get_dataset_V3(phase='train', way_num=ways, shot_num=shots,
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

    def show_TSNE(self, data, labels, perplexity=30, info=""):
        from sklearn import manifold
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import time
        '''t-SNE'''
        print("start t-SNE...", info)
        t0 = time.time()
        n_components = 2
        tsne = manifold.TSNE(n_components=n_components, init='pca', metric="cosine", random_state=0,
                             perplexity=perplexity)
        data = tf.reshape(data, [-1, tf.shape(data)[-1]])
        data = tsne.fit_transform(data.numpy())
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        t1 = time.time()
        print("t-SNE: %.2g sec" % (t1 - t0))
        plt.figure(figsize=(8, 8))
        plt.title("t-SNE (%.2g sec)" % (t1 - t0))
        label_dict = {v.split("_")[-1]: index for index, v in enumerate(list(set(labels)))}

        colors = list(mcolors.XKCD_COLORS.keys())
        for i in range(data.shape[0]):
            text = labels[i]
            if "P" in text:
                plt.plot(data[i, 0], data[i, 1], 'b*')
            else:
                plt.text(data[i, 0], data[i, 1], text,
                         # color=plt.cm.Set3(label_dict[text]),
                         color=colors[-label_dict[text.split("_")[-1]]],
                         fontdict={'weight': 'bold', 'size': 9})

        plt.xticks([])
        plt.yticks([])
        plt.show()

    def show(self, weights=None, data_dir_path="/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224",
             phase="train"):
        if weights is not None:
            self.load_weights(weights, by_name=True, skip_mismatch=True)

        dataloader = DataLoader(data_dir_path=data_dir_path)

        meta_test_ds, name_projector = dataloader.get_dataset(phase=phase, way_num=5, shot_num=5,
                                                              episode_test_sample_num=15,
                                                              episode_num=600,
                                                              augment=False,
                                                              mix_up=False,
                                                              batch=1)

        self.train_step = self.train_step_normal
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
            training = False

            support_features = self.encoder(support_image, training=training)
            _, f_h, f_w, f_c = tf.unstack(tf.shape(support_features))
            support_logits = self.gap(self.last_max_pooling(support_features))
            support_logits = tf.nn.l2_normalize(support_logits, -1)

            support_logits_mean = tf.reshape(support_logits,
                                             [ways, shots, tf.shape(support_logits)[-1]])
            support_logits_mean = tf.nn.l2_normalize(support_logits_mean, -1)
            support_logits_mean = tf.reduce_mean(support_logits_mean, -2)

            query_image = tf.reshape(query_image, tf.concat([[-1], tf.shape(query_image)[-3:]], 0))

            query_features = self.encoder(query_image, training=training)
            query_logits = self.gap(self.last_max_pooling(query_features))
            query_logits = tf.nn.l2_normalize(query_logits, -1)

            logits = tf.concat([support_logits, query_logits, support_logits_mean], 0)
            s_label = tf.reshape(tf.argmax(support_label, -1), [-1, 1]).numpy()
            q_label = tf.reshape(tf.argmax(query_label, -1), [-1, 1]).numpy()
            p_labels = tf.reshape(tf.argmax(tf.reduce_sum(support_label, -2), -1), [-1, 1]).numpy()
            all_lables = []
            all_lables.extend(["s{}".format(l) for l in s_label])
            all_lables.extend(["q{}".format(l) for l in q_label])
            all_lables.extend(["P{}".format(l) for l in p_labels])

            self.show_TSNE(logits, all_lables)

        contrastive = True
        batch = 32
        if contrastive is False:
            batch *= 5
        meta_train_ds, steps_per_epoch = dataloader.get_all_dataset(phase=phase, batch=batch,
                                                                    augment=False, contrastive=contrastive)
        if contrastive is False:
            for x, y in meta_train_ds:
                y = tf.argmax(y, -1).numpy()
                features = self.encoder(x, training=False)
                logits = self.gap(self.last_max_pooling(features))
                all_lables = []
                all_lables.extend(["_{}".format(l) for l in y])
                self.show_TSNE(logits, all_lables)
        else:
            for x, y in meta_train_ds:
                image, x_rotated_1, x_rotated_2, x_resized_1, x_resized_2 = x
                y, rotate_label_1, rotate_label_2, resize_label_1, resize_label_2 = y
                training = True
                x = tf.concat([image, x_rotated_1, x_rotated_2, x_resized_1, x_resized_2], 0)
                y = tf.argmax(y, -1).numpy()
                features = self.encoder(x, training=training)
                logits = self.gap(self.last_max_pooling(features))
                all_lables = []
                all_lables.extend(["_{}".format(l) for l in y])
                all_lables.extend(["ra_{}".format(l) for l in y])
                all_lables.extend(["rb_{}".format(l) for l in y])
                all_lables.extend(["sa_{}".format(l) for l in y])
                all_lables.extend(["sb_{}".format(l) for l in y])

                logits_base, logits_rotated_1, logits_rotated_2, logits_resized_1, logits_resized_2 \
                    = tf.split(logits, [batch, batch, batch, batch, batch], 0)

                proto = tf.stack([logits_base, logits_rotated_1, logits_rotated_2, logits_resized_1, logits_resized_2],
                                 0)
                means = tf.math.reduce_mean(proto, 0, keepdims=True)
                var = tf.math.reduce_variance(proto, 0, keepdims=True)
                sample_num = 20
                sampled_logits = self.reparameterize_batch(means, var, sample_num)
                sampled_label = tf.repeat(tf.expand_dims(y, 0), sample_num, axis=0)
                sampled_label = tf.reshape(sampled_label, [-1])
                logits = tf.concat([logits, sampled_logits], 0)
                all_lables.extend(["S_{}".format(l) for l in sampled_label])
                self.show_TSNE(logits, all_lables)


multi_gpu = True
seed = 100
random.seed(seed)
if multi_gpu is True:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = FSLModel(imageshape=(84, 84, 3), num_class=64)
else:
    model = FSLModel(imageshape=(84, 84, 3), num_class=64)
# model.test(weights="/data/giraffe/0_FSL/Normal__ckpts/latest.h5", episode_num=600)
model.run(weights=None)
# model.show("/data2/giraffe/0_FSL/Normal_centroid_ckpts/latest.h5")
# model.show("model_e273-l 0.82473.h5")
# model.show("/data2/giraffe/0_FSL/TRSN_2_ckpts/model_e054-l 0.85513.h5")
# model.fine_tune(weights="/data/giraffe/0_FSL/Normal__ckpts/model_e221-l 0.81636.h5", lr=0.005)
# model.run(weights="/data/giraffe/0_FSL/TRRN_2_ckpts/model_e273-l 0.82473.h5")
# model.run(data_dir_path="/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224")
# model.run(data_dir_path="/data/giraffe/0_FSL/data/FC100",
#           weights="/data/giraffe/0_FSL/TRRN_2_ckpts/model_e399-l 0.61740.h5")
# model.run(data_dir_path="/data/giraffe/0_FSL/data/CUB_200_2011/CUB_200_2011/processed_images_224_crop")
# model.run(weights="/data2/giraffe/0_FSL/TRRN_2_ckpts/model_e001-l 0.82144.h5")
# model.run(weights="/data/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name),
#           data_dir_path="/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224")
# model.run(weights="/data2/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name))
# model.test(weights="/data2/giraffe/0_FSL/TRRN_2_ckpts/model_e273-l 0.82473.h5", shots=5)
# model.init("/data/giraffe/0_FSL/TRSN_ckpts/model_e526-l 0.84958.h5", phase="train")
# model.init_and_test("{}.h5".format(model.name), phase="train")
# random.seed(seed)
# model.show("/data2/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name), phase="test")

# model.test(weights="/data/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name), shots=5, episode_num=600)
# model.test(weights=None, shots=5, episode_num=600)
# model.test(weights="/data2/giraffe/0_FSL/{}_ckpts//model_e786-l 0.84222.h5".format(model_name), shots=5,
#            episode_num=1000)
# model.test(weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e047-l 0.84964.h5", shots=5)
# model.fine_tune(lr=0.0001, weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e526-l 0.84958.h5")
