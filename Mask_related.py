import tqdm

from tools.augmentations import *
from functools import partial
import os
import datetime
import random
import cv2
import numpy as np, scipy.stats as st
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


class MRN(tf.keras.Model):
    def __init__(self, imageshape=(84, 84, 3), num_class=64, name="MRN"):
        super(MRN, self).__init__(name=name)

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

        self.mrn = Sequential([
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
            layers.Dense(feature_dim),
            layers.BatchNormalization(),
            layers.Activation("sigmoid")], name="mrn")
        self.mrn.build([None, feature_dim * 2])

        self.acc = tf.keras.metrics.CategoricalAccuracy(name="acc")
        self.loss_metric = tf.keras.metrics.Mean(name="clc_loss")
        self.entropy_loss_metric = tf.keras.metrics.Mean(name="ep_loss")
        self.cluster_loss_metric = tf.keras.metrics.Mean(name="cluster_loss")

        self.query_loss_metric = tf.keras.metrics.Mean("query_loss")
        self.mean_query_acc = tf.keras.metrics.Mean(name="mean_query_acc")

        self.mean_query_acc_base = tf.keras.metrics.Mean(name="mean_query_acc_base")

        self.build([None, *imageshape])
        self.summary()

    def call(self, inputs, training=None):
        return None

    def compile(self, optimizer, **kwargs):
        super(MRN, self).compile(**kwargs)
        self.optimizer = optimizer

    def reset_metrics(self):
        # Resets the state of all the metrics in the model.
        for m in self.metrics:
            m.reset_states()

        self.acc.reset_states()
        self.loss_metric.reset_states()
        self.entropy_loss_metric.reset_states()
        self.cluster_loss_metric.reset_states()

        self.query_loss_metric.reset_states()
        self.mean_query_acc.reset_states()

        self.mean_query_acc_base.reset_states()

    def baseline_task_train_step(self, data, training=True):
        support, query = data
        support_image, _, support_global_label = support
        query_image, _, query_global_label = query

        image_shape = tf.unstack(tf.shape(support_image)[-3:])
        global_dim_shape = tf.shape(support_global_label)[-1]

        support_x = tf.reshape(support_image, [-1, *image_shape])
        query_x = tf.reshape(query_image, [-1, *image_shape])

        support_global_label = tf.reshape(support_global_label, [-1, global_dim_shape])
        query_global_label = tf.reshape(query_global_label, [-1, global_dim_shape])

        x = tf.concat([support_x, query_x], 0)
        y = tf.concat([support_global_label, query_global_label], 0)
        features = self.encoder(x, training=training)
        logits = self.gap(features)
        logits = self.lpn(logits)
        pred = self.clc(logits)
        loss_clc = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y, pred, from_logits=True))

        return loss_clc

    def meta_train_step(self, data, tape, training=True):
        support, query = data
        support_image, support_label, support_global_label = support
        query_image, query_label, query_global_label = query

        image_shape = tf.unstack(tf.shape(support_image)[-3:])
        dim_shape = tf.shape(support_label)[-1]
        global_dim_shape = tf.shape(support_global_label)[-1]

        batch = tf.shape(support_image)[0]
        ways = tf.shape(support_image)[1]
        shots = tf.shape(support_image)[2]
        query_shots = tf.shape(query_image)[2]

        support_global_label = tf.reshape(support_global_label, [-1, global_dim_shape])
        query_global_label = tf.reshape(query_global_label, [-1, global_dim_shape])
        y = tf.concat([support_global_label, query_global_label], 0)
        support_x = tf.reshape(support_image, [-1, *image_shape])
        query_x = tf.reshape(query_image, [-1, *image_shape])

        support_logits = self.gap(self.encoder(support_x, training=training))
        query_logits = self.gap(self.encoder(query_x, training=training))

        _, logits_dim = tf.unstack(tf.shape(support_logits))

        # the clc loss begin
        logits = tf.concat([support_logits, query_logits], 0)
        logits = tf.nn.l2_normalize(logits, -1)
        pred = self.clc(logits)
        loss_clc = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y, pred, from_logits=True))
        # the clc loss end
        support_logits = tf.reshape(support_logits,
                                    [batch, ways, shots, tf.shape(support_logits)[-1]])

        support_logits = tf.nn.l2_normalize(support_logits, -1)
        support_logits_base, support_label = self.random_sample_support(support_logits, support_label)
        support_mean_base = tf.nn.l2_normalize(support_logits_base, -1)
        query_logits = tf.nn.l2_normalize(query_logits, -1)
        query_logits = tf.reshape(query_logits, [batch, ways * query_shots, logits_dim])
        sim = tf.linalg.matmul(query_logits, support_mean_base, transpose_b=True)
        sim = tf.reshape(sim, [batch, ways, query_shots, -1])
        sim = tf.nn.softmax(sim * 20, -1)
        entropy = -1. * sim * tf.math.log(sim + tf.keras.backend.epsilon())
        entropy = tf.reduce_sum(entropy, -1)

        broad_support_logits = tf.reshape(support_mean_base, [batch, 1, ways, logits_dim])
        broad_support_logits = tf.broadcast_to(broad_support_logits, [batch, ways * query_shots, ways, logits_dim])
        broad_query_logits = tf.reshape(query_logits, [batch, ways * query_shots, 1, logits_dim])
        broad_query_logits = tf.broadcast_to(broad_query_logits, [batch, ways * query_shots, ways, logits_dim])
        logits_pairs = tf.concat([broad_query_logits, broad_support_logits], -1)
        logits_pairs = tf.reshape(logits_pairs, [-1, logits_dim * 2])

        logits_mask = self.mrn(logits_pairs, training=training)
        logits_mask = tf.reshape(logits_mask, [batch, ways * query_shots, ways, logits_dim])
        masked_logits = logits_mask * broad_query_logits
        masked_logits = tf.nn.l2_normalize(masked_logits, -1)

        masked_pred = self.clc(tf.reshape(masked_logits, [-1, logits_dim]))
        masked_label = tf.reshape(query_global_label, [batch, ways * query_shots, 1, global_dim_shape])
        # 相同类别变得 熵最大，不同类别 熵不变
        query_label = tf.reshape(query_label, [batch, ways * query_shots, dim_shape, 1])
        masked_label = masked_label * (1. - query_label) \
                       + tf.ones_like(masked_label) / tf.cast(global_dim_shape, tf.float32) * query_label
        masked_label = tf.reshape(masked_label, [-1, global_dim_shape])
        masked_pred_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(masked_label, masked_pred, from_logits=True))
        masked_logits = tf.reshape(masked_logits, [batch, ways * query_shots * ways, logits_dim])

        masked_sim = tf.linalg.matmul(masked_logits, support_mean_base, transpose_b=True)
        masked_sim = tf.reshape(masked_sim, [batch, ways, query_shots, ways, -1])
        masked_sim = tf.nn.softmax(masked_sim * 20., -1)
        masked_entropy = -1. * masked_sim * tf.math.log(masked_sim + tf.keras.backend.epsilon())
        masked_entropy = tf.reduce_sum(masked_entropy, -1)

        entropy = tf.reshape(entropy, [batch, ways, query_shots, 1])
        # the valued of entropy_diff from the same category should be bigger. 
        entropy_diff = tf.abs(masked_entropy - entropy)
        entropy_diff = tf.nn.softmax(entropy_diff * 10., -1)
        meta_entropy_loss = tf.keras.losses.categorical_crossentropy(
            tf.reshape(query_label, [-1, dim_shape]),
            tf.reshape(entropy_diff, [-1, dim_shape]))

        meta_entropy_loss = tf.reduce_mean(meta_entropy_loss)

        return loss_clc, meta_entropy_loss, masked_pred_loss

    # @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            clc_loss, cluster_loss, masked_pred_loss = self.meta_train_step(data, tape)
        trainable_vars = self.encoder.trainable_weights \
                         + self.clc.trainable_weights \
                         + self.mrn.trainable_weights
        grads = tape.gradient([clc_loss, cluster_loss, masked_pred_loss], trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        self.loss_metric.update_state(clc_loss)
        self.entropy_loss_metric.update_state(masked_pred_loss)
        self.cluster_loss_metric.update_state(cluster_loss)

        logs = {
            self.loss_metric.name: self.loss_metric.result(),
            self.entropy_loss_metric.name: self.entropy_loss_metric.result(),
            self.cluster_loss_metric.name: self.cluster_loss_metric.result()
        }

        return logs

    # @tf.function
    def test_step(self, data):
        support, query = data
        support_image, support_label, _ = support
        query_image, query_label, _ = query

        dim_shape = tf.shape(query_label)[-1]

        batch = tf.shape(support_image)[0]
        ways = tf.shape(support_image)[1]
        shots = tf.shape(support_image)[2]
        query_shots = tf.shape(query_image)[2]

        image_shape = tf.unstack(tf.shape(support_image)[-3:])
        support_x = tf.reshape(support_image, [-1, *image_shape])
        query_x = tf.reshape(query_image, [-1, *image_shape])
        training = False

        support_features = self.encoder(support_x, training=training)
        _, f_h, f_w, f_c = tf.unstack(tf.shape(support_features))
        support_logits = self.gap(support_features)
        support_logits = tf.nn.l2_normalize(support_logits, -1)
        support_logits_base = tf.reshape(support_logits,
                                         [batch, ways, shots, tf.shape(support_logits)[-1]])
        support_logits_base = tf.nn.l2_normalize(support_logits_base, -1)
        x_mean_base = tf.reduce_mean(support_logits_base, 2)

        query_features = self.encoder(query_x, training=training)
        query_logits = self.gap(query_features)
        query_logits = tf.nn.l2_normalize(query_logits, -1)

        logits_dim = tf.shape(x_mean_base)[-1]

        support_mean_base = tf.reshape(x_mean_base, [batch, ways, logits_dim])
        support_mean_base = tf.nn.l2_normalize(support_mean_base, -1)
        reshape_query_logits_base = tf.reshape(query_logits, [batch, ways * query_shots, logits_dim])
        reshape_query_logits_base = tf.nn.l2_normalize(reshape_query_logits_base, -1)
        sim = tf.linalg.matmul(reshape_query_logits_base, support_mean_base, transpose_b=True)
        sim = tf.nn.softmax(sim * 20., -1)
        acc_base = tf.keras.metrics.categorical_accuracy(tf.reshape(query_label, [batch, -1, dim_shape]),
                                                         sim)
        acc_base = tf.reduce_mean(acc_base, -1)
        self.mean_query_acc_base.update_state(acc_base)

        entropy = -1. * sim * tf.math.log(sim + tf.keras.backend.epsilon())
        entropy = tf.reduce_sum(entropy, -1)

        broad_support_logits = tf.reshape(support_mean_base, [batch, 1, ways, logits_dim])
        broad_support_logits = tf.broadcast_to(broad_support_logits, [batch, ways * query_shots, ways, logits_dim])
        broad_query_logits = tf.reshape(query_logits, [batch, ways * query_shots, 1, logits_dim])
        broad_query_logits = tf.broadcast_to(broad_query_logits, [batch, ways * query_shots, ways, logits_dim])
        logits_pairs = tf.concat([broad_query_logits, broad_support_logits], -1)
        logits_pairs = tf.reshape(logits_pairs, [-1, logits_dim * 2])

        logits_mask = self.mrn(logits_pairs, training=training)
        logits_mask = tf.reshape(logits_mask, [batch, ways * query_shots, ways, logits_dim])
        masked_logits = logits_mask * broad_query_logits
        masked_logits = tf.nn.l2_normalize(masked_logits, -1)
        masked_logits = tf.reshape(masked_logits, [batch, ways * query_shots * ways, logits_dim])
        masked_sim = tf.linalg.matmul(masked_logits, support_mean_base, transpose_b=True)
        masked_sim = tf.reshape(masked_sim, [batch, ways, query_shots, ways, -1])
        masked_sim = tf.nn.softmax(masked_sim * 20., -1)
        masked_entropy = -1. * masked_sim * tf.math.log(masked_sim + tf.keras.backend.epsilon())
        masked_entropy = tf.reduce_sum(masked_entropy, -1)
        entropy = tf.reshape(entropy, [batch, ways, query_shots, 1])
        # the valued of entropy_diff from the same category should be bigger.
        entropy_diff = tf.abs(masked_entropy - entropy)
        entropy_diff = tf.nn.softmax(entropy_diff * 10., -1)
        acc = tf.keras.metrics.categorical_accuracy(tf.reshape(query_label, [batch, -1, dim_shape]),
                                                    tf.reshape(entropy_diff, [batch, -1, ways]))

        acc = tf.reduce_mean(acc, -1)
        self.mean_query_acc.update_state(acc)
        ########
        logs = {
            self.mean_query_acc.name: self.mean_query_acc.result(),
            self.mean_query_acc_base.name: self.mean_query_acc_base.result(),
            # "mean_query_acc_current": tf.reduce_mean(acc, -1),
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
        train_test_num = 4
        train_batch = 8
        episode_num = 1200
        steps_per_epoch = episode_num // train_batch
        train_epoch = 2
        mix_up = True
        meta_train_ds, meta_train_name_projector = dataloader.get_dataset_V2(phase='train', way_num=ways,
                                                                             shot_num=shots,
                                                                             episode_test_sample_num=train_test_num,
                                                                             episode_num=episode_num,
                                                                             batch=train_batch,
                                                                             augment=True,
                                                                             mix_up=mix_up,
                                                                             epochs=train_epoch)
        scheduled_lrs = WarmUpStep(
            learning_rate_base=lr,
            warmup_learning_rate=0.0,
            warmup_steps=steps_per_epoch * 3,
        )

        self.compile(tf.keras.optimizers.Adam(scheduled_lrs))

        # for data in meta_train_ds:
        #     self.train_step(data)
        # for data in meta_test_ds:
        #     self.test_step(data)
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
        out = self.evaluate(meta_test_ds, return_dict=True)
        self.fit(meta_train_ds.repeat(), epochs=1000,
                 steps_per_epoch=steps_per_epoch,
                 validation_data=meta_test_ds,
                 callbacks=callbacks, initial_epoch=0)

    def test(self, weights=None, ways=5, shots=5, episode_num=10000,
             data_dir_path="/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224"):
        if weights is not None:
            self.load_weights(weights, by_name=True, skip_mismatch=True)

        dataloader = DataLoader(data_dir_path=data_dir_path)

        meta_test_ds, meta_test_name_projector = dataloader.get_dataset(phase='test', way_num=ways, shot_num=shots,
                                                                        episode_test_sample_num=15,
                                                                        episode_num=episode_num,
                                                                        batch=4,
                                                                        augment=False)

        self.compile(tf.keras.optimizers.Adam(0.0001))
        for data in meta_test_ds:
            self.test_step(data)

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


seed = 100
random.seed(seed)
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = MRN(imageshape=(84, 84, 3), num_class=64, name="MRN")
model.run(weights=None)
# model.test(weights="/data2/giraffe/0_FSL/MRN_ckpts/latest.h5", shots=5)
# model.init("/data/giraffe/0_FSL/TRSN_ckpts/model_e526-l 0.84958.h5", phase="train")
# model.init_and_test("{}.h5".format(model.name), phase="train")
# model.show("/data/giraffe/0_FSL/TRSN_ckpts/model_e526-l 0.84958.h5", phase="test")
# model.test(weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e047-l 0.84964.h5", shots=1)
# model.test(weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e047-l 0.84964.h5", shots=5)
# model.fine_tune(lr=0.0001, weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e526-l 0.84958.h5")
