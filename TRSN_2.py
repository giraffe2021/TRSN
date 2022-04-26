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


class CosineDecayRestarts(keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay schedule with restarts.
    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.
    When training a model, it is often useful to lower the learning rate as
    the training progresses. This schedule applies a cosine decay function with
    restarts to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.
    The schedule is a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    The learning rate multiplier first decays
    from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
    restart is performed. Each new warm restart runs for `t_mul` times more
    steps and with `m_mul` times initial learning rate as the new learning rate.
    Example usage:
    ```python
    first_decay_steps = 1000
    lr_decayed_fn = (
      tf.keras.optimizers.schedules.CosineDecayRestarts(
          initial_learning_rate,
          first_decay_steps))
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """

    def __init__(
            self,
            initial_learning_rate,
            first_decay_steps,
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.0,
            name=None):
        """Applies cosine decay with restarts to the learning rate.
        Args:
          initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
            number. The initial learning rate.
          first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python
            number. Number of steps to decay over.
          t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
            Used to derive the number of iterations in the i-th period.
          m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
            Used to derive the initial learning rate of the i-th period.
          alpha: A scalar `float32` or `float64` Tensor or a Python number.
            Minimum learning rate value as a fraction of the initial_learning_rate.
          name: String. Optional name of the operation.  Defaults to 'SGDRDecay'.
        """
        super(CosineDecayRestarts, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self._t_mul = t_mul
        self._m_mul = m_mul
        self.alpha = alpha
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "SGDRDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            first_decay_steps = tf.cast(self.first_decay_steps, dtype)
            alpha = tf.cast(self.alpha, dtype)
            t_mul = tf.cast(self._t_mul, dtype)
            m_mul = tf.cast(self._m_mul, dtype)

            global_step_recomp = tf.cast(step, dtype)
            completed_fraction = global_step_recomp / first_decay_steps

            def compute_step(completed_fraction, geometric=False):
                """Helper for `cond` operation."""
                if geometric:
                    i_restart = tf.floor(
                        tf.math.log(1.0 - completed_fraction * (1.0 - t_mul)) /
                        tf.math.log(t_mul))

                    sum_r = (1.0 - t_mul ** i_restart) / (1.0 - t_mul)
                    completed_fraction = (completed_fraction - sum_r) / t_mul ** i_restart

                else:
                    i_restart = tf.floor(completed_fraction)
                    completed_fraction -= i_restart

                return i_restart, completed_fraction

            i_restart, completed_fraction = tf.cond(
                tf.equal(t_mul, 1.0),
                lambda: compute_step(completed_fraction, geometric=False),
                lambda: compute_step(completed_fraction, geometric=True))

            m_fac = m_mul ** i_restart
            cosine_decayed = 0.5 * m_fac * (1.0 + tf.cos(
                tf.constant(tf.pi, dtype=dtype) * completed_fraction))
            decayed = (1 - alpha) * cosine_decayed + alpha

            return tf.multiply(initial_learning_rate, decayed, name=name)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "alpha": self.alpha,
            "name": self.name
        }


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

        self.last_max_pooling = tf.keras.layers.MaxPool2D(padding="same", name="last_max_pooling")
        self.last_max_pooling.build([None, feature_size_h, feature_size_w, feature_dim])
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="gap")
        self.gap.build([None, feature_size_h, feature_size_w, feature_dim])

        self.clc = tf.keras.layers.Dense(self.num_class,
                                         kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                         name="clc")
        self.clc.build([None, feature_dim])

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

    def call(self, inputs, training=None):
        return None

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

            support_mean_fusion, support_mean_label = self.random_sample_support(support_logits_fusion, support_label)

            new_shape = tf.concat([[-1], tf.shape(query_image)[-3:]], axis=0)
            query_image = tf.reshape(query_image, new_shape)

            query_features = self.encoder(query_image, training=training)
            query_logits = self.gap(self.last_max_pooling(query_features))
            query_logits = tf.nn.l2_normalize(query_logits, -1)
            support_logits_broad = tf.broadcast_to(support_logits_merge, [batch, ways * query_shots, f_h, f_w, f_c])
            support_logits_broad = tf.reshape(support_logits_broad, [-1, f_h, f_w, f_c])
            merge_feature_query = tf.reduce_sum(tf.nn.l2_normalize(query_features, -1) *
                                                support_logits_broad, -1, keepdims=True) + query_features
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

            query_logits_fusion = tf.reshape(query_logits_fusion,
                                             [batch, -1, tf.shape(query_logits_fusion)[-1]])

            query_logits_fusion = tf.nn.l2_normalize(query_logits_fusion, -1)
            sim = tf.linalg.matmul(query_logits_fusion, support_mean_fusion, transpose_b=True)

            sim = tf.reshape(sim, [batch, ways, query_shots, -1])
            sim = tf.nn.softmax(sim * 20, -1)
            meta_contrast_loss = tf.keras.losses.categorical_crossentropy(
                tf.reshape(query_label, [-1, tf.shape(query_label)[-1]]),
                tf.reshape(sim, [-1, tf.shape(sim)[-1]]))
            meta_contrast_loss = tf.reduce_mean(meta_contrast_loss)

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
        x_mean = tf.reduce_mean(support_logits_fusion, 2)

        support_logits_base = tf.reshape(support_logits,
                                         [batch, ways, shots, tf.shape(support_logits)[-1]])
        support_logits_base = tf.nn.l2_normalize(support_logits_base, -1)
        x_mean_base = tf.reduce_mean(support_logits_base, 2)

        support_logits_attention = tf.reshape(support_logits_attention,
                                              [batch, ways, shots, tf.shape(support_logits_attention)[-1]])
        support_logits_attention = tf.nn.l2_normalize(support_logits_attention, -1)
        x_mean_attention = tf.reduce_mean(support_logits_attention, 2)

        new_shape = tf.concat([[-1], tf.shape(query_image)[-3:]], axis=0)
        query_image = tf.reshape(query_image, new_shape)

        query_features = self.encoder(query_image, training=training)
        query_logits = self.gap(self.last_max_pooling(query_features))
        query_logits = tf.nn.l2_normalize(query_logits, -1)
        support_logits_broad = tf.broadcast_to(support_logits_merge, [batch, ways * query_shots, f_h, f_w, f_c])
        support_logits_broad = tf.reshape(support_logits_broad, [-1, f_h, f_w, f_c])
        merge_feature_query = tf.reduce_sum(tf.nn.l2_normalize(query_features, -1) *
                                            support_logits_broad, -1, keepdims=True) + query_features

        query_self_attention = self.self_attention_referenced_conv(merge_feature_query, training=training)
        query_features = tf.nn.l2_normalize(query_features, -1)
        query_logits_attention = query_features * query_self_attention
        query_logits_attention = tf.math.divide_no_nan(tf.reduce_sum(query_logits_attention, [1, 2]),
                                                       tf.reduce_sum(query_self_attention, [1, 2]))
        query_logits_attention = tf.nn.l2_normalize(query_logits_attention, -1)
        query_logits_fusion = tf.concat([query_logits, query_logits_attention], -1)

        logits_dim = tf.shape(support_logits_fusion)[-1]
        dim_shape = tf.shape(query_label)[-1]

        support_mean = tf.reshape(x_mean, [batch, ways, logits_dim])
        support_mean = tf.nn.l2_normalize(support_mean, -1)
        reshape_query_logits = tf.reshape(query_logits_fusion, [batch, ways * query_shots, logits_dim])
        reshape_query_logits = tf.nn.l2_normalize(reshape_query_logits, -1)
        dist = tf.linalg.matmul(reshape_query_logits, support_mean, transpose_b=True)
        query_pred = tf.clip_by_value(dist, 0., 1.)

        loss = tf.losses.binary_crossentropy(tf.reshape(query_label, [batch, -1, dim_shape, 1]),
                                             tf.expand_dims(query_pred, -1))
        avg_loss = tf.reduce_mean(loss)

        acc = tf.keras.metrics.categorical_accuracy(tf.reshape(query_label, [batch, -1, dim_shape]), dist)
        acc = tf.reduce_mean(acc, -1)
        self.query_loss_metric.update_state(avg_loss)
        self.mean_query_acc.update_state(acc)

        logits_dim = tf.shape(x_mean_attention)[-1]

        support_mean_attention = tf.reshape(x_mean_attention, [batch, ways, logits_dim])
        support_mean_attention = tf.nn.l2_normalize(support_mean_attention, -1)
        reshape_query_logits_attention = tf.reshape(query_logits_attention, [batch, ways * query_shots, logits_dim])
        reshape_query_logits_attention = tf.nn.l2_normalize(reshape_query_logits_attention, -1)
        dist_attention = tf.linalg.matmul(reshape_query_logits_attention, support_mean_attention, transpose_b=True)
        query_pred_attention = tf.clip_by_value(dist_attention, 0., 1.)

        loss_attention = tf.losses.binary_crossentropy(tf.reshape(query_label, [batch, -1, dim_shape, 1]),
                                                       tf.expand_dims(query_pred_attention, -1))
        loss_attention = tf.reduce_mean(loss_attention)

        acc_attention = tf.keras.metrics.categorical_accuracy(tf.reshape(query_label, [batch, -1, dim_shape]),
                                                              dist_attention)
        acc_attention = tf.reduce_mean(acc_attention, -1)
        self.mean_query_acc_attention.update_state(acc_attention)
        self.query_loss_metric_attenion.update_state(loss_attention)

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

        logs = {
            self.mean_query_acc.name: self.mean_query_acc.result(),
            self.mean_query_acc_attention.name: self.mean_query_acc_attention.result(),
            self.mean_query_acc_base.name: self.mean_query_acc_base.result(),
            self.query_loss_metric.name: self.query_loss_metric.result(),
            self.query_loss_metric_attenion.name: self.query_loss_metric_attenion.result(),
            self.query_loss_metric_base.name: self.query_loss_metric_base.result(),
            "mean_query_acc_current": tf.reduce_mean(acc, -1),
        }
        return logs

    def test(self, weights=None, ways=5, shots=5, episode_num=10000,
             data_dir_path="/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224"):
        if weights is not None:
            self.load_weights(weights, by_name=True, skip_mismatch=True)

        # data_dir_path = "/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224"

        dataloader = MiniImageNetDataLoader_v2(data_dir_path=data_dir_path)

        meta_test_ds, meta_test_name_projector = dataloader.get_dataset(phase='test', way_num=ways, shot_num=shots,
                                                                        episode_test_sample_num=15,
                                                                        episode_num=episode_num,
                                                                        batch=8,
                                                                        augment=False)

        self.compile(tf.keras.optimizers.Adam(0.0001))
        self.train_step = self.train_step_meta

        self.test_step = self.test_step_meta
        self.predict_step = self.test_step_meta

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
# model.run(weights="/data/giraffe/0_FSL/TRSN__ckpts/model_e026-l 0.85133.h5")
# model.fine_tune(lr=0.005)
# model.run(weights="model_e238-l 0.86049.h5",
#           data_dir_path="/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224")

model.fine_tune(weights="/data/giraffe/0_FSL/Normal_npairs_3_ckpts/model_e253-l 0.82962.h5",
                lr=0.005,
                )

# model.run(weights="model_e335-l 0.86331.h5",
#           data_dir_path="/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224")
# model.fine_tune(lr=0.005, data_dir_path="/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224")
# model.fine_tune(weights="model_e335-l 0.86331.h5",
#                 lr=0.005,
#                 data_dir_path="/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224")

# model.fine_tune(weights="/data/giraffe/0_FSL/TRRN_2_ckpts/model_e405-l 0.62236.h5",
#                 lr=0.005,
#                 data_dir_path="/data/giraffe/0_FSL/data/FC100")

# model.fine_tune(weights="/data/giraffe/0_FSL/TRRN_2_ckpts/model_e187-l 0.90229.h5",
#                 lr=0.005,
#                 data_dir_path="/data/giraffe/0_FSL/data/CUB_200_2011/CUB_200_2011/processed_images_224_crop")

# model.run(weights="model_e052-l 0.84416.h5",
#           data_dir_path="/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224")
# model.fine_tune(weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e015-l 0.83822.h55")
# model.test(weights="/data2/giraffe/0_FSL/TRSN_ckpts/model_e030-l 0.85144.h5")
# model.test(weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e009-l 0.86502.h5",
#            data_dir_path="/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224",shots=1)
# model.test(weights="/data2/giraffe/0_FSL/TRSN_ckpts/model_e081-l 0.85042.h5", shots=1)
# model.test(weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e023-l 0.85064.h5", shots=1)
# model.test(weights="/data/giraffe/0_FSL/TRRN_2_ckpts/model_e015-l 0.83822.h5")
# model.show("model_e112-l 0.85147.h5")
# model.show("/data2/giraffe/0_FSL/TRSN_ckpts/model_e078-l 0.93898.h5",
#            data_dir_path="/data/giraffe/0_FSL/data/CUB_200_2011/CUB_200_2011/processed_images_224_crop")
# model.show("/data2/giraffe/0_FSL/TRSN_2_ckpts/model_e005-l 0.83702.h5")
# model.test(weights="/data/giraffe/0_FSL/TRSN_2_ckpts/model_e025-l 0.85282.h5", shots=1)
# model.test(weights="/data/giraffe/0_FSL/TRSN_2_ckpts/model_e025-l 0.85282.h5", shots=5)
# model.test(weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e491-l 0.84962.h5", shots=5)
# model.fine_tune(lr=0.0001, weights=""/data/giraffe/0_FSL/TRSN_ckpts/model_e328-l 0.83613.h5"")
