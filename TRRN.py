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

        self.decoder = Sequential([
            layers.Dense(160, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(320, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(feature_dim, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)),
        ],
            name="proto_attention_referenced_conv")
        self.decoder.build([None, feature_dim])

        self.build([None, *imageshape])
        self.summary()
        self.acc = tf.keras.metrics.CategoricalAccuracy(name="acc")
        self.loss_metric = tf.keras.metrics.Mean(name="clc_loss")
        self.rc_loss_metric = tf.keras.metrics.Mean(name="rc_loss")
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
        self.rc_loss_metric.reset_states()
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
        support, query = data
        support_image, support_label, support_global_label = support
        query_image, query_label, query_global_label = query

        batch = tf.shape(support_image)[0]
        ways = tf.shape(support_image)[1]
        shots = tf.shape(support_image)[2]
        query_shots = tf.shape(query_image)[2]
        with tf.GradientTape() as tape:
            loss_clc, support_logits, query_logits = self.baseline_task_train_step(data, training=True)
            dim_shape = tf.shape(support_logits)[-1]
            protos = tf.reduce_mean(support_logits, -2)
            protos_broad = tf.repeat(tf.reshape(protos, [batch, ways, 1, dim_shape]), ways * shots, -2)
            support_logits_broad = tf.repeat(tf.reshape(support_logits, [batch, 1, ways * shots, dim_shape]), ways, 1)
            protos_broad = tf.nn.l2_normalize(protos_broad, -1)
            support_logits_broad = tf.nn.l2_normalize(support_logits_broad, -1)
            v = protos_broad * support_logits_broad
            weights = (v - tf.reduce_mean(v, -1, keepdims=True)) / (
                    tf.reduce_max(v, -1, keepdims=True) - tf.reduce_mean(v, -1, keepdims=True))
            enrode_protos = weights * support_logits_broad
            enrode_protos = tf.reduce_mean(enrode_protos, -2)
            enrode_protos = tf.reshape(enrode_protos, [batch, 1, ways, dim_shape])
            query_logits = tf.reshape(query_logits, [batch, ways * query_shots, 1, dim_shape])
            x = enrode_protos + query_logits
            x = tf.reshape(x, [-1, dim_shape])
            x_hat = self.decoder(x, training=True)
            x_hat = tf.reshape(x_hat, [batch, ways * query_shots, ways, dim_shape])
            protos_broad_x = tf.repeat(tf.reshape(protos, [batch, 1, ways, dim_shape]), ways * query_shots, 1)

            recon_loss = -1. * tf.keras.losses.mse(protos_broad_x, x_hat)
            # recon_loss = tf.keras.losses.cosine_similarity(protos_broad_x, x_hat)
            query_label = tf.reshape(query_label, [batch, ways * query_shots, -1])
            recon_loss = tf.keras.losses.categorical_crossentropy(query_label, recon_loss, from_logits=True)
            recon_loss = tf.reduce_mean(recon_loss)
            # recon_loss = tf.reduce_mean(tf.reduce_sum(recon_loss * query_label, -1))
        trainable_vars = self.encoder.trainable_weights + self.decoder.trainable_weights + self.clc.trainable_weights
        grads = tape.gradient([recon_loss], trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.loss_metric.update_state(loss_clc)
        self.rc_loss_metric.update_state(recon_loss)

        logs = {
            self.loss_metric.name: self.loss_metric.result(),
            self.rc_loss_metric.name: self.rc_loss_metric.result(),
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
        total_epoch = 200
        scheduled_lrs = WarmUpStep(
            learning_rate_base=lr,
            warmup_learning_rate=0.0,
            warmup_steps=steps_per_epoch * 3,
        )
        # scheduled_lrs = WarmUpCosine(
        #     learning_rate_base=0.01,
        #     total_steps=total_epoch * steps_per_epoch,
        #     warmup_learning_rate=0.001,
        #     warmup_steps=steps_per_epoch * 3,
        # )

        self.compile(tfa.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY, beta_1=0.937))
        # self.compile(tf.optimizers.Adam(scheduled_lrs))
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

        dataloader = DataLoader(data_dir_path=data_dir_path)

        meta_test_ds, name_projector = dataloader.get_dataset_V2(phase=phase, way_num=5, shot_num=5,
                                                                 episode_test_sample_num=15,
                                                                 episode_num=60,
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
            query_label = tf.reshape(query_label, [-1, ways])
            support_label = tf.reshape(support_label, [-1, ways])

            support_image = tf.reshape(support_image, tf.concat([[-1], tf.shape(support_image)[-3:]], 0))
            # support_image = tf.map_fn(do_augmentations, images, parallel_iterations=16)
            training = False
            support_image_origin = support_image
            support_image = tf.image.resize(support_image, [84, 84])
            support_features = self.encoder(support_image, training=training)
            _, f_h, f_w, f_c = tf.unstack(tf.shape(support_features))
            support_logits = self.gap(self.last_max_pooling(support_features))
            support_logits = tf.nn.l2_normalize(support_logits, -1)

            support_logits_base = tf.reshape(support_logits,
                                             [batch, ways, shots, tf.shape(support_logits)[-1]])
            # self_logits_weights = tf.linalg.matmul(support_logits_base, support_logits_base, transpose_b=True)
            # self_logits_weights = 1. - tf.clip_by_value(self_logits_weights, 0., 1.) + tf.eye(shots)
            # self_logits_weights = tf.reduce_sum(self_logits_weights, -1, keepdims=True)
            #
            # support_logits_base = tf.reduce_sum(support_logits_base * self_logits_weights, 2) / tf.reduce_sum(
            #     self_logits_weights, 2)
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
            support_self_attention = self.proto_attention_referenced_conv(merge_feature_support, training=training)
            support_self_attention = tf.reshape(support_self_attention, [-1, ways, f_h, f_w])
            support_self_attention = tf.transpose(support_self_attention, [0, 2, 3, 1])
            # support_self_attention = tf.nn.softmax(support_self_attention * 1., -1)
            #
            # support_self_attention = support_self_attention * tf.reshape(support_label, [-1, 1, 1, dim_shape])
            # support_self_attention = tf.reduce_sum(support_self_attention, -1, keepdims=True)
            #
            # support_self_attention_softmax = tf.nn.softmax(support_self_attention * 1., -1)
            support_self_attention = self.proto_attention_conv(support_self_attention, training=training)
            bbx = tf.map_fn(self.get_boundingBox, support_self_attention,
                            dtype=tf.float32)
            croped_images = tf.map_fn(self.get_croped_images, (support_image_origin, bbx),
                                      dtype=tf.float32)
            support_croped_features = self.encoder(croped_images, training=training)
            support_croped_logits = self.gap(self.last_max_pooling(support_croped_features))
            support_croped_logits = tf.nn.l2_normalize(support_croped_logits, -1)
            sim = tf.linalg.matmul(support_croped_logits, tf.squeeze(support_logits_base, 0), transpose_b=True)
            sim_origin = tf.linalg.matmul(support_logits, tf.squeeze(support_logits_base, 0), transpose_b=True)
            croped_images = (croped_images[..., ::-1] * 255).numpy().astype(np.uint8)
            support_image = (support_image[..., ::-1] * 255).numpy().astype(np.uint8)
            for index, _ in enumerate(croped_images):
                if tf.argmax(sim[index]).numpy() != tf.argmax(support_label[index]).numpy():
                    color = (0, 0, 255)
                else:
                    color = (0, 0, 0)
                croped_images[index, ...] = cv2.putText(croped_images[index, ...],
                                                        "{:.2f} {} {:.2f} ".format(
                                                            sim[index].numpy()[
                                                                tf.argmax(support_label[index]).numpy()],
                                                            tf.argmax(sim[index]).numpy(),
                                                            tf.reduce_max(sim[index]).numpy()),
                                                        (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.35,
                                                        color, 1)

                if tf.argmax(sim_origin[index]).numpy() != tf.argmax(support_label[index]).numpy():
                    color = (0, 0, 255)
                else:
                    color = (0, 0, 0)
                support_image[index, ...] = cv2.putText(support_image[index, ...],
                                                        "{:.2f}  {} {:.2f} ".format(
                                                            sim_origin[index].numpy()[
                                                                tf.argmax(support_label[index]).numpy()],
                                                            tf.argmax(sim_origin[index]).numpy(),
                                                            tf.reduce_max(sim_origin[index]).numpy()),
                                                        (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.35,
                                                        color, 1)
            # support_logits = 0.5 * support_logits + 0.5 * support_croped_logits
            # support_logits_base = tf.reshape(support_logits,
            #                                  [batch, ways, shots, tf.shape(support_logits)[-1]])
            # # support_logits_base = tf.concat([support_logits_base, support_croped_logits], 2)
            # support_logits_base = tf.reduce_mean(support_logits_base, 2)
            # support_logits_base = tf.nn.l2_normalize(support_logits_base, -1)
            # support_logits_mean = tf.reshape(support_logits_base, [batch, 1, ways, 1, 1, f_c])

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
            croped_images = tf.reshape(croped_images,
                                       [batch, ways, shots, croped_images.shape[-3], *croped_images.shape[-2:]])
            support_self_attention = tf.reshape(support_self_attention, [batch, ways, shots, f_h, f_w, 1])

            def transpose_and_reshape(x):
                b, way, s, h, w, c = tf.unstack(tf.shape(x))
                x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
                x = tf.reshape(x, [b, way * h, s * w, c])
                return x

            support_image = transpose_and_reshape(support_image)
            croped_images = transpose_and_reshape(croped_images)
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
            query_image_origin = query_image
            query_image = tf.image.resize(query_image, [84, 84])
            query_features = self.encoder(query_image, training=training)
            query_logits = self.gap(self.last_max_pooling(query_features))
            query_logits = tf.nn.l2_normalize(query_logits, -1)

            support_logits_broad = tf.broadcast_to(support_logits_mean,
                                                   [batch, ways * query_shots, ways, f_h, f_w, f_c])

            query_features = tf.reshape(query_features, [batch, ways * query_shots, 1, f_h, f_w, f_c])
            query_features_broad = tf.broadcast_to(query_features, [batch, ways * query_shots, ways, f_h, f_w, f_c])
            # merge_feature_query = tf.concat(
            #     [tf.nn.l2_normalize(query_features_broad, -1), support_logits_broad], -1)

            merge_feature_query = tf.nn.l2_normalize(query_features_broad, -1) * support_logits_broad
            merge_feature_query = tf.reshape(merge_feature_query,
                                             [-1, f_h, f_w, tf.shape(merge_feature_query)[-1]])
            query_enhanced_mask = self.proto_attention_referenced_conv(merge_feature_query, training=training)
            query_enhanced_mask = tf.reshape(query_enhanced_mask, [-1, ways, f_h, f_w])
            query_enhanced_mask = tf.transpose(query_enhanced_mask, [0, 2, 3, 1])
            query_enhanced_mask_softmax = tf.nn.softmax(query_enhanced_mask * 1., -1)
            query_self_attention = self.proto_attention_conv(query_enhanced_mask, training=training)

            bbx = tf.map_fn(self.get_boundingBox, query_self_attention,
                            dtype=tf.float32)
            croped_images_query = tf.map_fn(self.get_croped_images, (query_image_origin, bbx),
                                            dtype=tf.float32)
            query_croped_features = self.encoder(croped_images_query, training=training)
            query_croped_logits = self.gap(self.last_max_pooling(query_croped_features))
            query_croped_logits = tf.nn.l2_normalize(query_croped_logits, -1)
            sim = tf.linalg.matmul(query_croped_logits, tf.squeeze(support_logits_base, 0), transpose_b=True)
            sim_origin = tf.linalg.matmul(query_logits, tf.squeeze(support_logits_base, 0), transpose_b=True)
            croped_images_query = (croped_images_query[..., ::-1] * 255).numpy().astype(np.uint8)
            query_image = (query_image[..., ::-1] * 255).numpy().astype(np.uint8)
            for index, _ in enumerate(croped_images_query):
                if tf.argmax(sim[index]).numpy() != tf.argmax(query_label[index]).numpy():
                    color = (0, 0, 255)
                else:
                    color = (0, 0, 0)
                croped_images_query[index, ...] = cv2.putText(croped_images_query[index, ...],
                                                              "{:.2f}  {} {:.2f} ".format(
                                                                  sim[index].numpy()[
                                                                      tf.argmax(query_label[index]).numpy()],
                                                                  tf.argmax(sim[index]).numpy(),
                                                                  tf.reduce_max(sim[index]).numpy()),
                                                              (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                                              0.35,
                                                              color, 1)
                if tf.argmax(sim_origin[index]).numpy() != tf.argmax(query_label[index]).numpy():
                    color = (0, 0, 255)
                else:
                    color = (0, 0, 0)
                query_image[index, ...] = cv2.putText(query_image[index, ...],
                                                      "{:.2f}  {} {:.2f} ".format(
                                                          sim_origin[index].numpy()[
                                                              tf.argmax(query_label[index]).numpy()],
                                                          tf.argmax(sim_origin[index]).numpy(),
                                                          tf.reduce_max(sim_origin[index]).numpy()),
                                                      (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                                      0.35,
                                                      color, 1)

            # query_self_attention = tf.concat([query_self_attention, query_enhanced_mask_softmax], -1)

            # query_self_attention = tf.reduce_max(query_self_attention, -1, keepdims=True)

            # query_features = tf.nn.l2_normalize(query_features, -1)
            #
            # query_features = tf.concat([query_features, query_features], -1)
            # _, f_h, f_w, f_c = tf.unstack(tf.shape(query_features))
            #
            # reshaped_query_features = tf.reshape(query_features,
            #                                      [batch, ways, query_shots, 1, f_h, f_w, f_c])
            # reshaped_query_features = tf.broadcast_to(reshaped_query_features,
            #                                           [batch, ways, query_shots, ways, f_h, f_w, f_c])
            # referenced_attention = -tf.losses.cosine_similarity(
            #     tf.broadcast_to(referenced_support_means, tf.shape(reshaped_query_features)),
            #     reshaped_query_features)
            # referenced_attention = tf.clip_by_value(referenced_attention, 0., 1.)
            # # referenced_attention = tf.cast(tf.greater(referenced_attention, 0.3), tf.float32)
            # referenced_attention = tf.expand_dims(referenced_attention, -1)
            # referenced_attention = tf.reshape(referenced_attention, [batch, ways, query_shots, ways * f_h, f_w, 1])
            # referenced_attention = transpose_and_reshape(referenced_attention)

            query_image = tf.reshape(query_image,
                                     [batch, ways, query_shots, query_image.shape[-2], *query_image.shape[-2:]])
            croped_images_query = tf.reshape(croped_images_query,
                                             [batch, ways, query_shots, croped_images_query.shape[-2],
                                              *croped_images_query.shape[-2:]])
            query_self_attention = tf.reshape(query_self_attention, [batch, ways, query_shots, f_h, f_w, -1])

            query_image = transpose_and_reshape(query_image)
            croped_images_query = transpose_and_reshape(croped_images_query)
            query_self_attention = transpose_and_reshape(query_self_attention)

            # for q_image, r_q_attention in \
            #         zip(q_referenced_image,
            #             referenced_attention):
            #     q_image = (q_image[..., ::-1] * 255).numpy().astype(np.uint8)
            #
            #     r_q_attention = tf.image.resize(r_q_attention * 255, q_image.shape[-3:-1],
            #                                     method='bilinear').numpy().astype(
            #         np.uint8)
            #     r_q_attention = cv2.applyColorMap(r_q_attention, cv2.COLORMAP_JET)
            #     r_q_attention = cv2.addWeighted(q_image, 0.5, r_q_attention, 0.5, 0)
            #
            #     q_show_image = cv2.hconcat([q_image, r_q_attention])
            #     cv2.imshow("q_image", q_show_image)
            #     cv2.waitKey(1)

            for image, c_image, origin_s_attention, q_image, c_q_image, origin_q_attention in \
                    zip(support_image,
                        croped_images,
                        support_self_attention,
                        query_image,
                        croped_images_query,
                        query_self_attention):
                image = image.numpy()
                c_image = c_image.numpy()

                origin_s_attention = tf.image.resize(origin_s_attention * 255, image.shape[-3:-1],
                                                     method='bilinear').numpy().astype(
                    np.uint8)
                origin_s_attention = cv2.applyColorMap(origin_s_attention, cv2.COLORMAP_JET)
                origin_s_attention = cv2.addWeighted(image, 0.5, origin_s_attention, 0.5, 0)

                show_image = cv2.hconcat([c_image, image, origin_s_attention])
                # show_image = cv2.hconcat([image, origin_s_attention])
                # show_image = cv2.transpose(show_image)

                cv2.imshow("image", show_image)

                q_image = q_image.numpy()
                c_q_image = c_q_image.numpy()
                origin_q_attention = tf.image.resize(origin_q_attention * 255, q_image.shape[-3:-1],
                                                     method='bilinear')

                origin_q_attention = tf.split(origin_q_attention, (tf.shape(origin_q_attention)[-1].numpy()), -1)
                origin_q_attention = [
                    cv2.addWeighted(q_image, 0.5, cv2.applyColorMap(a.numpy().astype(np.uint8), cv2.COLORMAP_JET), 0.5,
                                    0)
                    for a in origin_q_attention]
                origin_q_attention = tf.stack(origin_q_attention, 1)
                origin_q_attention = tf.reshape(origin_q_attention, [tf.shape(origin_q_attention)[0], -1, 3],
                                                ).numpy().astype(np.uint8)

                q_show_image = cv2.hconcat([q_image, origin_q_attention])
                cv2.imshow("q_show_image", q_show_image)
                cv2.waitKey(0)

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
        query_image = tf.reshape(query_image, tf.concat([[-1], tf.shape(query_image)[-3:]], axis=0))
        training = False

        support_features = self.encoder(support_image, training=training)
        support_logits = self.gap(support_features)
        query_features = self.encoder(query_image, training=training)
        query_logits = self.gap(query_features)

        dim_shape = tf.shape(support_logits)[-1]
        support_logits = tf.reshape(support_logits, [batch, ways, shots, dim_shape])
        query_logits = tf.reshape(query_logits, [batch, ways, query_shots, dim_shape])
        protos = tf.reduce_mean(support_logits, -2)
        protos_broad = tf.repeat(tf.reshape(protos, [batch, ways, 1, dim_shape]), ways * shots, -2)
        support_logits_broad = tf.repeat(tf.reshape(support_logits, [batch, 1, ways * shots, dim_shape]), ways, 1)
        protos_broad = tf.nn.l2_normalize(protos_broad, -1)
        support_logits_broad = tf.nn.l2_normalize(support_logits_broad, -1)
        v = protos_broad * support_logits_broad
        weights = (v - tf.reduce_mean(v, -1, keepdims=True)) / (
                tf.reduce_max(v, -1, keepdims=True) - tf.reduce_mean(v, -1, keepdims=True))
        enrode_protos = weights * support_logits_broad
        enrode_protos = tf.reduce_mean(enrode_protos, -2)
        enrode_protos = tf.reshape(enrode_protos, [batch, 1, ways, dim_shape])
        query_logits = tf.reshape(query_logits, [batch, ways * query_shots, 1, dim_shape])
        x = enrode_protos + query_logits
        x = tf.reshape(x, [-1, dim_shape])
        x_hat = self.decoder(x, training=training)
        x_hat = tf.reshape(x_hat, [batch, ways * query_shots, ways, dim_shape])
        protos_broad_x = tf.repeat(tf.reshape(protos, [batch, 1, ways, dim_shape]), ways * query_shots, 1)

        recon_loss = -1. * tf.keras.losses.mse(protos_broad_x, x_hat)
        query_label = tf.reshape(query_label, [batch, ways * query_shots, -1])

        acc = tf.keras.metrics.categorical_accuracy(
            tf.reshape(query_label, [batch, -1, tf.shape(query_label)[-1]]),
            tf.reshape(recon_loss, [batch, -1, tf.shape(recon_loss)[-1]]))
        acc = tf.reduce_mean(acc, -1)

        logits_dim = tf.shape(protos)[-1]

        support_mean_base = tf.reshape(protos, [batch, ways, logits_dim])
        support_mean_base = tf.nn.l2_normalize(support_mean_base, -1)
        reshape_query_logits_base = tf.reshape(query_logits, [batch, ways * query_shots, logits_dim])
        reshape_query_logits_base = tf.nn.l2_normalize(reshape_query_logits_base, -1)
        dist_base = tf.linalg.matmul(reshape_query_logits_base, support_mean_base, transpose_b=True)

        acc_base = tf.keras.metrics.categorical_accuracy(
            tf.reshape(query_label, [batch, -1, tf.shape(query_label)[-1]]),
            tf.reshape(dist_base, [batch, -1, tf.shape(recon_loss)[-1]]))
        acc_base = tf.reduce_mean(acc_base, -1)
        self.mean_query_acc_base.update_state(acc_base)
        self.mean_query_acc.update_state(acc)

        logs = {
            self.mean_query_acc.name: self.mean_query_acc.result(),
            self.mean_query_acc_base.name: self.mean_query_acc_base.result(),
            "mean_query_acc_current": tf.reduce_mean(acc, -1),
        }
        return logs

    def test(self, weights=None, ways=5, shots=5, episode_num=10000):
        if weights is not None:
            self.load_weights(weights, by_name=True, skip_mismatch=True)

        data_dir_path = "/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_224"
        # data_dir_path = "/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224_high_level"
        # data_dir_path = "/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224"

        dataloader = DataLoader(data_dir_path=data_dir_path)

        meta_test_ds, meta_test_name_projector = dataloader.get_dataset_V3(phase='test', way_num=ways, shot_num=shots,
                                                                           episode_test_sample_num=15,
                                                                           episode_num=episode_num,
                                                                           batch=8,
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
# model.run(weights="/data/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name))
model.run()
# model.run(weights="/data/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name),
#           data_dir_path="/data/giraffe/0_FSL/data/tiered_imagenet_tools/tiered_imagenet_224")
# model.run(weights="/data2/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name))
# model.test(None, shots=5)
# model.init("/data/giraffe/0_FSL/TRSN_ckpts/model_e526-l 0.84958.h5", phase="train")
# model.init_and_test("{}.h5".format(model.name), phase="train")
# random.seed(seed)
model.show("/data2/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name), phase="test")

# model.test(weights="/data/giraffe/0_FSL/{}_ckpts/latest.h5".format(model_name), shots=5, episode_num=600)
model.test(weights=None, shots=5, episode_num=600)
# model.test(weights="/data2/giraffe/0_FSL/{}_ckpts//model_e786-l 0.84222.h5".format(model_name), shots=5,
#            episode_num=1000)
# model.test(weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e047-l 0.84964.h5", shots=5)
# model.fine_tune(lr=0.0001, weights="/data/giraffe/0_FSL/TRSN_ckpts/model_e526-l 0.84958.h5")
