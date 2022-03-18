import tensorflow as tf
from functools import partial


class DropBlock2D(tf.keras.layers.Layer):
    """See: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format=None,
                 **kwargs):
        """Initialize the layer.
        :param block_size: Size for each mask bloctf.
        :param keep_prob: Probability of keeping the original feature.
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        :param kwargs: Arguments for parent class.
        """
        super(DropBlock2D, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = data_format
        self.supports_masking = True
        self.pooling = tf.keras.layers.MaxPool2D(
            pool_size=(self.block_size, self.block_size),
            padding='same',
            strides=1,
            data_format='channels_last',
        )

    def get_config(self):
        config = {'block_size': self.block_size,
                  'keep_prob': self.keep_prob,
                  'sync_channels': self.sync_channels,
                  'data_format': self.data_format}
        base_config = super(DropBlock2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_gamma(self, height, width, block_size, keep_prob):
        """Get the number of activation units to drop"""
        height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
        block_size = tf.cast(block_size, dtype=tf.float32)
        keep_prob = tf.cast(keep_prob, dtype=tf.float32)
        return ((1.0 - keep_prob) / (block_size ** 2)) * \
               (height * width / ((height - block_size + 1.0) * (width - block_size + 1.0)))

    def _compute_valid_seed_region(self, input, block_size):
        _, height, width, _ = tf.unstack(tf.shape(input))
        positions = tf.concat([
            tf.expand_dims(tf.tile(tf.expand_dims(tf.range(height), axis=1), [1, width]), axis=-1),
            tf.expand_dims(tf.tile(tf.expand_dims(tf.range(width), axis=0), [height, 1]), axis=-1),
        ], axis=-1)
        # tf.print(tf.shape(positions))
        half_block_size = block_size // 2
        valid_seed_region = tf.keras.backend.switch(
            tf.keras.backend.all(
                tf.stack(
                    [
                        positions[:, :, 0] >= half_block_size,
                        positions[:, :, 1] >= half_block_size,
                        positions[:, :, 0] < height - half_block_size,
                        positions[:, :, 1] < width - half_block_size,
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            tf.ones_like(input)[0][..., 0],
            tf.zeros_like(input)[0][..., 0],
        )
        return tf.expand_dims(tf.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, inputs, block_size):
        shape = tf.shape(inputs)
        _, height, width, _ = tf.unstack(shape)
        p = self._get_gamma(height, width, block_size, self.keep_prob)
        mask = tf.keras.backend.random_bernoulli(shape, p=p)
        mask *= self._compute_valid_seed_region(inputs, block_size)
        mask = self.pooling(mask)
        return 1.0 - mask

    def call(self, inputs, training=None):

        if training is None:
            training = tf.keras.backend.learning_phase()
        shape = tf.shape(inputs)
        _, r, w, c = tf.unstack(shape)
        min_size = tf.math.minimum(tf.math.ceil(r / 2), tf.math.ceil(w / 2))
        min_size = tf.cast(min_size, tf.int32)
        block_size = tf.math.minimum(min_size, self.block_size)
        # tf.print(block_size)

        # @tf.function
        def dropped_inputs(block_size):
            outputs = inputs
            if self.data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 2, 3, 1])
            shape = tf.shape(outputs)
            if self.sync_channels:
                mask = self._compute_drop_mask(outputs[..., :1], block_size)
            else:
                mask = self._compute_drop_mask(outputs, block_size)
            outputs = outputs * mask * \
                      (tf.cast(tf.keras.backend.prod(shape), dtype=tf.float32) / tf.reduce_sum(mask))
            if self.data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2])
            return outputs

        output = tf.python.keras.utils.tf_utils.smart_cond(training,
                                                           partial(dropped_inputs, block_size=block_size),
                                                           lambda: tf.python.ops.array_ops.identity(inputs))
        return output
