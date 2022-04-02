import tensorflow as tf

try:
    from .DropBlock import DropBlock2D
except:
    from DropBlock import DropBlock2D
# from .DropBlock import DropBlock2D
import math

DATA_FORMAT = 'channels_last'
bn_momentum = 0.9
drop_rate = 0.2
kernel_initializer = 'he_normal'


def make_basic_block(inputs, filter_num, use_bias=True, drop_rate=0., drop_block=False, dropblock_size=1,
                     pooling=False, dilation_rate=(1, 1)):
    shortcut = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=filter_num,
                                                           kernel_size=(1, 1),
                                                           strides=1,
                                                           kernel_initializer=kernel_initializer,
                                                           kernel_regularizer=tf.keras.regularizers.l2(
                                                               0.0005),
                                                           use_bias=use_bias,
                                                           padding="same"),
                                    tf.keras.layers.BatchNormalization(momentum=bn_momentum)]
                                   )(inputs)

    x = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=filter_num,
                                                    kernel_size=(3, 3),
                                                    strides=1,
                                                    kernel_initializer=kernel_initializer,
                                                    kernel_regularizer=tf.keras.regularizers.l2(
                                                        0.0005),
                                                    use_bias=use_bias,
                                                    dilation_rate=dilation_rate,
                                                    padding="same"),
                             tf.keras.layers.BatchNormalization(momentum=bn_momentum),
                             tf.keras.layers.LeakyReLU(0.1),
                             #######
                             tf.keras.layers.Conv2D(filters=filter_num,
                                                    kernel_size=(3, 3),
                                                    strides=1,
                                                    kernel_initializer=kernel_initializer,
                                                    kernel_regularizer=tf.keras.regularizers.l2(
                                                        0.0005),
                                                    use_bias=use_bias,
                                                    dilation_rate=dilation_rate,
                                                    padding="same"),
                             tf.keras.layers.BatchNormalization(momentum=bn_momentum),
                             tf.keras.layers.LeakyReLU(0.1),
                             #######
                             tf.keras.layers.Conv2D(filters=filter_num,
                                                    kernel_size=(3, 3),
                                                    strides=1,
                                                    kernel_initializer=kernel_initializer,
                                                    kernel_regularizer=tf.keras.regularizers.l2(
                                                        0.0005),
                                                    use_bias=use_bias,
                                                    dilation_rate=dilation_rate,
                                                    padding="same"),
                             tf.keras.layers.BatchNormalization(momentum=bn_momentum)
                             #######
                             ])(inputs)

    x = x + shortcut
    x = tf.keras.layers.LeakyReLU(0.1)(x)

    if pooling is True:
        x = tf.keras.layers.MaxPool2D(padding="same")(x)

    if drop_rate > 0.:
        if drop_block is True:
            _, r, w, c = x.shape
            min_size = int(min(math.ceil(r / 2), math.ceil(w / 2)))
            dropblock_size = min(min_size, dropblock_size)
            x = DropBlock2D(block_size=dropblock_size, keep_prob=1. - drop_rate)(x)
        else:
            x = tf.keras.layers.Dropout(drop_rate)(x)
    return x


# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).
def ResNet_12(input_shape=(84, 84, 3), pooling=False, use_bias=False,
              name="resnet12"):
    img_input = tf.keras.layers.Input(shape=input_shape)
    x = img_input
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(x)
    x = make_basic_block(x, use_bias=use_bias, filter_num=64, drop_rate=drop_rate,
                         pooling=True)

    x = make_basic_block(x, use_bias=use_bias, filter_num=160, drop_rate=drop_rate,
                         pooling=True)

    x = make_basic_block(x, use_bias=use_bias, filter_num=320, drop_rate=drop_rate, drop_block=True, dropblock_size=5,
                         pooling=True)

    x = make_basic_block(x, use_bias=use_bias, filter_num=640, drop_rate=drop_rate, drop_block=True, dropblock_size=5,
                         pooling=False)

    if pooling == 'avg':
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = tf.keras.layers.GlobalMaxPooling2D()(x)

    model = tf.keras.Model(img_input, x, name=name)
    return model


#
m = ResNet_12()
m.summary()
