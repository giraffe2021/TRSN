import tensorflow as tf

DATA_FORMAT = 'channels_last'
bn_momentum = 0.9


def ResNet18(include_top=False, weights=None, input_shape=None, layer_params=[2, 2, 2, 2], pooling=None, use_bias=True,
             name="resnet18"):
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=DATA_FORMAT,
                                      require_flatten=include_top,
                                      weights=weights)

    img_input = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = tf.keras.layers.Conv2D(64, (7, 7),
                               strides=(2, 2),
                               padding='valid',
                               kernel_initializer='he_normal',
                               name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(momentum=bn_momentum, name='bn_conv1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    # x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = make_basic_block_layer(x, filter_num=64,
                               blocks=layer_params[0])

    x = make_basic_block_layer(x, filter_num=128,
                               blocks=layer_params[1],
                               stride=2)
    x = make_basic_block_layer(x, filter_num=256,
                               blocks=layer_params[2],
                               stride=2)

    x = make_basic_block_layer(x, filter_num=512,
                               blocks=layer_params[3],
                               stride=2)

    if pooling == 'avg':
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = tf.keras.layers.GlobalMaxPooling2D()(x)

    model = tf.keras.Model(img_input, x, name=name)
    return model


def ResNet12(include_top=False, weights=None, input_shape=None, layer_params=[2, 2, 2, 2], pooling=None,
             use_bias=True, name="resnet12"):
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=DATA_FORMAT,
                                      require_flatten=include_top,
                                      weights=weights)

    img_input = tf.keras.layers.Input(shape=input_shape)

    # x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    # x = tf.keras.layers.Conv2D(64, (7, 7),
    #                            strides=(2, 2),
    #                            padding='valid',
    #                            kernel_initializer='he_normal',
    #                            name='conv1')(x)
    # x = tf.keras.layers.BatchNormalization(momentum=bn_momentum, name='bn_conv1')(x)
    # x = tf.keras.layers.Activation('relu')(x)
    # x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    # x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = tf.nn.space_to_depth(img_input, 2)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(x)
    # x = tf.keras.layers.Conv2D(64, (3, 3),
    #                            strides=(2, 2),
    #                            padding='valid',
    #                            kernel_initializer='he_normal',
    #                            name='conv1')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3),
                               strides=(1, 1),
                               padding='valid',
                               use_bias=use_bias,
                               kernel_initializer='he_normal',
                               name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(momentum=bn_momentum, name='bn_conv1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x1 = make_basic_block_layer(x, filter_num=64,
                                blocks=layer_params[0])

    x2 = make_basic_block_layer(x1, filter_num=128,
                                blocks=layer_params[1],
                                stride=2)
    x3 = make_basic_block_layer(x2, filter_num=256,
                                blocks=layer_params[2],
                                stride=2)

    x4 = make_basic_block_layer(x3, filter_num=512,
                                blocks=layer_params[3],
                                stride=2)
    x = x4

    if pooling == 'avg':
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = tf.keras.layers.GlobalMaxPooling2D()(x)

    model = tf.keras.Model(img_input, x, name=name)
    return model


def make_basic_block_base(inputs, filter_num, stride=1, use_bias=True):
    x = tf.keras.layers.Conv2D(filters=filter_num,
                               kernel_size=(3, 3),
                               strides=stride,
                               use_bias=use_bias,
                               kernel_initializer='he_normal',
                               padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=bn_momentum, )(x)
    x = tf.keras.layers.Conv2D(filters=filter_num,
                               kernel_size=(3, 3),
                               strides=1,
                               use_bias=use_bias,
                               kernel_initializer='he_normal',
                               padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=bn_momentum, )(x)

    shortcut = inputs
    if stride != 1:
        shortcut = tf.keras.layers.Conv2D(filters=filter_num,
                                          kernel_size=(1, 1),
                                          strides=stride,
                                          use_bias=use_bias,
                                          kernel_initializer='he_normal')(inputs)
        shortcut = tf.keras.layers.BatchNormalization(momentum=bn_momentum, )(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x


def make_basic_block_layer(inputs, filter_num, blocks, stride=1, use_bias=True):
    x = make_basic_block_base(inputs, filter_num, stride=stride, use_bias=use_bias)

    for _ in range(1, blocks):
        x = make_basic_block_base(x, filter_num, stride=1)

    return x


def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate a model's input shape.
    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
            If weights='imagenet' input channels must be equal to 3.
    # Returns
        An integer shape tuple (may include None entries).
    # Raises
        ValueError: In case of invalid argument values.
    """
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting `include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                        (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                        (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape

# ResNet12(input_shape=(84,84,3)).summary()
