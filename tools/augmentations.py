import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
import tensorflow_addons as tfa
import tensorflow.keras as keras
from functools import partial
import random
import cv2
import numpy as np


@tf.function
def color_jitter(x, strength=[0.4, 0.4, 0.4, 0.1]):
    x = tf.image.random_brightness(x, max_delta=0.8 * 0.4)
    x = tf.image.random_contrast(
        x, lower=1 - 0.8 * strength[1], upper=1 + 0.8 * 0.4
    )
    x = tf.image.random_saturation(
        x, lower=1 - 0.8 * strength[2], upper=1 + 0.8 * 0.4
    )
    x = tf.image.random_hue(x, max_delta=0.2 * strength[3])
    # Affine transformations can disturb the natural range of
    # RGB images, hence this is needed.
    x = tf.clip_by_value(x, 0, 255)
    return x


@tf.function
def random_augment(x, p=0.8):
    @tf.function
    def random_augment_(x):
        process_seq = list()
        # process_seq.append(partial(tf.image.random_brightness, max_delta=0.1))
        # process_seq.append(partial(tf.image.random_contrast, lower=0.8, upper=1.2))
        process_seq.append(partial(tf.image.random_hue, max_delta=0.3))
        process_seq.append(partial(tf.image.random_saturation, lower=0.5, upper=1.5))
        # process_seq.append(channel_shuffle)
        process_seq.append(color_drop)
        random.shuffle(process_seq)
        # process_seq.append(partial(tf.image.random_jpeg_quality, min_jpeg_quality=70, max_jpeg_quality=100))

        for process_func in process_seq:
            x = process_func(x)
            x = tf.clip_by_value(x, 0, 255)
        return x

    p = tf.cast(p * 100, tf.int64)
    chance = tf.random.uniform([], 0, 100, dtype=tf.int64)
    x = tf.cond(chance < tf.cast(p, chance.dtype),
                lambda: random_augment_(x),
                lambda: x)
    # lambda: tf.image.random_jpeg_quality(x, min_jpeg_quality=90, max_jpeg_quality=100))

    return x


@tf.function
def ratio_process(x, ratio_min=0.25, ratio_max=4, min_size=16, max_size=1024):
    origin_size = tf.shape(x)
    # max_len = tf.reduce_max(origin_size)
    # min_len = tf.reduce_min(origin_size)
    # ratio_max = tf.cast(tf.minimum(max_size / max_len, ratio_max), tf.float32)
    # ratio_min = tf.cast(tf.maximum(min_size / min_len, ratio_min), tf.float32)
    ratio = tf.random.uniform([], ratio_min, ratio_max, dtype=tf.float32)
    new_shape = tf.cast(tf.cast(origin_size, dtype=ratio.dtype) * ratio, dtype=origin_size.dtype)[:2]
    origin_type = x.dtype
    x = tf.image.resize(x, size=new_shape, method='bicubic')
    x = tf.cond(ratio > tf.cast(1, ratio.dtype),
                lambda: tf.image.random_crop(x, origin_size),
                lambda: tf.image.resize_with_crop_or_pad(
                    tf.tile(x, tf.cast([tf.math.ceil(1 / ratio_min), tf.math.ceil(1 / ratio_min), 1],
                                       dtype=origin_size.dtype)),
                    origin_size[0], origin_size[1]))

    ratio = (ratio - ratio_min) / (ratio_max - ratio_min)
    x = tf.cast(x, origin_type)
    return x, ratio


@tf.function
def size_ratio_process_v2(x, sample_class=2):
    ratio_min, ratio_max = 0.25, 4

    regin_size = sample_class * 2
    step = (ratio_max - ratio_min) / regin_size
    sample_range = tf.range(ratio_min, ratio_max, step)
    regin_id = tf.random.uniform([], 0, sample_class, dtype=tf.int64)
    ratio = tf.random.uniform([], sample_range[regin_id * 2], sample_range[regin_id * 2 + 1], dtype=tf.float32)

    origin_size = tf.shape(x)
    new_shape = tf.cast(tf.cast(origin_size, dtype=ratio.dtype) * ratio, dtype=origin_size.dtype)[:2]

    x = tf.image.resize(x, size=new_shape, method='bicubic')

    x = tf.cond(ratio > tf.cast(1, ratio.dtype),
                lambda: tf.image.random_crop(x, origin_size),
                lambda: tf.image.crop_to_bounding_box(
                    tf.tile(x, tf.cast([1 / ratio_min, 1 / ratio_min, 1], dtype=origin_size.dtype)),
                    0, 0,
                    origin_size[0], origin_size[1]))

    label = tf.one_hot(regin_id, sample_class)
    x = random_augment(x)
    return x, label


@tf.function
def rotation_process(x):
    origin_size = tf.shape(x)[:2]
    angel = tf.random.uniform([], 0, 3.141592653589793 * 2, dtype=tf.float32)
    x = tf.tile(x, tf.cast([3, 3, 1], dtype=origin_size.dtype))
    x = tfa.image.rotate(x, angel, "BILINEAR")
    x = tf.image.central_crop(x, 1 / 3)
    x = tf.image.resize(x, origin_size)
    return x, angel / (3.141592653589793 * 2)


@tf.function
def rotation_process_with_limit(x, theta=360):
    theta_range = theta / 180. * 3.141592653589793
    origin_size = tf.shape(x)[:2]
    angel = tf.random.uniform([], -theta_range, theta_range, dtype=tf.float32)
    x = tf.tile(x, tf.cast([3, 3, 1], dtype=origin_size.dtype))
    x = tfa.image.rotate(x, angel, "BILINEAR")
    x = tf.image.central_crop(x, 1 / 3)
    x = tf.image.resize(x, origin_size)
    return x, angel / (3.141592653589793 * 2)


@tf.function
def rotation_process_v2(x, sample_class=2):
    origin_size = tf.shape(x)[:2]
    regin_size = sample_class * 2
    step = 3.141592653589793 * 2 / regin_size
    sample_range = tf.range(-step, -step + step * regin_size, step)
    regin_id = tf.random.uniform([], 0, sample_class, dtype=tf.int64)
    angel = tf.random.uniform([], sample_range[regin_id * 2], sample_range[regin_id * 2 + 1], dtype=tf.float32)

    x = tf.tile(x, tf.cast([3, 3, 1], dtype=origin_size.dtype))
    x = tfa.image.rotate(x, angel, "BILINEAR")
    x = tf.image.central_crop(x, 1 / 3)
    x = tf.image.resize(x, origin_size)
    x = random_augment(x)
    return x, tf.one_hot(regin_id, sample_class)


@tf.function
def rotation_classification_process(x):
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    x = random_augment(x)
    return x, tf.one_hot(k, depth=tf.cast(4, dtype=tf.int32))


@tf.function
def self_augment_process(x, y=0):
    x = random_augment(x)
    return x, 0


randomTranslation = tf.keras.layers.experimental.preprocessing.RandomTranslation(
    (-0.3, 0.3),
    (-0.3, 0.3),
    fill_mode="reflect",
    interpolation="bilinear",
)
randomZoom = tf.keras.layers.experimental.preprocessing.RandomZoom(
    (-0.5, 1.),
    width_factor=None,
    fill_mode="reflect",
    interpolation="bilinear",
)
randomFlip = tf.keras.layers.experimental.preprocessing.RandomFlip()


@tf.function
def random_resize_and_crop(x, size=(84, 84), ratio_min=0.25, ratio_max=1.5, max_size=512):
    origin_size = tf.shape(x)
    max_len = tf.reduce_max(origin_size)
    x = tf.cond(max_len > tf.cast(max_size, origin_size.dtype),
                lambda: tf.image.resize(x, (max_size, max_size), preserve_aspect_ratio=True),
                lambda: x)
    x = random_size_ratio(x, ratio_min, ratio_max)
    x = tf.image.resize(x, size)
    return x


@tf.function
def do_augmentations(x, p=0.4):
    chance = tf.random.uniform([], 0, 100, dtype=tf.int32)
    x = tf.cast(x, tf.float32)

    @tf.function
    def process(x):
        # x = random_size_ratio(x, ratio_min=0.25, ratio_max=1.25)
        x = randomFlip(tf.expand_dims(x, 0))[0]
        x = random_rotate(x)
        x = random_augment(x)
        x = randomTranslation(tf.expand_dims(x, 0))[0]
        # x = tfa.image.random_cutout(tf.expand_dims(x, 0), (
        #     tf.random.uniform([], 12, 36, dtype=tf.int32), tf.random.uniform([], 12, 36, dtype=tf.int32)))[0]

        return x

    x_contra = tf.case([(tf.less(chance, tf.cast(p * 100, dtype=tf.int32)),
                         lambda: process(x))
                           , ((tf.greater(chance, tf.cast((1 - p) * 100, dtype=tf.int32)),
                               lambda: patch_augment(x)))
                        ],
                       default=lambda: x)

    return x_contra


@tf.function
def contrastive_process(x, p=0.4):
    x_contra = do_augmentations(x, p=0.4)

    return x_contra, 0.


@tf.function
def color_drop(x, p=0.2):
    @tf.function
    def process(x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 3])
        return x

    p = tf.cast(p * 100, tf.int64)
    chance = tf.random.uniform([], 0, 100, dtype=tf.int64)
    x = tf.cond(chance < tf.cast(p, chance.dtype),
                lambda: process(x),
                lambda: x)
    return x


@tf.function
def channel_shuffle(x, p=0.5):
    @tf.function
    def process(x):
        c1, c2, c3 = tf.unstack(tf.random.shuffle(tf.range(3)))
        x = tf.stack([x[..., c1], x[..., c2], x[..., c3]], -1)
        return x

    p = tf.cast(p * 100, tf.int64)
    chance = tf.random.uniform([], 0, 100, dtype=tf.int64)
    x = tf.cond(chance < tf.cast(p, chance.dtype),
                lambda: process(x),
                lambda: x)
    return x


@tf.function
def random_rotate(x, limit=360):
    x, _ = rotation_process_with_limit(x, limit)
    return x


@tf.function
def patch_augment(image):
    nums_jig = random.choice([2, 4])
    image = random_augment(image)
    image = randomFlip(tf.expand_dims(image, 0))[0]
    r, w, c = tf.unstack(tf.shape(image))

    step_r = r // nums_jig
    step_c = w // nums_jig
    x = tf.range(nums_jig)
    y = tf.range(nums_jig)
    X, Y = tf.meshgrid(x, y)
    mask = tf.stack([X, Y], -1)
    flatten_mask = tf.reshape(mask, [-1, 2])

    @tf.function
    def jig(xy):
        slice = tf.slice(image, tf.concat([xy * [step_c, step_r], [0]], axis=0),
                         tf.concat([[step_c, step_r], [3]], axis=0))
        slice = random_rotate(slice, 3)
        slice = random_size_ratio(slice, ratio_min=0.8, ratio_max=1.2)
        return slice

    x = tf.map_fn(jig, flatten_mask, image.dtype)
    x = tf.split(x, nums_jig, 0)

    @tf.function
    def merge(sub):
        return tf.concat(sub, -2)

    x = tf.map_fn(merge, x, image.dtype)
    x = tf.reshape(x, [r, w, c])

    return x


def possion_fusion(big_image, small_image, fusion_type_list=[cv2.MONOCHROME_TRANSFER, cv2.NORMAL_CLONE]):
    h, w, c = small_image.shape
    small_image = (small_image * 255).astype(np.uint8)
    big_image = (big_image * 255).astype(np.uint8)
    mask = np.ones_like(small_image)[..., 0] * 255
    big_image_h, big_image_w, big_image_c = big_image.shape
    diff_w = big_image_w // 2 - w // 2
    diff_h = big_image_h // 2 - h // 2
    center_x = big_image_w // 2 + int(random.uniform(-diff_w, diff_w))
    center_y = big_image_h // 2 + int(random.uniform(-diff_h, diff_h))

    fusion_type = random.choice(fusion_type_list)
    fusion = cv2.seamlessClone(small_image, big_image, mask, (center_x, center_y), fusion_type)
    fusion = fusion.astype(np.float32) / 255.
    return fusion


@tf.function
def possion_fusion_augment(big_image, small_image):
    fusion = tf.numpy_function(possion_fusion, inp=[big_image, small_image], Tout=big_image.dtype)
    return fusion


@tf.function
def random_size_ratio(x, ratio_min=0.25, ratio_max=1.5):
    x, _ = ratio_process(x, ratio_min=ratio_min, ratio_max=ratio_max)
    return x


@tf.function
def jigsaw(image, nums_jig=3):
    r, w, c = tf.unstack(tf.shape(image))

    step_r = r // nums_jig
    step_c = w // nums_jig
    x = tf.random.shuffle(tf.range(nums_jig))
    y = tf.random.shuffle(tf.range(nums_jig))
    X, Y = tf.meshgrid(x, y)
    mask = tf.stack([X, Y], -1)
    flatten_mask = tf.reshape(mask, [-1, 2])

    @tf.function
    def jig(xy):
        return tf.slice(image, tf.concat([xy * [step_c, step_r], [0]], axis=0),
                        tf.concat([[step_c, step_r], [3]], axis=0))

    x = tf.map_fn(jig, flatten_mask, image.dtype)
    x = tf.split(x, nums_jig, 0)

    @tf.function
    def merge(sub):
        return tf.concat(sub, -2)

    x = tf.map_fn(merge, x, image.dtype)
    x = tf.reshape(x, [r, w, c])

    flatten_mask = flatten_mask[..., -1] * nums_jig + flatten_mask[..., 0]
    return x, mask, flatten_mask


@tf.function
def jigsaw_with_augment(image, nums_jig=3):
    r, w, c = tf.unstack(tf.shape(image))

    step_r = r // nums_jig
    step_c = w // nums_jig
    x = tf.random.shuffle(tf.range(nums_jig))
    y = tf.random.shuffle(tf.range(nums_jig))
    X, Y = tf.meshgrid(x, y)
    mask = tf.stack([X, Y], -1)
    flatten_mask = tf.reshape(mask, [-1, 2])

    @tf.function
    def jig(xy):
        slice = tf.slice(image, tf.concat([xy * [step_c, step_r], [0]], axis=0),
                         tf.concat([[step_c, step_r], [3]], axis=0))
        slice = randomFlip(tf.expand_dims(slice, 0))[0]
        slice = random_rotate(slice, 10)
        slice = random_augment(slice)
        return slice

    x = tf.map_fn(jig, flatten_mask, image.dtype)
    x = tf.split(x, nums_jig, 0)

    @tf.function
    def merge(sub):
        return tf.concat(sub, -2)

    x = tf.map_fn(merge, x, image.dtype)
    x = tf.reshape(x, [r, w, c])

    flatten_mask = flatten_mask[..., -1] * nums_jig + flatten_mask[..., 0]
    return x, mask, flatten_mask

#
# image = tf.io.read_file("/data/giraffe/0_FSL/FSL/test.jpg")  # 根据路径读取 图片
# image = tf.image.decode_jpeg(image, channels=3)  # 图片解码
# # image = tf.image.resize(image, [92, 92])
# # image = tf.image.central_crop(image, 84 / 92.)
# image = tf.expand_dims(image, 0)
# image = tf.cast(image, tf.float32)
#
# for _ in range(100):
#     x = tf.map_fn(do_augmentations, image)
#     x = tf.cast(x, tf.uint8)
#     cv2.imshow("image", x[0, ..., ::-1].numpy())
#     cv2.waitKey(0)

# # # new_image = jigsaw(image)
# #
