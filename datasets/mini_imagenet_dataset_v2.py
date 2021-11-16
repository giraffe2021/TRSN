import copy
import os

import cv2
from tqdm import trange, tqdm
import random
import numpy as np
import tensorflow as tf
from functools import partial
import json
from multiprocessing.dummy import Pool as ThreadPool
from tools.augmentations import *

try:
    from .mini_imagenet_name import data_name_dict
except:
    from mini_imagenet_name import data_name_dict


class MiniImageNetDataLoader:
    def __init__(self, data_dir_path, way_num=5, shot_num=1, episode_test_sample_num=15):
        self.base_dir = data_dir_path
        self.meta_test_folder = os.path.join(self.base_dir, "test")
        self.meta_val_folder = os.path.join(self.base_dir, "val")
        self.meta_train_folder = os.path.join(self.base_dir, "train")
        assert os.path.exists(self.meta_test_folder)
        assert os.path.exists(self.meta_val_folder)
        assert os.path.exists(self.meta_train_folder)

        self.meta_test_folder_lists = sorted([os.path.join(self.meta_test_folder, label) \
                                              for label in os.listdir(self.meta_test_folder) \
                                              if os.path.isdir(os.path.join(self.meta_test_folder, label)) \
                                              ])
        self.meta_test_global_label_dict = {foldername: index for index, foldername in
                                            enumerate(self.meta_test_folder_lists)}

        self.meta_test_image_dict = {foldername: os.listdir(foldername) for index, foldername in
                                     enumerate(self.meta_test_folder_lists)}

        self.num_images_test = 0
        for v in self.meta_test_image_dict.values():
            self.num_images_test += len(v)

        self.meta_val_folder_lists = sorted([os.path.join(self.meta_val_folder, label) \
                                             for label in os.listdir(self.meta_val_folder) \
                                             if os.path.isdir(os.path.join(self.meta_val_folder, label)) \
                                             ])
        self.meta_val_global_label_dict = {foldername: index for index, foldername in
                                           enumerate(self.meta_val_folder_lists)}
        self.meta_val_image_dict = {foldername: os.listdir(foldername) for index, foldername in
                                    enumerate(self.meta_val_folder_lists)}

        self.num_images_val = 0
        for v in self.meta_val_image_dict.values():
            self.num_images_val += len(v)

        self.meta_train_folder_lists = sorted([os.path.join(self.meta_train_folder, label) \
                                               for label in os.listdir(self.meta_train_folder) \
                                               if os.path.isdir(os.path.join(self.meta_train_folder, label)) \
                                               ])
        self.meta_train_global_label_dict = {foldername: index for index, foldername in
                                             enumerate(self.meta_train_folder_lists)}
        self.meta_train_image_dict = {foldername: os.listdir(foldername) for index, foldername in
                                      enumerate(self.meta_train_folder_lists)}

        self.num_images_train = 0
        for v in self.meta_train_image_dict.values():
            self.num_images_train += len(v)

        print("=> {} loaded".format(data_dir_path))
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # cats | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(len(self.meta_train_folder_lists), self.num_images_train))
        print("  val      | {:5d} | {:8d}".format(len(self.meta_val_folder_lists), self.num_images_val))
        print("  test     | {:5d} | {:8d}".format(len(self.meta_test_folder_lists), self.num_images_test))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(len(self.meta_train_folder_lists)
                                                  + len(self.meta_val_folder_lists)
                                                  + len(self.meta_test_folder_lists),
                                                  self.num_images_train
                                                  + self.num_images_val
                                                  + self.num_images_test))
        print("  ------------------------------")

    def get_images(self, paths, labels, nb_samples=None, shuffle=True):
        if nb_samples is not None:
            sampler = lambda x: random.sample(x, nb_samples)
        else:
            sampler = lambda x: x
        images = [(i, os.path.join(path, image)) \
                  for i, path in zip(labels, paths) \
                  for image in sampler(os.listdir(path))]
        if shuffle:
            random.shuffle(images)
        return images

    def generate_origin_data_list(self, phase='train'):
        if phase == 'train':
            folders = self.meta_train_folder
        elif phase == 'val':
            folders = self.meta_val_folder
        elif phase == 'test':
            folders = self.meta_test_folder
        else:
            print('Please select vaild phase')
            all_filenames = []

        print('Generating filenames')
        all_filenames = []
        class_id = []
        class_dict = dict()
        for root, dirs, files in os.walk(folders, True):
            for name in files:
                if name.endswith(".jpg") or name.endswith(".png"):
                    file_path = os.path.join(root, name)
                    all_filenames.append(file_path)
                    if os.path.dirname(file_path) not in class_dict:
                        class_dict[os.path.dirname(file_path)] = len(class_dict)
                    class_id.append(class_dict[os.path.dirname(file_path)])

        return all_filenames

    def generate_data_list(self, way_num, num_samples_per_class, episode_num=None, phase='train'):
        if phase == 'train':
            folders = self.meta_train_folder_lists
            label_dict = self.meta_train_global_label_dict
            image_dict = self.meta_train_image_dict
            global_label_depth = len(label_dict)

            if episode_num is None:
                episode_num = 20000
            print('Generating train filenames')

        elif phase == 'val':
            folders = self.meta_val_folder_lists
            label_dict = self.meta_val_global_label_dict
            image_dict = self.meta_val_image_dict
            global_label_depth = len(label_dict)

            if episode_num is None:
                episode_num = 600
            print('Generating val filenames')

        elif phase == 'test':
            folders = self.meta_test_folder_lists
            label_dict = self.meta_test_global_label_dict
            image_dict = self.meta_test_image_dict
            global_label_depth = len(label_dict)

            if episode_num is None:
                episode_num = 600
            print('Generating test filenames')
        else:
            print('Please select vaild phase')
            all_labels = []
            all_filenames = []
            return all_filenames, all_labels
        try:
            name_projector = {v: data_name_dict[os.path.basename(k)] for k, v in label_dict.items()}
        except:
            name_projector = {}

        all_filenames = [[] for _ in range(episode_num)]
        all_meta_labels = [[] for _ in range(episode_num)]
        all_global_labels = [[] for _ in range(episode_num)]

        for index in tqdm(range(episode_num)):
            sampled_character_folders = random.sample(folders, way_num)
            random.shuffle(sampled_character_folders)
            filenames = []
            labels = []
            global_labels = []
            for meta_label, cls in enumerate(sampled_character_folders):
                image_list = image_dict[cls]
                global_label = label_dict[cls]
                images = random.sample(image_list, num_samples_per_class)
                filenames.extend([os.path.join(cls, name) for name in images])
                labels.extend([meta_label for _ in images])
                global_labels.extend([global_label for _ in images])

            all_filenames[index] = filenames
            all_meta_labels[index] = labels
            all_global_labels[index] = global_labels
        return (all_filenames, all_meta_labels, all_global_labels), global_label_depth, name_projector

    def generate_data_list_no_putback(self, way_num, num_samples_per_class, episode_num=None, phase='train'):
        if phase == 'train':
            folders = self.meta_train_folder_lists
            label_dict = self.meta_train_global_label_dict
            image_dict = self.meta_train_image_dict
            global_label_depth = len(label_dict)

            if episode_num is None:
                episode_num = 20000
            print('Generating train filenames')

        elif phase == 'val':
            folders = self.meta_val_folder_lists
            label_dict = self.meta_val_global_label_dict
            image_dict = self.meta_val_image_dict
            global_label_depth = len(label_dict)

            if episode_num is None:
                episode_num = 600
            print('Generating val filenames')

        elif phase == 'test':
            folders = self.meta_test_folder_lists
            label_dict = self.meta_test_global_label_dict
            image_dict = self.meta_test_image_dict
            global_label_depth = len(label_dict)

            if episode_num is None:
                episode_num = 600
            print('Generating test filenames')
        else:
            print('Please select vaild phase')
            all_labels = []
            all_filenames = []
            return all_filenames, all_labels
        try:
            name_projector = {v: data_name_dict[os.path.basename(k)] for k, v in label_dict.items()}
        except:
            name_projector = {}

        all_filenames = [[] for _ in range(episode_num)]
        all_meta_labels = [[] for _ in range(episode_num)]
        all_global_labels = [[] for _ in range(episode_num)]
        temp_image_dict = copy.deepcopy(image_dict)

        flag_of_sampled = set()
        for image_list in image_dict.values():
            flag_of_sampled = flag_of_sampled.union(set(image_list))
        total_count = len(flag_of_sampled)

        bar = tqdm(range(episode_num))
        for index in bar:
            bar.set_description("Processing %s" % "per:{} %".format(100. - 100 * len(flag_of_sampled) / total_count))
            sampled_character_folders = random.sample(folders, way_num)
            random.shuffle(sampled_character_folders)
            filenames = []
            labels = []
            global_labels = []
            for meta_label, cls in enumerate(sampled_character_folders):
                image_list = temp_image_dict[cls]
                global_label = label_dict[cls]
                if len(image_list) > num_samples_per_class:
                    images = random.sample(image_list, num_samples_per_class)
                    temp_image_dict[cls] = list(set(temp_image_dict[cls]) - set(images))
                    flag_of_sampled = flag_of_sampled - set(images)
                else:
                    images = copy.deepcopy(image_list)
                    rest_images = random.sample(image_dict[cls], num_samples_per_class - len(images))
                    temp_image_dict[cls] = list(set(image_dict[cls]) - set(rest_images))
                    images = images + rest_images
                    flag_of_sampled = flag_of_sampled - set(images)
                filenames.extend([os.path.join(cls, name) for name in images])
                labels.extend([meta_label for _ in images])
                global_labels.extend([global_label for _ in images])

            all_filenames[index] = filenames
            all_meta_labels[index] = labels
            all_global_labels[index] = global_labels

        return (all_filenames, all_meta_labels, all_global_labels), global_label_depth, name_projector

    def get_dataset(self, phase='train', way_num=5, shot_num=5, episode_test_sample_num=15, episode_num=None,
                    batch=1,
                    augment=False,
                    mix_up=False,
                    epochs=1):

        # @tf.function
        # def read_image(path, meta_label, global_label):
        #     image = tf.io.read_file(path)  # 根据路径读取图片
        #     image = tf.image.decode_jpeg(image, channels=3)  # 图片解码
        #
        #     image = tf.cast(image, dtype=tf.float32) / 255.
        #
        #     meta_label = tf.cast(meta_label, dtype=tf.int64)
        #     global_label = tf.cast(global_label, dtype=tf.int64)
        #     return image, meta_label, global_label

        @tf.function
        def read_image_with_random_crop_resize(path):
            image = tf.io.read_file(path)  # 根据路径读取图片
            image = tf.image.decode_jpeg(image, channels=3)  # 图片解码

            image = tf.cast(image, dtype=tf.float32) / 255.
            image = random_resize_and_crop(image, (84, 84))
            return image

        @tf.function
        def split(x, y, g_y, way_num, episode_test_sample_num):
            x = tf.stack(tf.split(x, way_num), axis=0)
            y = tf.stack(tf.split(y, way_num), axis=0)
            g_y = tf.stack(tf.split(g_y, way_num), axis=0)

            query_x = x[:, :episode_test_sample_num, ...]
            query_y = y[:, :episode_test_sample_num, ...]
            query_g_y = g_y[:, :episode_test_sample_num, ...]

            support_x = x[:, episode_test_sample_num:, ...]
            surport_y = y[:, episode_test_sample_num:, ...]
            surport_g_y = g_y[:, episode_test_sample_num:, ...]

            return (support_x, surport_y, surport_g_y), (query_x, query_y, query_g_y)

        @tf.function
        def read_image(path):
            image = tf.io.read_file(path)  # 根据路径读取图片
            image = tf.image.decode_jpeg(image, channels=3)  # 图片解码
            image = tf.cast(image, dtype=tf.float32) / 255.
            image = tf.image.resize(image, [92, 92])
            image = tf.image.central_crop(image, 84 / 92.)
            image = tf.image.resize(image, [84, 84])
            return image

        @tf.function
        def process_with_augment(path, meta_label, global_label, way_num, episode_test_sample_num,
                                 global_label_depth):
            images = tf.map_fn(read_image_with_random_crop_resize, path, dtype=tf.float32)
            images = tf.map_fn(do_augmentations, images, parallel_iterations=16)
            # images = (images - tf.cast([0.485, 0.456, 0.406], tf.float32)) / tf.cast([0.229, 0.224, 0.225], tf.float32)
            meta_label = tf.one_hot(meta_label, axis=-1, depth=way_num)
            global_label = tf.one_hot(global_label, axis=-1, depth=global_label_depth)
            return split(images, meta_label, global_label, way_num, episode_test_sample_num)

        @tf.function
        def process_with_mip_up_augment(path, meta_label, global_label, way_num, episode_test_sample_num,
                                        global_label_depth):
            images = tf.map_fn(read_image_with_random_crop_resize, path, dtype=tf.float32)
            x = tf.map_fn(do_augmentations, images, parallel_iterations=16)
            # x = (x - tf.cast([0.485, 0.456, 0.406], tf.float32)) / tf.cast([0.229, 0.224, 0.225], tf.float32)

            meta_y = tf.one_hot(meta_label, axis=-1, depth=way_num)
            y = tf.one_hot(global_label, axis=-1, depth=global_label_depth)

            alfa_shape = tf.unstack(tf.shape(images)[:1])
            alfa = tf.random.uniform(alfa_shape, 0.3, 1.0)
            alfa_x = tf.reshape(alfa, [*alfa_shape, 1, 1, 1])

            indices = tf.random.shuffle(tf.range(tf.shape(x)[0]))
            x_reference = tf.gather(x, indices)
            y_reference = tf.gather(y, indices)
            meta_y_reference = tf.gather(meta_y, indices)
            # 保持 相同类别,和不同类别的混合
            x = x * alfa_x + x_reference * (1.0 - alfa_x)
            alfa_y = tf.reshape(alfa, [*alfa_shape, 1])
            y = y * alfa_y + y_reference * (1.0 - alfa_y)
            meta_y = meta_y * alfa_y + meta_y_reference * (1.0 - alfa_y)

            return split(x, meta_y, y, way_num, episode_test_sample_num)

        @tf.function
        def process_without_augment(path, meta_label, global_label, way_num, episode_test_sample_num,
                                    global_label_depth):
            images = tf.map_fn(read_image, path, dtype=tf.float32)

            # images = (images - tf.cast([0.485, 0.456, 0.406], tf.float32)) / tf.cast([0.229, 0.224, 0.225], tf.float32)

            meta_label = tf.one_hot(meta_label, axis=-1, depth=way_num)
            global_label = tf.one_hot(global_label, axis=-1, depth=global_label_depth)
            return split(images, meta_label, global_label, way_num, episode_test_sample_num)

        if episode_num is None:
            if phase == 'train':
                episode_num = 20000
            else:
                episode_num = 600

        num_samples_per_class = shot_num + episode_test_sample_num
        dataset = None
        for _ in tqdm(range(epochs)):

            data_set, global_label_depth, name_projector = self.generate_data_list(way_num,
                                                                                   num_samples_per_class,
                                                                                   episode_num=episode_num,
                                                                                   phase=phase)
            ds = tf.data.Dataset.from_tensor_slices(data_set)

            if augment is True:
                process = process_with_augment
            else:
                process = process_without_augment

            ds_episode = ds.map(partial(process,
                                        way_num=way_num,
                                        episode_test_sample_num=episode_test_sample_num,
                                        global_label_depth=global_label_depth),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch, drop_remainder=True)

            if mix_up is True:
                ds_episode_mixup = ds.map(partial(process_with_mip_up_augment,
                                                  way_num=way_num,
                                                  episode_test_sample_num=episode_test_sample_num,
                                                  global_label_depth=global_label_depth),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch,
                                                                                                  drop_remainder=True)
                ds_episode = ds_episode_mixup.concatenate(ds_episode)
                ds_episode_with_out_augments = ds.map(partial(process_without_augment,
                                                              way_num=way_num,
                                                              episode_test_sample_num=episode_test_sample_num,
                                                              global_label_depth=global_label_depth),
                                                      num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch,
                                                                                                              drop_remainder=True)
                ds_episode = ds_episode.concatenate(ds_episode_with_out_augments)

            if dataset is None:
                dataset = ds_episode
            else:
                dataset = dataset.concatenate(ds_episode)
        return dataset, name_projector

    def get_dataset_V2(self, phase='train', way_num=5, shot_num=5, episode_test_sample_num=15, episode_num=None,
                       batch=1,
                       augment=False,
                       mix_up=False,
                       epochs=1):

        # @tf.function
        # def read_image(path, meta_label, global_label):
        #     image = tf.io.read_file(path)  # 根据路径读取图片
        #     image = tf.image.decode_jpeg(image, channels=3)  # 图片解码
        #
        #     image = tf.cast(image, dtype=tf.float32) / 255.
        #
        #     meta_label = tf.cast(meta_label, dtype=tf.int64)
        #     global_label = tf.cast(global_label, dtype=tf.int64)
        #     return image, meta_label, global_label

        @tf.function
        def read_image_with_random_crop_resize(path):
            image = tf.io.read_file(path)  # 根据路径读取图片
            image = tf.image.decode_jpeg(image, channels=3)  # 图片解码

            image = tf.cast(image, dtype=tf.float32) / 255.
            image = random_resize_and_crop(image, (84, 84))
            return image

        @tf.function
        def split(x, y, g_y, way_num, episode_test_sample_num):
            x = tf.stack(tf.split(x, way_num), axis=0)
            y = tf.stack(tf.split(y, way_num), axis=0)
            g_y = tf.stack(tf.split(g_y, way_num), axis=0)

            query_x = x[:, :episode_test_sample_num, ...]
            query_y = y[:, :episode_test_sample_num, ...]
            query_g_y = g_y[:, :episode_test_sample_num, ...]

            support_x = x[:, episode_test_sample_num:, ...]
            surport_y = y[:, episode_test_sample_num:, ...]
            surport_g_y = g_y[:, episode_test_sample_num:, ...]

            return (support_x, surport_y, surport_g_y), (query_x, query_y, query_g_y)

        @tf.function
        def read_image(path):
            image = tf.io.read_file(path)  # 根据路径读取图片
            image = tf.image.decode_jpeg(image, channels=3)  # 图片解码
            image = tf.cast(image, dtype=tf.float32) / 255.
            image = tf.image.resize(image, [92, 92])
            image = tf.image.central_crop(image, 84 / 92.)
            image = tf.image.resize(image, [84, 84])
            return image

        @tf.function
        def process_with_augment(path, meta_label, global_label, way_num, episode_test_sample_num,
                                 global_label_depth):
            images = tf.map_fn(read_image_with_random_crop_resize, path, dtype=tf.float32)
            images = tf.map_fn(do_augmentations, images, parallel_iterations=16)
            # images = (images - tf.cast([0.485, 0.456, 0.406], tf.float32)) / tf.cast([0.229, 0.224, 0.225], tf.float32)
            meta_label = tf.one_hot(meta_label, axis=-1, depth=way_num)
            global_label = tf.one_hot(global_label, axis=-1, depth=global_label_depth)
            return split(images, meta_label, global_label, way_num, episode_test_sample_num)

        @tf.function
        def process_with_mip_up_augment(path, meta_label, global_label, way_num, episode_test_sample_num,
                                        global_label_depth):
            images = tf.map_fn(read_image_with_random_crop_resize, path, dtype=tf.float32)
            x = tf.map_fn(do_augmentations, images, parallel_iterations=16)
            # x = (x - tf.cast([0.485, 0.456, 0.406], tf.float32)) / tf.cast([0.229, 0.224, 0.225], tf.float32)

            meta_y = tf.one_hot(meta_label, axis=-1, depth=way_num)
            y = tf.one_hot(global_label, axis=-1, depth=global_label_depth)

            alfa_shape = tf.unstack(tf.shape(images)[:1])
            alfa = tf.random.uniform(alfa_shape, 0.3, 1.0)
            alfa_x = tf.reshape(alfa, [*alfa_shape, 1, 1, 1])

            indices = tf.random.shuffle(tf.range(tf.shape(x)[0]))
            x_reference = tf.gather(x, indices)
            y_reference = tf.gather(y, indices)
            meta_y_reference = tf.gather(meta_y, indices)
            # 保持 相同类别,和不同类别的混合
            x = x * alfa_x + x_reference * (1.0 - alfa_x)
            alfa_y = tf.reshape(alfa, [*alfa_shape, 1])
            y = y * alfa_y + y_reference * (1.0 - alfa_y)
            meta_y = meta_y * alfa_y + meta_y_reference * (1.0 - alfa_y)

            return split(x, meta_y, y, way_num, episode_test_sample_num)

        @tf.function
        def process_without_augment(path, meta_label, global_label, way_num, episode_test_sample_num,
                                    global_label_depth):
            images = tf.map_fn(read_image, path, dtype=tf.float32)

            # images = (images - tf.cast([0.485, 0.456, 0.406], tf.float32)) / tf.cast([0.229, 0.224, 0.225], tf.float32)

            meta_label = tf.one_hot(meta_label, axis=-1, depth=way_num)
            global_label = tf.one_hot(global_label, axis=-1, depth=global_label_depth)
            return split(images, meta_label, global_label, way_num, episode_test_sample_num)

        if episode_num is None:
            if phase == 'train':
                episode_num = 20000
            else:
                episode_num = 600

        num_samples_per_class = shot_num + episode_test_sample_num
        if phase == 'train':
            all_data_set, global_label_depth, name_projector = self.generate_data_list_no_putback(way_num,
                                                                                                  num_samples_per_class,
                                                                                                  episode_num=episode_num * epochs,
                                                                                                  phase=phase)
        else:
            all_data_set, global_label_depth, name_projector = self.generate_data_list(way_num,
                                                                                       num_samples_per_class,
                                                                                       episode_num=episode_num * epochs,
                                                                                       phase=phase)

        dataset = None
        for ep in tqdm(range(epochs)):
            data_set = all_data_set[0][ep * episode_num:(ep + 1) * episode_num] \
                , all_data_set[1][ep * episode_num:(ep + 1) * episode_num] \
                , all_data_set[2][ep * episode_num:(ep + 1) * episode_num]
            if augment is True:
                process = process_with_augment
            else:
                process = process_without_augment

            if mix_up is True:
                splits = episode_num // 3
                splits_data_set = (data_set[0][:splits], data_set[1][:splits], data_set[2][:splits])
                ds = tf.data.Dataset.from_tensor_slices(splits_data_set).shuffle(10000)
                ds_episode = ds.map(partial(process,
                                            way_num=way_num,
                                            episode_test_sample_num=episode_test_sample_num,
                                            global_label_depth=global_label_depth),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch, drop_remainder=True)

                splits_data_set = (
                    data_set[0][splits:splits * 2], data_set[1][splits:splits * 2], data_set[2][splits:splits * 2])
                ds = tf.data.Dataset.from_tensor_slices(splits_data_set).shuffle(10000)
                ds_episode_mixup = ds.map(partial(process_with_mip_up_augment,
                                                  way_num=way_num,
                                                  episode_test_sample_num=episode_test_sample_num,
                                                  global_label_depth=global_label_depth),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch,
                                                                                                  drop_remainder=True)
                ds_episode = ds_episode_mixup.concatenate(ds_episode)

                splits_data_set = (
                    data_set[0][splits * 2:], data_set[1][splits * 2:], data_set[2][splits * 2:])
                ds = tf.data.Dataset.from_tensor_slices(splits_data_set).shuffle(10000)
                ds_episode_with_out_augments = ds.map(partial(process_without_augment,
                                                              way_num=way_num,
                                                              episode_test_sample_num=episode_test_sample_num,
                                                              global_label_depth=global_label_depth),
                                                      num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch,
                                                                                                              drop_remainder=True)
                ds_episode = ds_episode.concatenate(ds_episode_with_out_augments)
            else:
                ds = tf.data.Dataset.from_tensor_slices(data_set)
                ds_episode = ds.map(partial(process,
                                            way_num=way_num,
                                            episode_test_sample_num=episode_test_sample_num,
                                            global_label_depth=global_label_depth),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch, drop_remainder=True)

            if dataset is None:
                dataset = ds_episode
            else:
                dataset = dataset.concatenate(ds_episode)
        return dataset, name_projector


if __name__ == '__main__':
    data_dir_path = "/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images"
    dataloader = MiniImageNetDataLoader(data_dir_path=data_dir_path)
    meta_train_ds, meta_train_name_projector = dataloader.get_dataset(phase='train', way_num=5, shot_num=5,
                                                                      episode_test_sample_num=5,
                                                                      episode_num=600,
                                                                      batch=4,
                                                                      augment=False)
