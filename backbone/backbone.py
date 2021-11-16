import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from tensorflow.keras import layers, Model
from .resnet18 import ResNet18, ResNet12
from .conv4 import conv4_net
from .wrn_28_10 import wrn_28_10
from .resnet_12 import ResNet_12
from functools import partial
import os
import fnmatch
import random
import traceback
import numpy as np

# self supervised training
back_bone_dict = {"resnet18": ResNet18, "conv4": conv4_net, "wrn_28_10": wrn_28_10, "resnet12": ResNet12,
                  "resnet50": partial(tf.keras.applications.ResNet50, include_top=False, weights=None),
                  "MobileNetV2_0_35": partial(tf.keras.applications.MobileNetV2, alpha=0.35, include_top=False,
                                              weights='imagenet'),
                  "resnet_12": ResNet_12}


class Backbone:
    def __init__(self, backbone="conv4", input_shape=(84, 84, 3), pooling='avg', use_bias=True, name=None):
        if name is not None:
            self.encoder = back_bone_dict[backbone](input_shape=input_shape, pooling=pooling, use_bias=use_bias,
                                                    name=name)
        else:
            self.encoder = back_bone_dict[backbone](input_shape=input_shape, pooling=pooling, use_bias=use_bias)

    def load_weights(self, path):
        self.encoder.load_weights(path)

    def get_model(self, *args, **kwargs):
        return self.encoder


if __name__ == '__main__':
    encoder = Backbone()
    encoder.load_weights("/data/giraffe/0_FSL/FSL/ckpts/pretrain/export/encoder.h5")
    model = encoder.get_model()
    model.summary()
