#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Amir Alansary <amiralansary@gmail.com>
# Modified: Athanasios Vlontzos <athanasiosvlontzos@gmail.com>

def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)

import numpy as np

import os
import sys
import time
import argparse
import random
from collections import deque
import shutil
import importlib
import csv

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # suppresses DeprecationWarning s
from medical import MedicalPlayer, FrameStack
from tensorpack.input_source import QueueInput
from tensorpack_medical.models.conv3d import Conv3D
from tensorpack_medical.models.pool3d import MaxPooling3D
from common import Evaluator, eval_model_multithread, play_n_episodes
from DQNModel import Model3D as DQNModel
from expreplay import ExpReplay
from attention import multihead_attention
from attention_commom import multihead_attention_nd
from augmented_attention import augmented_conv2d, augmented_conv3d

from tensorpack import (PredictConfig, OfflinePredictor, get_model_loader,
                        logger, TrainConfig, ModelSaver, PeriodicTrigger,
                        ScheduledHyperParamSetter, ObjAttrParam,
                        HumanHyperParamSetter, argscope, RunOp, LinearWrap,
                        FullyConnected, PReLU, SimpleTrainer,
                        launch_train_with_config, fix_rng_seed, AutoResumeTrainConfig, HyperParamSetterWithFunc)
from tensorpack.models import BatchNorm
from tensorboardX import SummaryWriter

###############################################################################
# BATCH SIZE USED IN NATURE PAPER IS 32 - MEDICAL IS 256
BATCH_SIZE = 48  # 48
# BREAKOUT (84,84) - MEDICAL 2D (60,60) - MEDICAL 3D (26,26,26)
IMAGE_SIZE = (15, 15, 15)
# how many frames to keep
# in other words, how many observations the network can see
FRAME_HISTORY = 4  # default 4
# the frequency of updating the target network
UPDATE_FREQ = 4  # default 4
# DISCOUNT FACTOR - NATURE (0.99) - MEDICAL (0.9)
GAMMA = 0.9  # 0.99
# REPLAY MEMORY SIZE - NATURE (1e6) - MEDICAL (1e5 view-patches)
MEMORY_SIZE = 1e5  # 6#3   # to debug on bedale use 1e4
# consume at least 1e6 * 27 * 27 * 27 bytes
INIT_MEMORY_SIZE = MEMORY_SIZE // 20  # 5e4
# each epoch is 100k played frames
STEPS_PER_EPOCH = 15000 // UPDATE_FREQ  # default prev: 15000, diana: 100000 e.g. 100000//4

# TODO: understand: EPOCHS_PER_EVAL, EVAL_EPISODE
# TODO: consider global history length setting from Medical Player

# num training epochs in between model evaluations
EPOCHS_PER_EVAL = 1  # default: 1
# the number of episodes to run during evaluation
EVAL_EPISODE = 20  # default: 50
# the number of epochs to run during training
EPOCHS = 20  # default: 20
# the number of steps to perform when evaluating
STEPS_PER_EVAL = 20  # default: 50
# maximum number of movements per step
MAX_NUM_FRAMES = 1500  # 10000 # default: 1500
# how many epochs should be saved?
MAX_TO_KEEP = EPOCHS  # default: EPOCHS = 20
# use multiscaling?
MULTISCALE = True  # default: True

ATTENTION = 0  # 0 means no attention
if ATTENTION == 1 or ATTENTION == 15:
    IMAGE_SIZE = (45, 45, 45)

SEED = 1


###############################################################################

def get_player(directory=None, files_list=None, viz=False,
               task='play', saveGif=False, saveVideo=False, agents=2, reward_strategy=1, coords_init=None, mask=None):
    # in atari paper, max_num_frames = 30000
    env = MedicalPlayer(directory=directory, screen_dims=IMAGE_SIZE,
                        viz=viz, saveGif=saveGif, saveVideo=saveVideo,
                        task=task, files_list=files_list, multiscale=MULTISCALE, agents=agents,
                        max_num_frames=MAX_NUM_FRAMES,
                        reward_strategy=reward_strategy, coords_init=coords_init, mask=mask, seed=SEED)
    if (task != 'train'):
        # in training, env will be decorated by ExpReplay, and history
        # is taken care of in expreplay buffer
        # otherwise, FrameStack modifies self.step to save observations into a queue
        env = FrameStack(env, FRAME_HISTORY, agents=agents)
    return env


###############################################################################

class Model(DQNModel):
    def __init__(self, agents=2):
        super(Model, self).__init__(IMAGE_SIZE, FRAME_HISTORY, METHOD, NUM_ACTIONS, GAMMA, agents)

    def _get_DQN_prediction(self, images):
        """ image: [0,255]

        :returns predicted Q values"""

        agents = len(images)

        Q_list = []

        with argscope(Conv3D, nl=PReLU.symbolic_function, use_bias=True):
            if ATTENTION == 19:
                images[0] = images[0] / 255.0
                conv_0 = tf.layers.conv3d(images[0], name='conv0',
                                          filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1], padding='same',
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                              2.0, seed=SEED),
                                          bias_initializer=tf.zeros_initializer())
                max_pool_0 = tf.layers.max_pooling3d(conv_0, 2, 2, name='max_pool0')
                conv_1 = tf.layers.conv3d(max_pool_0, name='conv1',
                                          filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1], padding='same',
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                              2.0, seed=SEED),
                                          bias_initializer=tf.zeros_initializer())
                max_pool1 = tf.layers.max_pooling3d(conv_1, 2, 2, name='max_pool1')
                conv_2 = tf.layers.conv3d(max_pool1, name='conv2',
                                          filters=64, kernel_size=[4, 4, 4], strides=[1, 1, 1], padding='same',
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                              2.0, seed=SEED),
                                          bias_initializer=tf.zeros_initializer())
                max_pool2 = tf.layers.max_pooling3d(conv_2, 2, 2, name='max_pool2')
                conv_3 = tf.layers.conv3d(max_pool2, name='conv3',
                                          filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding='same',
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                              2.0, seed=SEED), bias_initializer=tf.zeros_initializer())
                conv_3_reshaped = tf.reshape(conv_3, (-1, 1, 64))
                qkv = tf.concat([conv_3_reshaped, conv_3_reshaped, conv_3_reshaped], axis=0)
                attn = multihead_attention('attn', qkv, num_heads=8)
                attn_min = tf.reduce_min(attn, name="attn_min")
                attn_max = tf.reduce_max(attn, name="attn_max")
                attn_1 = tf.div(tf.subtract(attn, attn_min), tf.subtract(attn_max, attn_min))
                attn_2 = tf.add(conv_3_reshaped, attn_1, name="attn_added")

                for i in range(0, agents):
                    fc0 = FullyConnected('fc0V_{}'.format(i), attn_2, 512, activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                           seed=SEED))
                    fc1 = FullyConnected('fc1V_{}'.format(i), fc0, 256, activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                           seed=SEED))
                    fc2 = FullyConnected('fc2V_{}'.format(i), fc1, 128, activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                           seed=SEED))
                    V = FullyConnected('fctV_{}'.format(i), fc2, 1, nl=tf.identity)
                    fcA0 = FullyConnected('fc0A_{}'.format(i), attn_2, 512, activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                            seed=SEED))
                    fcA1 = FullyConnected('fc1A_{}'.format(i), fcA0, 256, activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                            seed=SEED))
                    fcA2 = FullyConnected('fc2A_{}'.format(i), fcA1, 128, activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                            seed=SEED))
                    A = FullyConnected('fctA_{}'.format(i), fcA2, self.num_actions, nl=tf.identity)

                    # TODO: why are we subtracting the mean here? It sure helps!
                    V = tf.subtract(V, tf.reduce_mean(A, 1, keepdims=True), name='Vvalue_{}'.format(i))
                    Q = tf.add(A, V, name='Qvalue_{}'.format(i))

                    Q_list.append(Q)
            elif ATTENTION == 21:
                # test = np.array(np.linspace(1,36, 36), dtype=np.float32).reshape((-1, 3, 3, 4))
                # tf_test = tf.Variable(test)
                images[0] = images[0] / 255.0
                # # two_d = images[0][:, 0]
                # # out = augmented_conv2d('aug_attn_t', tf_test, Fout=32, k=[3, 3], dk=3, dv=3, Nh=1, relative=True)
                # # conv_0 = augmented_conv2d('aug_attn', two_d, Fout=32, k=[5, 5], dk=4, dv=4, Nh=4, relative=True)
                # # max_pool_0 = tf.layers.max_pooling3d(conv_0, 2, 2, name='max_pool0')
                # conv_1 = augmented_conv3d('aug_attn_1', images[0], Fout=32, k=[5, 5, 5], dk=4, dv=4, Nh=4,
                #                           relative=True)
                # print("du kranker ficker du")

                with argscope(Conv3D, nl=PReLU.symbolic_function, use_bias=True,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                  2.0, seed=SEED)):

                    # conv_0 = augmented_conv3d('conv0', images[0],
                    #                           filters=32, kernel_size=[5, 5, 5], dk=4, dv=4, Nh=1,
                    #                           relative=True, reuse=None)
                    conv_0 = tf.layers.conv3d(images[0], name='conv0',
                                              filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1],
                                              padding='same',
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                  2.0, seed=SEED),
                                              bias_initializer=tf.zeros_initializer())
                    max_pool_0 = tf.layers.max_pooling3d(conv_0, 2, 2, name='max_pool0')
                    conv_1 = augmented_conv3d('conv1', max_pool_0,
                                              filters=32, kernel_size=[5, 5, 5], dk=4, dv=4, Nh=4,
                                              relative=True, reuse=None)
                    # conv_1 = BatchNorm("conv_1_norm", conv_1)
                    conv_1 = tf.contrib.layers.batch_norm(conv_1, is_training=True, updates_collections=None, decay=0.9)
                    max_pool1 = tf.layers.max_pooling3d(conv_1, 2, 2, name='max_pool1')
                    conv_2 = augmented_conv3d('conv2', max_pool1,
                                              filters=64, kernel_size=[4, 4, 4], dk=4, dv=4, Nh=4,
                                              relative=True, reuse=None)
                    # conv_2 = BatchNorm("conv_2_norm", inputs=conv_2, axis=0)
                    conv_2 = tf.contrib.layers.batch_norm(conv_2, is_training=True, updates_collections=None, decay=0.9)
                    max_pool2 = tf.layers.max_pooling3d(conv_2, 2, 2, name='max_pool2')
                    conv_3 = augmented_conv3d('conv3', max_pool2,
                                              filters=64, kernel_size=[3, 3, 3], dk=4, dv=4, Nh=4,
                                              relative=True, reuse=None)
                    # conv_3 = BatchNorm("conv_3_norm", inputs=conv_3)
                    conv_3 = tf.contrib.layers.batch_norm(conv_3, is_training=True, updates_collections=None, decay=0.9)
                    for i in range(0, agents):
                        assert "duel" in str.lower(self.method)
                        fc0 = FullyConnected('fc0V_{}'.format(i), conv_3, 512, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                               seed=SEED))
                        fc1 = FullyConnected('fc1V_{}'.format(i), fc0, 256, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                               seed=SEED))
                        fc2 = FullyConnected('fc2V_{}'.format(i), fc1, 128, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                               seed=SEED))
                        V = FullyConnected('fctV_{}'.format(i), fc2, 1, nl=tf.identity)
                        fcA0 = FullyConnected('fc0A_{}'.format(i), conv_3, 512, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                seed=SEED))
                        fcA1 = FullyConnected('fc1A_{}'.format(i), fcA0, 256, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                seed=SEED))
                        fcA2 = FullyConnected('fc2A_{}'.format(i), fcA1, 128, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                seed=SEED))
                        A = FullyConnected('fctA_{}'.format(i), fcA2, self.num_actions, nl=tf.identity)

                        # TODO: why are we subtracting the mean here? It sure helps!
                        V = tf.subtract(V, tf.reduce_mean(A, 1, keepdims=True), name='Vvalue_{}'.format(i))
                        Q = tf.add(A, V, name='Qvalue_{}'.format(i))

                        Q_list.append(Q)

            elif ATTENTION == 22:
                images[0] = images[0] / 255.0
                with argscope(Conv3D, nl=PReLU.symbolic_function, use_bias=True,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                  2.0, seed=SEED)):

                    # conv_0 = augmented_conv3d('conv0', images[0],
                    #                           filters=32, kernel_size=[5, 5, 5], dk=4, dv=4, Nh=1,
                    #                           relative=True, reuse=None)
                    conv_0 = tf.layers.conv3d(images[0], name='conv0',
                                              filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1],
                                              padding='same',
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                  2.0, seed=SEED),
                                              bias_initializer=tf.zeros_initializer())
                    max_pool_0 = tf.layers.max_pooling3d(conv_0, 2, 2, name='max_pool0')
                    conv_1 = augmented_conv3d('conv1', max_pool_0,
                                              filters=32, kernel_size=[5, 5, 5], dk=4, dv=4, Nh=4,
                                              relative=True, reuse=None)
                    # conv_1 = BatchNorm("conv_1_norm", conv_1)
                    conv_1 = tf.contrib.layers.batch_norm(conv_1, is_training=True, updates_collections=None, decay=0.9)
                    conv_1 = tf.add(conv_1, max_pool_0)  # residual
                    max_pool_1 = tf.layers.max_pooling3d(conv_1, 2, 2, name='max_pool1')
                    # conv_2 = augmented_conv3d('conv2', max_pool_1,
                    #                           filters=64, kernel_size=[4, 4, 4], dk=4, dv=4, Nh=4,
                    #                           relative=True, reuse=None)
                    # conv_2 = BatchNorm("conv_2_norm", inputs=conv_2, axis=0)
                    # conv_2 = tf.contrib.layers.batch_norm(conv_2, is_training=True, updates_collections=None, decay=0.9)
                    # adding here is not possible due to depth 32 -> 64, probably best to use normal conv layer
                    conv_2 = tf.layers.conv3d(max_pool_1, name='conv2', filters=64, kernel_size=[4, 4, 4],
                                              strides=[1, 1, 1], padding='same',
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                seed=SEED),
                                              bias_initializer=tf.zeros_initializer())
                    max_pool_2 = tf.layers.max_pooling3d(conv_2, 2, 2, name='max_pool2')
                    conv_3 = augmented_conv3d('conv3', max_pool_2,
                                              filters=64, kernel_size=[3, 3, 3], dk=4, dv=4, Nh=4,
                                              relative=True, reuse=None)
                    # conv_3 = BatchNorm("conv_3_norm", inputs=conv_3)
                    conv_3 = tf.contrib.layers.batch_norm(conv_3, is_training=True, updates_collections=None, decay=0.9)
                    conv_3 = tf.add(conv_3, max_pool_2)
                    for i in range(0, agents):
                        assert "duel" in str.lower(self.method)
                        fc0 = FullyConnected('fc0V_{}'.format(i), conv_3, 512, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                               seed=SEED))
                        fc1 = FullyConnected('fc1V_{}'.format(i), fc0, 256, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                               seed=SEED))
                        fc2 = FullyConnected('fc2V_{}'.format(i), fc1, 128, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                               seed=SEED))
                        V = FullyConnected('fctV_{}'.format(i), fc2, 1, nl=tf.identity)
                        fcA0 = FullyConnected('fc0A_{}'.format(i), conv_3, 512, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                seed=SEED))
                        fcA1 = FullyConnected('fc1A_{}'.format(i), fcA0, 256, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                seed=SEED))
                        fcA2 = FullyConnected('fc2A_{}'.format(i), fcA1, 128, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                seed=SEED))
                        A = FullyConnected('fctA_{}'.format(i), fcA2, self.num_actions, nl=tf.identity)

                        # TODO: why are we subtracting the mean here? It sure helps!
                        V = tf.subtract(V, tf.reduce_mean(A, 1, keepdims=True), name='Vvalue_{}'.format(i))
                        Q = tf.add(A, V, name='Qvalue_{}'.format(i))

                        Q_list.append(Q)
            elif ATTENTION == 23 or ATTENTION == 24:
                images[0] = images[0] / 255.0
                with argscope(Conv3D, nl=PReLU.symbolic_function, use_bias=True,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                  2.0, seed=SEED)):

                    # conv_0 = augmented_conv3d('conv0', images[0],
                    #                           filters=32, kernel_size=[5, 5, 5], dk=4, dv=4, Nh=1,
                    #                           relative=True, reuse=None)
                    conv_0 = tf.layers.conv3d(images[0], name='conv0',
                                              filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1],
                                              padding='same',
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                  2.0, seed=SEED),
                                              bias_initializer=tf.zeros_initializer())
                    max_pool_0 = tf.layers.max_pooling3d(conv_0, 2, 2, name='max_pool0')
                    conv_1 = augmented_conv3d('conv1', max_pool_0,
                                              filters=32, kernel_size=[5, 5, 5], dk=4, dv=4, Nh=4,
                                              relative=True, reuse=None)
                    # conv_1 = BatchNorm("conv_1_norm", conv_1)
                    conv_1 = tf.contrib.layers.batch_norm(conv_1, is_training=True, updates_collections=None, decay=0.9)
                    conv_1 = tf.add(conv_1, max_pool_0)  # residual
                    max_pool_1 = tf.layers.max_pooling3d(conv_1, 2, 2, name='max_pool1')
                    # conv_2 = augmented_conv3d('conv2', max_pool_1,
                    #                           filters=64, kernel_size=[4, 4, 4], dk=4, dv=4, Nh=4,
                    #                           relative=True, reuse=None)
                    # conv_2 = BatchNorm("conv_2_norm", inputs=conv_2, axis=0)
                    # conv_2 = tf.contrib.layers.batch_norm(conv_2, is_training=True, updates_collections=None, decay=0.9)
                    # adding here is not possible due to depth 32 -> 64, probably best to use normal conv layer
                    conv_2 = tf.layers.conv3d(max_pool_1, name='conv2', filters=64, kernel_size=[4, 4, 4],
                                              strides=[1, 1, 1], padding='same',
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                seed=SEED),
                                              bias_initializer=tf.zeros_initializer())
                    max_pool_2 = tf.layers.max_pooling3d(conv_2, 2, 2, name='max_pool2')
                    conv_3 = augmented_conv3d('conv3', max_pool_2,
                                              filters=64, kernel_size=[3, 3, 3], dk=4, dv=4, Nh=4,
                                              relative=True, reuse=None)
                    # conv_3 = BatchNorm("conv_3_norm", inputs=conv_3)
                    conv_3 = tf.contrib.layers.batch_norm(conv_3, is_training=True, updates_collections=None, decay=0.9)
                    conv_3 = tf.add(conv_3, max_pool_2)
                    for i in range(0, agents):
                        assert "duel" in str.lower(self.method)
                        fc0 = FullyConnected('fc0V_{}'.format(i), conv_3, 512, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                               seed=SEED))
                        fc1 = FullyConnected('fc1V_{}'.format(i), fc0, 256, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                               seed=SEED))
                        fc2 = FullyConnected('fc2V_{}'.format(i), fc1, 128, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                               seed=SEED))
                        V = FullyConnected('fctV_{}'.format(i), fc2, 1, nl=tf.identity)
                        fcA0 = FullyConnected('fc0A_{}'.format(i), conv_3, 512, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                seed=SEED))
                        fcA1 = FullyConnected('fc1A_{}'.format(i), fcA0, 256, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                seed=SEED))
                        fcA2 = FullyConnected('fc2A_{}'.format(i), fcA1, 128, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                seed=SEED))
                        A = FullyConnected('fctA_{}'.format(i), fcA2, self.num_actions, nl=tf.identity)

                        # TODO: why are we subtracting the mean here? It sure helps!
                        V = tf.subtract(V, tf.reduce_mean(A, 1, keepdims=True), name='Vvalue_{}'.format(i))
                        Q = tf.add(A, V, name='Qvalue_{}'.format(i))

                        Q_list.append(Q)

            else:
                for i in range(0, agents):
                    # normalize image values to [0, 1]; they are normalized to [0, 255] while loading
                    images[i] = images[i] / 255.0
                    with argscope(Conv3D, nl=PReLU.symbolic_function, use_bias=True):

                        if i == 0:
                            conv_0 = tf.layers.conv3d(images[i], name='conv0',
                                                      filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1],
                                                      padding='same',
                                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                          2.0, seed=SEED),
                                                      bias_initializer=tf.zeros_initializer())
                            max_pool_0 = tf.layers.max_pooling3d(conv_0, 2, 2, name='max_pool0')
                            conv_1 = tf.layers.conv3d(max_pool_0, name='conv1',
                                                      filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1],
                                                      padding='same',
                                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                          2.0, seed=SEED),
                                                      bias_initializer=tf.zeros_initializer())
                            max_pool1 = tf.layers.max_pooling3d(conv_1, 2, 2, name='max_pool1')
                            conv_2 = tf.layers.conv3d(max_pool1, name='conv2',
                                                      filters=64, kernel_size=[4, 4, 4], strides=[1, 1, 1],
                                                      padding='same',
                                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                          2.0, seed=SEED),
                                                      bias_initializer=tf.zeros_initializer())
                            max_pool2 = tf.layers.max_pooling3d(conv_2, 2, 2, name='max_pool2')
                            conv_3 = tf.layers.conv3d(max_pool2, name='conv3',
                                                      filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1],
                                                      padding='same',
                                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                          2.0, seed=SEED), bias_initializer=tf.zeros_initializer())
                            if ATTENTION == 20:
                                conv_3_reshaped = tf.reshape(conv_3, (-1, 1, 64))
                                qkv = tf.concat([conv_3_reshaped, conv_3_reshaped, conv_3_reshaped], axis=0)
                                attn = multihead_attention('attn', qkv, num_heads=8)
                                attn_min = tf.reduce_min(attn, name="attn_min")
                                attn_max = tf.reduce_max(attn, name="attn_max")
                                attn_1 = tf.div(tf.subtract(attn, attn_min), tf.subtract(attn_max, attn_min))
                                attn_2 = tf.add(conv_3_reshaped, attn_1, name="attn_added")

                        else:
                            conv_0 = tf.layers.conv3d(images[i], name='conv0', reuse=True,
                                                      filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1],
                                                      padding='same',
                                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                          2.0, seed=SEED),
                                                      bias_initializer=tf.zeros_initializer())
                            max_pool_0 = tf.layers.max_pooling3d(conv_0, 2, 2, name='max_pool0')
                            conv_1 = tf.layers.conv3d(max_pool_0, name='conv1', reuse=True,
                                                      filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1],
                                                      padding='same',
                                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                          2.0, seed=SEED),
                                                      bias_initializer=tf.zeros_initializer())
                            max_pool1 = tf.layers.max_pooling3d(conv_1, 2, 2, name='max_pool1')
                            conv_2 = tf.layers.conv3d(max_pool1, name='conv2', reuse=True,
                                                      filters=64, kernel_size=[4, 4, 4], strides=[1, 1, 1],
                                                      padding='same',
                                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                          2.0, seed=SEED),
                                                      bias_initializer=tf.zeros_initializer())
                            max_pool2 = tf.layers.max_pooling3d(conv_2, 2, 2, name='max_pool2')
                            conv_3 = tf.layers.conv3d(max_pool2, name='conv3', reuse=True,
                                                      filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1],
                                                      padding='same',
                                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                          2.0, seed=SEED), bias_initializer=tf.zeros_initializer())
                            if ATTENTION == 20:
                                conv_3_reshaped = tf.reshape(conv_3, (-1, 1, 64))
                                qkv = tf.concat([conv_3_reshaped, conv_3_reshaped, conv_3_reshaped], axis=0)
                                attn = multihead_attention('attn', qkv, num_heads=8)
                                attn_min = tf.reduce_min(attn, name="attn_min")
                                attn_max = tf.reduce_max(attn, name="attn_max")
                                attn_1 = tf.div(tf.subtract(attn, attn_min), tf.subtract(attn_max, attn_min))
                                attn_2 = tf.add(conv_3_reshaped, attn_1, name="attn_added")
                    ### now for the dense layers##
                    if 'Dueling' not in self.method:
                        fc0 = FullyConnected('fc0_{}'.format(i), conv_3, 512, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                               seed=SEED))
                        fc1 = FullyConnected('fc1_{}'.format(i), fc0, 256, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                               seed=SEED))
                        fc2 = FullyConnected('fc2_{}'.format(i), fc1, 128, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                               seed=SEED))
                        Q = FullyConnected('fct_{}'.format(i), fc2, self.num_actions, nl=tf.identity,
                                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                             seed=SEED))
                        Q_list.append(tf.identity(Q, name='Qvalue_{}'.format(i)))



                    else:
                        if ATTENTION == 6:
                            conv_3_reshaped = tf.reshape(conv_3, (-1, 1, 64))
                            qkv = tf.concat([conv_3_reshaped, conv_3_reshaped, conv_3_reshaped], axis=0)
                            attn = multihead_attention('attn_{}'.format(i), qkv, num_heads=1)
                            attn_min = tf.reduce_min(attn, name="attn_min_{}".format(i))
                            attn_max = tf.reduce_max(attn, name="attn_max_{}".format(i))
                            attn_1 = tf.div(tf.subtract(attn, attn_min), tf.subtract(attn_max, attn_min))
                            attn_2 = tf.add(conv_3_reshaped, attn_1, name="attn_added_{}".format(i))

                            fc0 = FullyConnected('fc0V_{}'.format(i), attn_2, 512, activation=tf.nn.relu,
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                   seed=SEED))
                        elif ATTENTION == 16:
                            conv_3_reshaped = tf.reshape(conv_3, (-1, 1, 64))
                            qkv = tf.concat([conv_3_reshaped, conv_3_reshaped, conv_3_reshaped], axis=0)
                            attn = multihead_attention('attn_{}'.format(i), qkv, num_heads=8)
                            attn_min = tf.reduce_min(attn, name="attn_min_{}".format(i))
                            attn_max = tf.reduce_max(attn, name="attn_max_{}".format(i))
                            attn_1 = tf.div(tf.subtract(attn, attn_min), tf.subtract(attn_max, attn_min))

                            fc0 = FullyConnected('fc0V_{}'.format(i), attn_1, 512, activation=tf.nn.relu,
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                   seed=SEED))
                        elif ATTENTION == 17:
                            conv_3_reshaped = tf.reshape(conv_3, (-1, 1, 64))
                            qkv = tf.concat([conv_3_reshaped, conv_3_reshaped, conv_3_reshaped], axis=0)
                            attn = multihead_attention('attn_{}'.format(i), qkv, num_heads=8)
                            attn_min = tf.reduce_min(attn, name="attn_min_{}".format(i))
                            attn_max = tf.reduce_max(attn, name="attn_max_{}".format(i))
                            attn_1 = tf.div(tf.subtract(attn, attn_min, name="sub_1_{}".format(i)),
                                            tf.subtract(attn_max, attn_min, name="sub_2_{}".format(i)),
                                            name="attn_normalized_{}".format(i))
                            attn_2 = tf.add(conv_3_reshaped, attn_1, name="attn_added_{}".format(i))
                            fc0 = FullyConnected('fc0V_{}'.format(i), attn_2, 512, activation=tf.nn.relu,
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                   seed=SEED))
                        elif ATTENTION == 18:
                            conv_3_reshaped = tf.reshape(conv_3, (-1, 1, 64))
                            qkv = tf.concat([conv_3_reshaped, conv_3_reshaped, conv_3_reshaped], axis=0)
                            attn = multihead_attention('attn_{}'.format(i), qkv, num_heads=8)
                            attn_1 = tf.add(conv_3_reshaped, attn, name="attn_added_{}".format(i))
                            attn_min = tf.reduce_min(attn_1, name="attn_min_{}".format(i))
                            attn_max = tf.reduce_max(attn_1, name="attn_max_{}".format(i))
                            attn_2 = tf.div(tf.subtract(attn_1, attn_min, name="sub_1_{}".format(i)),
                                            tf.subtract(attn_max, attn_min, name="sub_2_{}".format(i)),
                                            name="attn_normalized_{}".format(i))
                            fc0 = FullyConnected('fc0V_{}'.format(i), attn_2, 512, activation=tf.nn.relu,
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                   seed=SEED))
                        elif ATTENTION == 20:
                            fc0 = FullyConnected('fc0V_{}'.format(i), attn_2, 512, activation=tf.nn.relu,
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                   seed=SEED))
                        else:
                            fc0 = FullyConnected('fc0V_{}'.format(i), conv_3, 512, activation=tf.nn.relu,
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                   seed=SEED))
                        fc1 = FullyConnected('fc1V_{}'.format(i), fc0, 256, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                               seed=SEED))
                        fc2 = FullyConnected('fc2V_{}'.format(i), fc1, 128, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                               seed=SEED))
                        V = FullyConnected('fctV_{}'.format(i), fc2, 1, nl=tf.identity)

                        if ATTENTION == 6:
                            fcA0 = FullyConnected('fc0A_{}'.format(i), attn_2, 512, activation=tf.nn.relu,
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                    seed=SEED))
                        elif ATTENTION == 16:
                            fcA0 = FullyConnected('fc0A_{}'.format(i), attn_1, 512, activation=tf.nn.relu,
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                    seed=SEED))
                        elif ATTENTION == 17:
                            fcA0 = FullyConnected('fc0A_{}'.format(i), attn_2, 512, activation=tf.nn.relu,
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                    seed=SEED))
                        elif ATTENTION == 18:
                            fcA0 = FullyConnected('fc0A_{}'.format(i), attn_2, 512, activation=tf.nn.relu,
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                    seed=SEED))
                        elif ATTENTION == 20:
                            fcA0 = FullyConnected('fc0A_{}'.format(i), attn_2, 512, activation=tf.nn.relu,
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                    seed=SEED))
                        else:
                            fcA0 = FullyConnected('fc0A_{}'.format(i), conv_3, 512, activation=tf.nn.relu,
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                    seed=SEED))
                        fcA1 = FullyConnected('fc1A_{}'.format(i), fcA0, 256, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                seed=SEED))
                        fcA2 = FullyConnected('fc2A_{}'.format(i), fcA1, 128, activation=tf.nn.relu,
                                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0,
                                                                                                                seed=SEED))
                        A = FullyConnected('fctA_{}'.format(i), fcA2, self.num_actions, nl=tf.identity)

                        # TODO: why are we subtracting the mean here? It sure helps!
                        V = tf.identity(tf.identity(V) - tf.reduce_mean(A, 1, keepdims=True), name='Vvalue_{}'.format(i))
                        Q = tf.add(A, V, name='Qvalue_{}'.format(i))

                        Q_list.append(Q)

        return Q_list


###############################################################################

def get_config(files_list, input_names=['state_1', 'state_2'],
               output_names=['Qvalue_1', 'Qvalue_2'], agents=2, reward_strategy=1, coords_init=None, mask=None,
               load_model=False):
    """This is only used during training."""
    expreplay = ExpReplay(
        predictor_io_names=(input_names, output_names),
        player=get_player(task='train', files_list=files_list, agents=agents, reward_strategy=reward_strategy,
                          coords_init=coords_init, mask=mask),
        state_shape=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY,
        agents=agents
    )
    if load_model:
        return AutoResumeTrainConfig(
            # dataflow=expreplay,
            data=QueueInput(expreplay),
            model=Model(agents=agents),
            callbacks=[
                ModelSaver(max_to_keep=MAX_TO_KEEP),
                PeriodicTrigger(
                    RunOp(DQNModel.update_target_param, verbose=True),
                    # update target network every 10k steps
                    every_k_steps=10000 // UPDATE_FREQ),
                expreplay,
                ScheduledHyperParamSetter('learning_rate',
                                          [(60, 4e-4), (100, 2e-4)]),
                ScheduledHyperParamSetter(
                    ObjAttrParam(expreplay, 'exploration'),
                    # 1->0.1 in the first million steps
                    [(0, 1), (10, 0.1), (320, 0.01)],
                    interp='linear'),
                PeriodicTrigger(
                    Evaluator(nr_eval=EVAL_EPISODE, input_names=input_names,
                              output_names=output_names, files_list=files_list,
                              get_player_fn=get_player, agents=agents, reward_strategy=reward_strategy),
                    every_k_epochs=EPOCHS_PER_EVAL),
                HumanHyperParamSetter('learning_rate'),
            ],
            steps_per_epoch=STEPS_PER_EPOCH,
            max_epoch=EPOCHS,
        )
    else:
        if ATTENTION == 22:
            # lr = ScheduledHyperParamSetter('learning_rate',
            #                                [(0, 0), (0.05 * EPOCHS, (0.2 * BATCH_SIZE) / 256)])

            def func(epoch_num, old_value):
                # start_epochs = np.linspace(0, 0.05 * EPOCHS, 0.05 * EPOCHS)
                # start_lr = np.linspace(0, (0.2 * BATCH_SIZE) / 256, (0.2 * BATCH_SIZE) / 256)

                start_epochs = [0, 1]
                start_lr = [0, 0.0375]

                min_lr = 0  # TODO: is this correct?
                decayed_learning_rate = min_lr + 0.5 * ((0.2 * BATCH_SIZE) / 256 - min_lr) * (
                        1 + np.cos((epoch_num * np.pi) / EPOCHS))
                # decayed_learning_rate = 0 + 0.5 * ((0.2 * 48) / 256 - 0) * (1 + np.cos((i*np.pi / 20)))
                if epoch_num in start_epochs:
                    return start_lr[epoch_num]
                else:
                    decayed_learning_rate

            lr = HyperParamSetterWithFunc('learning_rate', func)
        elif ATTENTION == 24:
            def func(epoch_num, old_value):
                # start_epochs = np.linspace(0, 0.05 * EPOCHS, 0.05 * EPOCHS)
                # start_lr = np.linspace(0, (0.2 * BATCH_SIZE) / 256, (0.2 * BATCH_SIZE) / 256)

                start_epochs = [0, 1]
                start_lr = [0, 0.0375]

                min_lr = 0  # TODO: is this correct?
                decayed_learning_rate = min_lr + 0.5 * ((0.2 * BATCH_SIZE) / 256 - min_lr) * (
                        1 + np.cos((epoch_num * np.pi) / EPOCHS))
                # decayed_learning_rate = 0 + 0.5 * ((0.2 * 48) / 256 - 0) * (1 + np.cos((i*np.pi / 20)))
                if epoch_num in start_epochs:
                    return start_lr[epoch_num]
                else:
                    return decayed_learning_rate
            print([func(i, 0) for i in range(0,EPOCHS)])
            lr = HyperParamSetterWithFunc('learning_rate', func)
        else:
            lr = ScheduledHyperParamSetter('learning_rate',
                                           [(60, 4e-4), (100, 2e-4)])
        return TrainConfig(
            # dataflow=expreplay,
            data=QueueInput(expreplay),
            model=Model(agents=agents),
            callbacks=[
                ModelSaver(max_to_keep=MAX_TO_KEEP),
                PeriodicTrigger(
                    RunOp(DQNModel.update_target_param, verbose=True),
                    # update target network every 10k steps
                    every_k_steps=10000 // UPDATE_FREQ),
                expreplay,
                lr,
                ScheduledHyperParamSetter(
                    ObjAttrParam(expreplay, 'exploration'),
                    # 1->0.1 in the first million steps
                    [(0, 1), (10, 0.1), (320, 0.01)],
                    interp='linear'),
                PeriodicTrigger(
                    Evaluator(nr_eval=EVAL_EPISODE, input_names=input_names,
                              output_names=output_names, files_list=files_list,
                              get_player_fn=get_player, agents=agents, reward_strategy=reward_strategy),
                    every_k_epochs=EPOCHS_PER_EVAL),
                HumanHyperParamSetter('learning_rate'),
            ],
            steps_per_epoch=STEPS_PER_EPOCH,
            max_epoch=EPOCHS,
        )


###############################################################################
###############################################################################
# TODO: Verify that the init point box for eval is centered around the landmark pos from training
# TODO: Attention
# TODO: Prepend U-Net for automatic masking
# TODO: fix visualization

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='filepath to model for loading')
    parser.add_argument('--task', help='task to perform. Must load a pretrained model if task is "play" or "eval"',
                        choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--test_all_acc', action='store_true',
                        help='use this flag when using eval mode to generate accuracies for all saved training '
                             'checkpoints, not just the last one. In the load flag specify the path to the checkpoints '
                             'file not to an individual model checkpoint. Results are saved as a tensorboard log file')
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling', 'DuelingDouble'],
                        default='DQN')
    parser.add_argument('--files', type=argparse.FileType('r'), nargs='+',
                        default=("/Landmarks_RL_Data/Training_Data/data_AAE/filenames/image_files.txt",
                                 "/Landmarks_RL_Data/Training_Data/data_AAE/filenames/landmark_files.txt"),
                        help="""Filepath to the text file that contains list of images.
                                Each line of this file is a full path to an image scan.
                                For (task == train or eval) there should be two input files ['images', 'landmarks']""")
    parser.add_argument('--saveGif', help='CURRENTLY NOT WORKING save gif image of the game',
                        action='store_true', default=False)
    parser.add_argument('--saveVideo', help='CURRENTLY NOT WORKING save video of the game',
                        action='store_true', default=False)
    parser.add_argument('--logDir', help='store logs in this directory during training',
                        default='train_log')
    parser.add_argument('--testDir', help='store eval logs in this directory during eval', default='test_log')
    parser.add_argument('--configDir',
                        help='name of directory in which to look when specifying a config to load (see --load_config)',
                        default='configs')
    parser.add_argument('--mask_roi', help='path to file with roi mask (1 where ok to go 0 otherwise)')
    parser.add_argument('--coords_init', help="""Path to the text file that contains the paths to the init coords
                        Each line is a full path to a .npy init coord file. 
                        The ordering is exactly the same as in the argument provided to --files 
                        When using tum_box directly specify path of initial coordinates dont put it in a txt file
                        Defaults:
                            - training: point_box
                            - eval: point_box""",
                        default=None)
    parser.add_argument('--name', help='name of current experiment for logs',
                        default='dev')
    parser.add_argument('--agents', help='Number of agents to train together', default=2)
    parser.add_argument('--reward_strategy',
                        help='Which reward strategy you want? 1 is simple, 2 is line based, 3 is agent based',
                        default=1)
    parser.add_argument('--load_config',
                        help='specify the path of a config file relative to the configDir (default: configs/)',
                        default=None)

    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    coords_init = args.coords_init

    if args.load_config is not None:
        if args.load_config.endswith(".py"):
            args.load_config = args.load_config[:-3]
        assert (args.load_config + ".py") in os.listdir(args.configDir)
        # do not use filepaths, use path syntax as with importing libraries
        path_to_config_module = args.configDir + "." + args.load_config
        BATCH_SIZE, IMAGE_SIZE, FRAME_HISTORY, UPDATE_FREQ, GAMMA, MEMORY_SIZE, INIT_MEMORY_SIZE, STEPS_PER_EPOCH, \
        EPOCHS_PER_EVAL, EVAL_EPISODE, EPOCHS, STEPS_PER_EVAL, MAX_NUM_FRAMES, MAX_TO_KEEP, MULTISCALE, \
        ATTENTION, SEED = importlib.import_module(path_to_config_module).get_hyperparams()
        logger.info("succesfully loaded configs from {}".format(os.path.join(args.configDir, args.load_config + ".py")))

    logger.info("ATTENTION = {}".format(ATTENTION))

    # theres some multithreaded stuff in common.py, but changing # of threads to 1 didnt seem to help with reproducibility
    # still not reproducible :(
    np.random.seed(SEED)
    random.seed(SEED)
    fix_rng_seed(SEED)
    tf.set_random_seed(SEED)

    # enforce naming convention
    if args.name is not "dev" and args.task == "train":
        assert int(args.name.split("_")[1]) >= STEPS_PER_EPOCH * UPDATE_FREQ and int(args.name.split("_")[1]) <= (
                1 + STEPS_PER_EPOCH) * UPDATE_FREQ and int(args.name.split("_")[2]) == EPOCHS

    # check input files
    if args.task == 'play':
        error_message = """Wrong input files {} for {} task - should be 1 \'images.txt\' """.format(len(args.files),
                                                                                                    args.task)
        assert len(args.files) == 1
    else:
        error_message = """Wrong input files {} for {} task - should be 2 [\'images.txt\', \'landmarks.txt\'] """.format(
            len(args.files), args.task)
        assert len(args.files) == 2, (error_message)

    args.agents = int(args.agents)

    METHOD = args.algo
    # load files into env to set num_actions, num_validation_files
    init_player = MedicalPlayer(files_list=args.files,
                                screen_dims=IMAGE_SIZE,
                                task=args.task,
                                agents=args.agents, multiscale=MULTISCALE, reward_strategy=args.reward_strategy,
                                coords_init=args.coords_init, mask=args.mask_roi, seed=SEED)
    NUM_ACTIONS = init_player.action_space.n
    num_files = init_player.files.num_files

    ##########################################################
    # initialize states and Qvalues for the various agents
    state_names = []
    qvalue_names = []
    vvalue_names = []
    for i in range(0, args.agents):
        state_names.append('state_{}'.format(i))
        qvalue_names.append('Qvalue_{}'.format(i))
    for i in range(0, args.agents):
        qvalue_names.append('Vvalue_{}'.format(i))

    ############################################################

    if args.task != 'train':
        assert args.load is not None
        if args.task == 'eval' and args.test_all_acc:
            checkpoints = []
            mean_dists = []
            dists = []
            test_steps = []

            fieldnames = ["step", "agent", "mean_dist", "var_dist"] + ["dist_{}".format(i) for i in
                                                                       range(0, STEPS_PER_EVAL)]
            rows = []

            pat_name = args.name[:3]
            try:
                allready_tested = [args.name == tested_pat for tested_pat in os.listdir(args.testDir)]
            except:
                allready_tested = [False]
            if any(allready_tested):
                rmdir = os.path.join(args.testDir, os.listdir(args.testDir)[allready_tested.index(True)])
                print("removing {}".format(rmdir))
                shutil.rmtree(rmdir)

            with open(args.load, 'r') as fh:
                writer = SummaryWriter(logdir=os.path.join(args.testDir, args.name))
                for idx, line in enumerate(fh):
                    if idx > 0:
                        checkpoints.append(os.path.join(args.logDir, args.name, str.split(line)[1][1:-1]))
                        try:
                            test_step = int(str.split(line)[1][1:-1].replace('model-', ''))
                        except:
                            print("Problem fetching the correct step nr from checkpoints, defaulting to 0")
                            test_step = 0
                        pred = OfflinePredictor(PredictConfig(
                            model=Model(agents=args.agents),
                            session_init=get_model_loader(checkpoints[idx - 1]),
                            input_names=state_names,
                            output_names=qvalue_names))

                        mean_distances, var_distances, distances = play_n_episodes(
                            player=get_player(files_list=args.files, viz=0.01,
                                              saveGif=args.saveGif,
                                              saveVideo=args.saveVideo,
                                              task='eval', agents=args.agents,
                                              reward_strategy=args.reward_strategy,
                                              coords_init=coords_init,
                                              mask=args.mask_roi),
                            predfunc=pred, nr=STEPS_PER_EVAL, agents=args.agents,
                            test_all_acc=True)

                        mean_dists.append(mean_distances)
                        dists.append(distances)
                        test_steps.append(test_step)
                        for i in range(args.agents):
                            writer.add_scalar('expreplay/mean_dist_{}'.format(i), mean_dists[-1][i], test_step)
                            rows.append([test_step, i, mean_distances[i], var_distances[i]] + list(distances[i]))
                writer.close()

                with open(os.path.join(args.testDir, args.name, args.name + ".csv"), "w+", newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(fieldnames)
                    csv_writer.writerows(rows)

                print("finished saving test accuracy to {}".format(os.path.join(args.testDir, args.name)))

        else:
            pred = OfflinePredictor(PredictConfig(
                model=Model(agents=args.agents),
                session_init=get_model_loader(args.load),
                input_names=state_names,
                output_names=qvalue_names))
            # demo pretrained model one episode at a time
            if args.task == 'play':
                play_n_episodes(get_player(files_list=args.files, viz=0.01,
                                           saveGif=args.saveGif,
                                           saveVideo=args.saveVideo,
                                           task='play', agents=args.agents, reward_strategy=args.reward_strategy,
                                           coords_init=coords_init, mask=args.mask_roi),
                                pred, num_files)
            # run episodes in parallel and evaluate pretrained model
            elif args.task == 'eval':
                logger_dir = os.path.join(args.testDir, args.name)
                logger.set_logger_dir(logger_dir)
                play_n_episodes(player=get_player(files_list=args.files, viz=0.01,
                                                  saveGif=args.saveGif,
                                                  saveVideo=args.saveVideo,
                                                  task='eval', agents=args.agents, reward_strategy=args.reward_strategy,
                                                  coords_init=coords_init, mask=args.mask_roi),
                                predfunc=pred, nr=STEPS_PER_EVAL, agents=args.agents)
    else:  # train model
        logger_dir = os.path.join(args.logDir, args.name)
        logger.set_logger_dir(logger_dir)
        config = get_config(args.files, input_names=state_names,
                            output_names=qvalue_names, agents=args.agents, reward_strategy=args.reward_strategy,
                            coords_init=coords_init, mask=args.mask_roi, load_model=args.load)
        # if args.load:  # resume training from a saved checkpoint
        # config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SimpleTrainer())
