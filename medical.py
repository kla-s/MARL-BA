#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: medical.py
# Author: Amir Alansary <amiralansary@gmail.com>
# Modified: Athanasios Vlontzos <athanasiosvlontzos@gmail.com>

import csv
import itertools


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
import copy
import os
import sys
import six
import random
import threading
import numpy as np
from tensorpack import logger
from collections import (Counter, defaultdict, deque, namedtuple)

import cv2
import math
import time
from PIL import Image
import subprocess
import shutil

import gym
from gym import spaces

# try:
#     import pyglet
# except ImportError as e:
#     reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'.")

from tensorpack.utils.utils import get_rng
from tensorpack.utils.stats import StatCounter

from IPython.core.debugger import set_trace
from dataReader import *

_ALE_LOCK = threading.Lock()

Rectangle = namedtuple('Rectangle', ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'])


# ===================================================================
# =================== 3d medical environment ========================
# ===================================================================

class MedicalPlayer(gym.Env):
    """Class that provides 3D medical image environment.
    This is just an implementation of the classic "agent-environment loop".
    Each time-step, the agent chooses an action, and the environment returns
    an observation and a reward."""

    def __init__(self, directory=None, viz=False, task=False, files_list=None,
                 screen_dims=(27, 27, 27), history_length=20, multiscale=True,
                 max_num_frames=0, saveGif=False, saveVideo=False, agents=2, reward_strategy=1, coords_init=None,
                 mask=None, seed=0):
        """
        :param train_directory: environment or game name
        :param viz: visualization
            set to 0 to disable
            set to +ve number to be the delay between frames to show
            set to a string to be the directory for storing frames
        :param screen_dims: shape of the frame cropped from the image to feed
            it to dqn (d,w,h) - defaults (27,27,27)
        :param nullop_start: start with random number of null ops
        :param location_history_length: consider lost of lives as end of
            episode (useful for training)
        :max_num_frames: maximum numbe0r of frames per episode.
            Diana also uses history length = 20
        """
        self.csvfile = 'test.csv'
        self.reward_strategy = reward_strategy

        if task != 'train':
            with open(self.csvfile, 'w') as outcsv:
                fields = ["filename", "dist_error"]
                writer = csv.writer(outcsv)
                writer.writerow(map(lambda x: x, fields))

        self.coords_init = coords_init  # will be set in _restart_episode

        self.total_loc = []
        ######################################################################

        super(MedicalPlayer, self).__init__()

        self.seed(seed)
        # number of agents
        self.agents = agents

        # inits stat counters
        self.reset_stat()

        # counter to limit number of steps per episodes
        self.cnt = 0
        # maximum number of frames (steps) per episodes
        self.max_num_frames = max_num_frames
        # stores information: terminal, score, distError
        self.info = None
        # option to save display as gif
        self.saveGif = saveGif
        self.saveVideo = saveVideo
        # training flag
        self.task = task
        # image dimension (2D/3D)
        self.screen_dims = screen_dims
        self.dims = len(self.screen_dims)
        # multi-scale agent
        self.multiscale = multiscale

        # init env dimensions
        if self.dims == 2:
            self.width, self.height = screen_dims
        else:
            self.width, self.height, self.depth = screen_dims

        # get action space and minimal action set
        self.action_space = spaces.Discrete(6)  # change number actions here
        self.actions = self.action_space.n
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=self.screen_dims,
                                            dtype=np.uint8)
        # history buffer for storing last locations to check oscilations
        self._history_length = history_length
        self._loc_history = []
        self._qvalues_history = []
        self._qvalue_history = []
        self._vvalues_history = []
        self._state_values = None
        # stat counter to store current score or accumulated reward
        self.current_episode_score = []
        self.rectangle = []
        for i in range(0, self.agents):
            self.current_episode_score.append(StatCounter())
            self._loc_history.append([(0,) * self.dims] * self._history_length)
            self._qvalues_history.append([(0,) * self.actions] * self._history_length)
            self._qvalue_history.append([0] * self._history_length)
            self._vvalues_history.append([0] * self._history_length)
            self.rectangle.append(
                Rectangle(0, 0, 0, 0, 0, 0))  # initialize rectangle limits from input image coordinates

        # add your data loader here
        if self.task == 'play':
            self.files = filesListBrainMRLandmark(files_list,
                                                  returnLandmarks=False,
                                                  agents=self.agents)
        else:
            self.files = filesListBrainMRLandmark(files_list,
                                                  returnLandmarks=True,
                                                  agents=self.agents)

        if mask is not None:
            self.mask = np.load(mask)
        else:
            self.mask = np.ones(self.files.image_dims)

        # prepare file sampler
        self.filepath = None
        self.sampled_files = self.files.sample_circular()
        # reset buffer, terminal, counters, and init new_random_game
        self._restart_episode()

    def reset(self):
        # with _ALE_LOCK:
        self._restart_episode()
        return self._current_state()

    def _restart_episode(self):
        """
        restart current episoide
        """
        self.terminal = [False] * self.agents
        self.reward = np.zeros((self.agents,))
        self.cnt = 0  # counter to limit number of steps per episodes
        self.num_games.feed(1)
        self._loc_history = []
        self._qvalues_history = []
        self._qvalue_history = []
        self._vvalues_history = []
        for i in range(0, self.agents):
            self.current_episode_score[i].reset()

            self._loc_history.append([(0,) * self.dims] * self._history_length)
            # list of q-value lists
            self._qvalues_history.append([(0,) * self.actions] * self._history_length)
            self._qvalue_history.append([0] * self._history_length)
            self._vvalues_history.append([0] * self._history_length)
        self.new_random_game()

    def new_random_game(self):
        """
        load image,
        set dimensions,
        randomize start point,
        init _screen, qvals,
        calc distance to goal
        """
        self.terminal = [False] * self.agents

        self.viewer = None

        self._image, self._target_loc, self.filepath, self.spacing = next(self.sampled_files)
        ###################### calculate distance of all the landmarks ################################
        combi = itertools.combinations(range(self.agents), 2)
        combi = list(combi)
        self.all_distances = np.zeros((self.agents, self.agents))
        for a, b in combi:
            self.all_distances[a, b] = self.calcDistance(self._target_loc[a], self._target_loc[b], self.spacing)
            self.all_distances[b, a] = self.all_distances[a, b]
        #######################################################################

        self.filename = []
        # # sample a new image
        self._image, self._target_loc, self.filepath, self.spacing = next(self.sampled_files)
        for i in range(0, self.agents):
            self.filename.append(os.path.basename(self.filepath[i]))
        # multiscale (e.g. start with 3 -> 2 -> 1)
        # scale can be thought of as sampling stride
        if self.multiscale:
            ## brain
            self.action_steps = [9 for i in range(self.agents)]
            self.xscales = [3 for i in range(self.agents)]
            self.yscales = [3 for i in range(self.agents)]
            self.zscales = [3 for i in range(self.agents)]
            ## cardiac
            # self.action_step =   6
            # self.xscale = 2
            # self.yscale = 2
            # self.zscale = 2
        else:
            self.action_steps = [3 for i in range(self.agents)]
            self.xscales = [3 for i in range(self.agents)]
            self.yscales = [3 for i in range(self.agents)]
            self.zscales = [3 for i in range(self.agents)]
        # image volume size
        self._image_dims = self._image[0].dims

        ## select random starting point
        # add padding to avoid start right on the border of the image

        #######################################################################

        self._location = []
        self._start_location = []
        self.start_points = []
        if self.coords_init is not None:
            if self.coords_init.endswith(".txt"):

                init_coords = getInitCoords(self.coords_init)
                if init_coords[0].shape == (3,):
                    # when using train_points
                    self.start_points = init_coords
                else:
                    # when using point boxes
                    for i in range(0, self.agents):
                        start_point = init_coords[i][np.random.choice(init_coords[0].shape[0])][:]
                        while self.mask[tuple(start_point)] == 0:  # mask is initialized with 1s when not provided
                            start_point = init_coords[i][np.random.choice(init_coords[0].shape[0])][:]
                        self.start_points.append(tuple(start_point))
            else:
                # when using tumor boxes
                for i in range(0, self.agents):
                    start_point = random.choice(np.load(self.coords_init))
                    while self.mask[tuple(start_point)] == 0:
                        start_point = random.choice(np.load(self.coords_init))
                    self.start_points.append(tuple(start_point))

                # when using tum_box directly specify path of initial coordinates dont put it in a txt file
                # self.start_points = [np.load(coords_init)[np.random.choice(np.load(coords_init).shape[0])]
                #                     for i in range(agents)]
        else:
            for i in range(0, self.agents):
                start_point = []
                for j in self._image_dims:
                    x = np.random.randint(0, j)
                    start_point.append(x)
                while self.mask[tuple(start_point)] == 0:
                    start_point = []
                    for j in self._image_dims:
                        x = np.random.randint(0, j)
                        start_point.append(x)

                self.start_points.append(tuple(start_point))

        self._location = self.start_points
        self._start_location = self.start_points

        self._qvalues = [[0, ] * self.actions] * self.agents
        self._screen = self._current_state()

        if self.task == 'play':
            self.cur_dist = [0, ] * self.agents
        else:
            self.cur_dist = []
            for i in range(0, self.agents):
                self.cur_dist.append(self.calcDistance(self._location[i],
                                                       self._target_loc[i],
                                                       self.spacing))

    def calcDistance(self, points1, points2, spacing=(1, 1, 1)):
        """ calculate the distance between two points in mm"""
        spacing = np.array(spacing)
        points1 = spacing * np.array(points1)
        points2 = spacing * np.array(points2)
        return np.linalg.norm(points1 - points2)

    def step(self, act, q_values, isOver, state_values=None):
        """The environment's step function returns exactly what we need.
        Args:
          act:
        Returns:
          observation (object):
            an environment-specific object representing your observation of
            the environment. For example, pixel data from a camera, joint angles
            and joint velocities of a robot, or the board state in a board game.
          reward (float):
            amount of reward achieved by the previous action. The scale varies
            between environments, but the goal is always to increase your total
            reward.
          done (boolean):
            whether it's time to reset the environment again. Most (but not all)
            tasks are divided up into well-defined episodes, and done being True
            indicates the episode has terminated. (For example, perhaps the pole
            tipped too far, or you lost your last life.)
          info (dict):
            diagnostic information useful for debugging. It can sometimes be
            useful for learning (for example, it might contain the raw
            probabilities behind the environment's last state change). However,
            official evaluations of your agent are not allowed to use this for
            learning.
        """
        for i in range(0, self.agents):
            if isOver[i]: act[i] = 10
        self._qvalues = q_values
        if state_values is not None:
            self._state_values = np.stack(state_values).flatten().tolist()

        self.act = act
        current_loc = self._location
        next_location = copy.deepcopy(current_loc)

        self.terminal = [False] * self.agents
        go_out = out_mask = [False] * self.agents

        ######################## agent i movement #############################
        for i in range(0, self.agents):
            # UP Z+ -----------------------------------------------------------
            if (act[i] == 0):
                next_location[i] = (current_loc[i][0],
                                    current_loc[i][1],
                                    round(current_loc[i][2] + self.action_steps[i]))
                if (next_location[i][2] >= self._image_dims[2]):
                    # print(' trying to go out the image Z+ ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
                if self.mask[next_location[i]] == 0:
                    out_mask[i] = True

            # FORWARD Y+ ---------------------------------------------------------
            if (act[i] == 1):
                next_location[i] = (current_loc[i][0],
                                    round(current_loc[i][1] + self.action_steps[i]),
                                    current_loc[i][2])
                if (next_location[i][1] >= self._image_dims[1]):
                    # print(' trying to go out the image Y+ ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
                if self.mask[next_location[i]] == 0:
                    out_mask[i] = True
            # RIGHT X+ -----------------------------------------------------------
            if (act[i] == 2):
                next_location[i] = (round(current_loc[i][0] + self.action_steps[i]),
                                    current_loc[i][1],
                                    current_loc[i][2])
                if next_location[i][0] >= self._image_dims[0]:
                    # print(' trying to go out the image X+ ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
                if self.mask[next_location[i]] == 0:
                    out_mask[i] = True
            # LEFT X- -----------------------------------------------------------
            if act[i] == 3:
                next_location[i] = (round(current_loc[i][0] - self.action_steps[i]),
                                    current_loc[i][1],
                                    current_loc[i][2])
                if next_location[i][0] <= 0:
                    # print(' trying to go out the image X- ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
                if self.mask[next_location[i]] == 0:
                    out_mask[i] = True
            # BACKWARD Y- ---------------------------------------------------------
            if act[i] == 4:
                next_location[i] = (current_loc[i][0],
                                    round(current_loc[i][1] - self.action_steps[i]),
                                    current_loc[i][2])
                if next_location[i][1] <= 0:
                    # print(' trying to go out the image Y- ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
                if self.mask[next_location[i]] == 0:
                    out_mask[i] = True
            # DOWN Z- -----------------------------------------------------------
            if act[i] == 5:
                next_location[i] = (current_loc[i][0],
                                    current_loc[i][1],
                                    round(current_loc[i][2] - self.action_steps[i]))
                if next_location[i][2] <= 0:
                    # print(' trying to go out the image Z- ',)
                    next_location[i] = current_loc[i]
                    go_out[i] = True
                if self.mask[next_location[i]] == 0:
                    out_mask[i] = True
            # -----------------------------------------------------------------

        #######################################################################

        # punish -1 reward if the agent tries to go out
        if (self.task != 'play'):
            for i in range(0, self.agents):
                if go_out[i] or out_mask[i]:
                    self.reward[i] = -1
                else:
                    # if self.task=='train' or self.task=='eval':
                    if self.reward_strategy == 1:
                        self.reward[i] = self._calc_reward(current_loc[i], next_location[i], agent=i)
                    elif self.reward_strategy == 2:
                        self.reward[i] = self._calc_reward_geometric(current_loc[i], next_location[i], agent=i)
                    elif self.reward_strategy == 3:
                        self.reward[i] = self._distance_to_other_agents(current_loc, next_location, agent=i)
                    elif self.reward_strategy == 4:
                        self.reward[i] = self._distance_to_other_agents_and_line(current_loc, next_location, agent=i)
                    elif self.reward_strategy == 5:
                        self.reward[i] = self._distance_to_other_agents_and_line_no_point(current_loc, next_location,
                                                                                          agent=i)
                    elif self.reward_strategy == 6:
                        self.reward[i] = self._calc_reward_geometric_generalized(current_loc[i], next_location[i],
                                                                                 agent=i)
                    # else:
                    #     self.reward[i]= self._calc_reward(current_loc[i], next_location[i],agent=i)

        # update screen, reward ,location, terminal
        self._location = next_location
        self._screen = self._current_state()

        # terminate if the distance is 1 or smaller during trainig
        if self.task == 'train':
            for i in range(0, self.agents):
                if self.cur_dist[i] < 1:
                    self.terminal[i] = True
                    self.num_success[i].feed(1)

        # terminate if maximum number of steps is reached
        # TODO: consider getting the best position in this case
        self.cnt += 1
        if self.cnt >= self.max_num_frames:
            for i in range(0, self.agents):
                self._location = self.getBestLocation(i)
                self._screen = self._current_state()
                self.terminal[i] = True
                # print("terminating bc of max num frames, agent: {}, goal: {}, curr: {}".format(i, self._target_loc[i],
                #                                                                                self._location[i]))

        # update history buffer with new location and qvalues
        if self.task != 'play':
            for i in range(0, self.agents):
                self.cur_dist[i] = self.calcDistance(self._location[i],
                                                     self._target_loc[i],
                                                     self.spacing)

        self._update_history()

        # check if agent oscillates
        for i in range(self.agents):
            if self._oscillate(i):
                self._location = self.getBestLocation(i)
                self._screen = self._current_state()

                if (self.task != 'play'):
                    self.cur_dist[i] = self.calcDistance(self._location[i],
                                                         self._target_loc[i],
                                                         self.spacing)

                # multi-scale steps
                if self.multiscale:
                    if self.xscales[i] > 1:
                        self.xscales[i] -= 1
                        self.yscales[i] -= 1
                        self.zscales[i] -= 1
                        self.action_steps[i] = int(self.action_steps[i] / 3)
                        self._clear_history(i)
                    # terminate if scale is 1 and agent oscillates
                    else:
                        self.terminal[i] = True
                        if self.cur_dist[i] <= 1:
                            self.num_success[i].feed(1)
                        # print("terminating bc of scale 1 & osci, agent: {}, goal: {}, curr: {}".format(i,
                        #                                                                                self._target_loc[
                        #                                                                                    i],
                        #                                                                                self._location[
                        #                                                                                    i]))
                else:
                    self.terminal[i] = True
                    if self.cur_dist[i] <= 1:
                        self.num_success[i].feed(1)

        distance_error = self.cur_dist
        for i in range(0, self.agents):
            self.current_episode_score[i].feed(self.reward[i])

        info = {}
        for i in range(0, self.agents):
            info['score_{}'.format(i)] = self.current_episode_score[i].sum
            info['gameOver_{}'.format(i)] = self.terminal[i]
            info['distError_{}'.format(i)] = distance_error[i]
            info['filename_{}'.format(i)] = self.filename[i]

        return self._current_state(), self.reward, self.terminal, info

    def getBestLocation(self, agent):
        ''' get best location for an agent based on qvalues from the history
        '''
        # TODO: why is the score better the smaller q or v values are not the greater
        # TODO: check idea: combine all 4 infos (min mean q values, freq, min v values, min q value ), if one state is
        #  ahead in 3 metrics choose it, then 2..., else choose based upon greatest advantage in metric weighted by smth
        best_location = self._location
        # if self.action_steps[agent] == 1:
        #     print("joa ne")
        #     [np.mean(x) for x in self._qvalues_history[agent]], [self.calcDistance(self._target_loc[agent], y) for y in
        #                                                          self._loc_history[agent]], np.argmin(
        #         [self.calcDistance(self._target_loc[agent], y) for y in self._loc_history[agent]]), \
        #     [np.mean(x) for x in self._qvalues_history[agent]][
        #         np.argmin([self.calcDistance(self._target_loc[agent], y) for y in self._loc_history[agent]])]

        # 5 possibilities using q values sorted from best to worst:
        #   best location = loc with the min of the mean of q values of the surrounding positions
        #   best location = loc with the min q value of the surrounding positions
        #   best location = loc with the max of the mean of absolute q values of the surrounding positions
        #   best location = loc with the min q value of the positions in the history
        #   best location = loc with the max q value of the positions in the history
        # previously only the last 4 locations/qvalues where looked at here but this doesnt help
        # if one really wants this idea then consider using the location visited most often (last approach)

        # best_idx = np.argmin(np.mean(self._qvalues_history, axis=2)[agent]) # doesnt work sometimes bc of some threading bs
        best_idx = np.argmin([np.mean(x) for x in self._qvalues_history[agent]])  # works more often

        # best_idx = int(np.argmin(self._qvalues_history[agent])/6)

        # best_idx = np.argmax([np.mean(tuple(map(abs, x))) for x in self._qvalues_history[agent]])

        # best_idx = np.argmin(self._qvalue_history[agent])

        # best_idx = np.argmax(self._qvalue_history[agent])

        # TODO: make sure this is only used with Dueling DQN
        # choose best location based on state value function
        # best_idx = np.argmin(self._vvalues_history[agent])
        # best_idx = np.argmax(self._vvalues_history[agent]) #bad idea

        best_location[agent] = self._loc_history[agent][best_idx]

        # choose best location based upon which location was visited most often in the history
        # counter = Counter(self._loc_history[agent])
        # freq = counter.most_common()
        # if freq[0][0] == (0, 0, 0):
        #     best_location[agent] = freq[1][0]
        # else:
        #     best_location[agent] = freq[0][0]


        # last_qvalues_history = self._qvalues_history[agent][-4:]
        # last_loc_history = self._loc_history[agent][-4:]
        # leads to:
        # Exception in thread SimulatorThread:
        # Traceback (most recent call last):
        #   File "/usr/lib/python3.6/threading.py", line 916, in _bootstrap_inner
        #     self.run()
        #   File "/usr/local/lib/python3.6/dist-packages/tensorpack/utils/concurrency.py", line 143, in run
        #     self._th.run()
        #   File "/usr/local/lib/python3.6/dist-packages/tensorpack/utils/concurrency.py", line 96, in run
        #     self._func()
        #   File "/MARL-BA/expreplay.py", line 209, in populate_job_func
        #     self._populate_exp()
        #   File "/MARL-BA/expreplay.py", line 266, in _populate_exp
        #     current_state, reward, isOver, info = self.player.step(act, q_values, isOver)
        #   File "/MARL-BA/medical.py", line 483, in step
        #     self._location = self.getBestLocation(i)
        #   File "/MARL-BA/medical.py", line 576, in getBestLocation
        #     best_qvalues = np.max(last_qvalues_history, axis=1)
        #   File "/usr/local/lib/python3.6/dist-packages/numpy/core/fromnumeric.py", line 2320, in amax
        #     out=out, **kwargs)
        #   File "/usr/local/lib/python3.6/dist-packages/numpy/core/_methods.py", line 26, in _amax
        #     return umr_maximum(a, axis, None, out, keepdims)
        # numpy.core._internal.AxisError: axis 1 is out of bounds for array of dimension 1

        # best_idx = np.argmin([np.max(x) for x in last_qvalues_history])
        # best_location[agent] = last_loc_history[best_idx]

        return best_location

    def _clear_history(self, agent):
        ''' clear history buffers of previous states and qvalues for a given agent
        '''

        self._loc_history[agent] = [(0,) * self.dims] * self._history_length
        self._qvalues_history[agent] = [(0,) * self.actions] * self._history_length
        self._qvalue_history[agent] = ([0] * self._history_length)
        self._vvalues_history[agent] = ([0] * self._history_length)

    def _update_history(self):
        ''' update history buffer with current state
        '''
        # update location history
        for i in range(0, self.agents):
            self._loc_history[i][:-1] = self._loc_history[i][1:]
            self._loc_history[i][-1] = self._location[i]

            # update q-value history
            self._qvalues_history[i][:-1] = self._qvalues_history[i][1:]
            self._qvalues_history[i][-1] = self._qvalues[i]

            self._qvalue_history[i][:-1] = self._qvalue_history[i][1:]
            direction = int(self.act[i])
            if direction in range(0, len(self._qvalues[i])):
                self._qvalue_history[i][-1] = self._qvalues[i][direction]
            else:
                self._qvalue_history[i][-1] = 0

            # update state value history
            self._vvalues_history[i][:-1] = self._vvalues_history[i][1:]
            if self._state_values is not None:
                self._vvalues_history[i][-1] = self._state_values[i]
            else:
                self._vvalues_history[i][-1] = 0

    def _current_state(self):
        """
        crop image data around current location to update what network sees.
        update rectangle

        :return: new state
        """
        # initialize screen with zeros - all background
        screen = np.zeros((self.agents, self.screen_dims[0], self.screen_dims[1], self.screen_dims[2])).astype(
            self._image[0].data.dtype)

        for i in range(0, self.agents):
            # screen uses coordinate system relative to origin (0, 0, 0)
            screen_xmin, screen_ymin, screen_zmin = 0, 0, 0
            screen_xmax, screen_ymax, screen_zmax = self.screen_dims

            # extract boundary locations using coordinate system relative to "global" image
            # width, height, depth in terms of screen coord system

            if self.xscales[i] % 2:
                xmin = self._location[i][0] - int(self.width * self.xscales[i] / 2) - 1
                xmax = self._location[i][0] + int(self.width * self.xscales[i] / 2)
                ymin = self._location[i][1] - int(self.height * self.yscales[i] / 2) - 1
                ymax = self._location[i][1] + int(self.height * self.yscales[i] / 2)
                zmin = self._location[i][2] - int(self.depth * self.zscales[i] / 2) - 1
                zmax = self._location[i][2] + int(self.depth * self.zscales[i] / 2)
            else:
                xmin = self._location[i][0] - round(self.width * self.xscales[i] / 2)
                xmax = self._location[i][0] + round(self.width * self.xscales[i] / 2)
                ymin = self._location[i][1] - round(self.height * self.yscales[i] / 2)
                ymax = self._location[i][1] + round(self.height * self.yscales[i] / 2)
                zmin = self._location[i][2] - round(self.depth * self.zscales[i] / 2)
                zmax = self._location[i][2] + round(self.depth * self.zscales[i] / 2)

            ###########################################################

            # check if they violate image boundary and fix it
            if xmin < 0:
                xmin = 0
                screen_xmin = screen_xmax - len(np.arange(xmin, xmax, self.xscales[i]))
            if ymin < 0:
                ymin = 0
                screen_ymin = screen_ymax - len(np.arange(ymin, ymax, self.yscales[i]))
            if zmin < 0:
                zmin = 0
                screen_zmin = screen_zmax - len(np.arange(zmin, zmax, self.zscales[i]))
            if xmax > self._image_dims[0]:
                xmax = self._image_dims[0]
                screen_xmax = screen_xmin + len(np.arange(xmin, xmax, self.xscales[i]))
            if ymax > self._image_dims[1]:
                ymax = self._image_dims[1]
                screen_ymax = screen_ymin + len(np.arange(ymin, ymax, self.yscales[i]))
            if zmax > self._image_dims[2]:
                zmax = self._image_dims[2]
                screen_zmax = screen_zmin + len(np.arange(zmin, zmax, self.zscales[i]))

            # crop image data to update what network sees
            # image coordinate system becomes screen coordinates
            # scale can be thought of as a stride
            screen[i, screen_xmin:screen_xmax, screen_ymin:screen_ymax, screen_zmin:screen_zmax] = self._image[i].data[
                                                                                                   xmin:xmax:
                                                                                                   self.xscales[i],
                                                                                                   ymin:ymax:
                                                                                                   self.yscales[i],
                                                                                                   zmin:zmax:
                                                                                                   self.zscales[i]]

            ###########################################################
            # update rectangle limits from input image coordinates
            # this is what the network sees
            self.rectangle = Rectangle(xmin, xmax,
                                       ymin, ymax,
                                       zmin, zmax)

        return screen

    def get_plane(self, z=0, agent=0):
        return self._image[agent].data[:, :, z]

    def _calc_reward(self, current_loc, next_loc, agent):
        """ Calculate the new reward based on the decrease in euclidean distance to the target location
        """

        curr_dist = self.calcDistance(current_loc, self._target_loc[agent],
                                      self.spacing)
        next_dist = self.calcDistance(next_loc, self._target_loc[agent],
                                      self.spacing)
        dist = curr_dist - next_dist

        return dist

    def _calc_reward_geometric(self, current_loc, next_loc, agent):
        """ Calculate the new reward based on the decrease in euclidean distance to the target location
        """
        curr_dist_line = np.linalg.norm(np.cross(self._target_loc[0] - current_loc,
                                                 self._target_loc[0] - self._target_loc[1])) / np.linalg.norm(
            self._target_loc[0] - self._target_loc[1])
        next_dist_line = np.linalg.norm(np.cross(self._target_loc[0] - next_loc,
                                                 self._target_loc[0] - self._target_loc[1])) / np.linalg.norm(
            self._target_loc[0] - self._target_loc[1])

        curr_dist = self.calcDistance(current_loc, self._target_loc[agent],
                                      self.spacing)
        next_dist = self.calcDistance(next_loc, self._target_loc[agent],
                                      self.spacing)
        dist = curr_dist - next_dist
        dist_line = curr_dist_line - next_dist_line

        tot_dist = dist + dist_line
        return tot_dist

    def _calc_reward_geometric_generalized(self, current_loc, next_loc, agent):
        """ Calculate the new reward based on the decrease in euclidean distance to the target location
        """
        curr_dist_line = []
        next_dist_line = []
        for i in range(0, self.agents):
            if i != agent:
                curr_dist_line.append(np.linalg.norm(np.cross(self._target_loc[agent] - current_loc,
                                                              self._target_loc[agent] - self._target_loc[
                                                                  i])) / np.linalg.norm(
                    self._target_loc[agent] - self._target_loc[i]))
                next_dist_line.append(np.linalg.norm(np.cross(self._target_loc[agent] - next_loc,
                                                              self._target_loc[agent] - self._target_loc[
                                                                  i])) / np.linalg.norm(
                    self._target_loc[agent] - self._target_loc[i]))

        curr_dist_line = np.mean(curr_dist_line)
        next_dist_line = np.mean(next_dist_line)

        curr_dist = self.calcDistance(current_loc, self._target_loc[agent],
                                      self.spacing)
        next_dist = self.calcDistance(next_loc, self._target_loc[agent],
                                      self.spacing)
        dist = curr_dist - next_dist
        dist_line = curr_dist_line - next_dist_line
        tot_dist = dist + dist_line
        return tot_dist

    def _distance_to_other_agents(self, current_locs, next_locs, agent):
        """ Calculate the new reward based on the decrease in euclidean distance to the target location
        """
        rel_improv = []
        for i in range(0, self.agents):
            if agent != i:
                current_loc_distance = self.calcDistance(current_locs[agent], current_locs[i], self.spacing)
                next_loc_distance = self.calcDistance(next_locs[agent], next_locs[i], self.spacing)
                current_distance_target_loc = current_loc_distance - self.all_distances[agent, i]
                next_distance_target_loc = next_loc_distance - self.all_distances[agent, i]
                rel_improv.append(np.abs(current_distance_target_loc) - np.abs(next_distance_target_loc))

        rel_improv = np.mean(rel_improv)

        curr_dist = self.calcDistance(current_locs[agent], self._target_loc[agent],
                                      self.spacing)
        next_dist = self.calcDistance(next_locs[agent], self._target_loc[agent],
                                      self.spacing)
        dist = curr_dist - next_dist

        tot_dist = dist + rel_improv
        return tot_dist

    def _distance_to_other_agents_and_line(self, current_locs, next_locs, agent):
        """ Calculate the new reward based on the decrease in euclidean distance to the target location
        """
        rel_improv = []
        for i in range(0, self.agents):
            if agent != i:
                current_loc_distance = self.calcDistance(current_locs[agent], current_locs[i], self.spacing)
                next_loc_distance = self.calcDistance(next_locs[agent], next_locs[i], self.spacing)
                current_distance_target_loc = current_loc_distance - self.all_distances[agent, i]
                next_distance_target_loc = next_loc_distance - self.all_distances[agent, i]
                rel_improv.append(np.abs(current_distance_target_loc) - np.abs(next_distance_target_loc))

        rel_improv = np.mean(rel_improv)

        curr_dist_line = np.linalg.norm(np.cross(self._target_loc[0] - current_locs[agent],
                                                 self._target_loc[0] - self._target_loc[1])) / np.linalg.norm(
            self._target_loc[0] - self._target_loc[1])
        next_dist_line = np.linalg.norm(np.cross(self._target_loc[0] - next_locs[agent],
                                                 self._target_loc[0] - self._target_loc[1])) / np.linalg.norm(
            self._target_loc[0] - self._target_loc[1])
        curr_dist = self.calcDistance(current_locs[agent], self._target_loc[agent], self.spacing)
        next_dist = self.calcDistance(next_locs[agent], self._target_loc[agent], self.spacing)
        dist = curr_dist - next_dist
        dist_line = curr_dist_line - next_dist_line
        tot_dist = dist + dist_line + rel_improv
        return tot_dist

    def _distance_to_other_agents_and_line_no_point(self, current_locs, next_locs, agent):
        """ Calculate the new reward based on the decrease in euclidean distance to the target location
        """
        rel_improv = []
        for i in range(0, self.agents):
            if agent != i:
                current_loc_distance = self.calcDistance(current_locs[agent], current_locs[i], self.spacing)
                next_loc_distance = self.calcDistance(next_locs[agent], next_locs[i], self.spacing)
                current_distance_target_loc = current_loc_distance - self.all_distances[agent, i]
                next_distance_target_loc = next_loc_distance - self.all_distances[agent, i]
                rel_improv.append(np.abs(current_distance_target_loc) - np.abs(next_distance_target_loc))

        rel_improv = np.mean(rel_improv)

        curr_dist_line = np.linalg.norm(np.cross(self._target_loc[0] - current_locs[agent],
                                                 self._target_loc[0] - self._target_loc[1])) / np.linalg.norm(
            self._target_loc[0] - self._target_loc[1])
        next_dist_line = np.linalg.norm(np.cross(self._target_loc[0] - next_locs[agent],
                                                 self._target_loc[0] - self._target_loc[1])) / np.linalg.norm(
            self._target_loc[0] - self._target_loc[1])

        dist_line = curr_dist_line - next_dist_line
        tot_dist = dist_line + rel_improv
        return tot_dist

    # TODO: understand and verify that this is working
    def _oscillate(self, agent):
        """ Return True if the agent is stuck and oscillating
        """
        counter = Counter(list(self._loc_history[agent]))
        freq = counter.most_common()

        # (0, 0, 0) (= init of loc history) is the most common location in the loc history
        if len(freq) > 1 and freq[0][0] == (0, 0, 0):
            # check how often the most common non init loc has occured
            if freq[1][1] >= 4:
                return True
            else:
                return False
        # check how often the most common non init loc has occured
        elif freq[0][1] >= 4:
            return True

    def get_action_meanings(self):
        """ return array of integers for actions"""
        ACTION_MEANING = {
            1: "UP",  # MOVE Z+
            2: "FORWARD",  # MOVE Y+
            3: "RIGHT",  # MOVE X+
            4: "LEFT",  # MOVE X-
            5: "BACKWARD",  # MOVE Y-
            6: "DOWN",  # MOVE Z-
        }
        return [ACTION_MEANING[i] for i in self.actions]

    @property
    def getScreenDims(self):
        """
        return screen dimensions
        """
        return (self.width, self.height, self.depth)

    def lives(self):
        return None

    def reset_stat(self):
        """ Reset all statistics counter"""
        self.stats = defaultdict(list)
        self.num_games = StatCounter()
        self.num_success = [StatCounter()] * int(self.agents)

    def display(self, return_rgb_array=False):
        pass


# =============================================================================
# ================================ FrameStack =================================
# =============================================================================
class FrameStack(gym.Wrapper):
    """used when not training. wrapper for Medical Env"""

    def __init__(self, env, k, agents=2):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.agents = agents
        self.k = k  # history length
        # self.frames=[]
        # for i in range(0,self.agents):
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self._base_dim = len(shp)
        new_shape = shp + (k,)
        self.observation_space = spaces.Box(low=0, high=255, shape=new_shape,
                                            dtype=np.uint8)

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        ob = tuple(ob)
        # for i in range(0, self.agents):
        for _ in range(self.k - 1):
            self.frames.append(np.zeros_like(ob))
        self.frames.append(ob)
        return self._observation()

    def step(self, act, q_values, isOver, state_values=None):
        for i in range(0, self.agents):
            if isOver[i]: act[i] = 15
        current_st, reward, terminal, info = self.env.step(act, q_values, isOver, state_values)
        # for i in range(0,self.agents):
        current_st = tuple(current_st)
        self.frames.append(current_st)
        return self._observation(), reward, terminal, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.stack(self.frames, axis=-1)


# =============================================================================
# ================================== notes ====================================
# =============================================================================
"""

## Notes from landmark detection Siemens paper
# states  -> ROI - center current pos - size (2D 60x60) (3D 26x26x26)
# actions -> move (up, down, left, right)
# rewards -> delta(d) relative distance change after executing a move (action)

# re-sample -> isotropic (2D 2mm) (3D 1mm)

# gamma = 0.9 , replay memory size P = 100000 , learning rate = 0.00025
# net : 3 conv+pool - 3 FC+dropout (3D kernels for 3d data)

# navigate till oscillation happen (terminate when infinite loop)

# location is a high-confidence landmark -> if the expected reward from this location is max(q*(s_target,a))<1 the agent is closer than one pixel

# object is not in the image: oscillation occurs at points where max(q)>4


## Other Notes:

    DeepMind's original DQN paper
        used frame skipping (for fast playing/learning) and
        applied pixel-wise max to consecutive frames (to handle flickering).

    so an input to the neural network is consisted of four frame;
        [max(T-1, T), max(T+3, T+4), max(T+7, T+8), max(T+11, T+12)]

    ALE provides mechanism for frame skipping (combined with adjustable random action repeat) and color averaging over skipped frames. This is also used in simple_dqn's ALEEnvironment

    Gym's Atari Environment has built-in stochastic frame skipping common to all games. So the frames returned from environment are not consecutive.

    The reason behind Gym's stochastic frame skipping is, as mentioned above, to make environment stochastic. (I guess without this, the game will be completely deterministic?)
    cf. in original DQN and simple_dqn same randomness is achieved by having agent performs random number of dummy actions at the beginning of each episode.

    I think if you want to reproduce the behavior of the original DQN paper, the easiest will be disabling frame skip and color averaging in ALEEnvironment then construct the mechanism on agent side.


"""
