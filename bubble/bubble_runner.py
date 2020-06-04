# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from dopamine.discrete_domains import run_experiment

import numpy as np
import tensorflow.compat.v1 as tf

import gin.tf


@gin.configurable
def create_runner(base_dir, schedule='continuous_train_and_eval', level = 0):
  """Creates an Bubble Runner.
  - originally copied via run_experiment.create_runner

  Args:
    level: the initial stage level to start (reset condition)
  """
  assert base_dir is not None
  from dopamine.discrete_domains.run_experiment import TrainRunner
  from dopamine.discrete_domains.run_experiment import create_agent
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return BubbleRunner(base_dir, create_agent, game_level = level)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return TrainRunner(base_dir, create_agent)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


@gin.configurable
class BubbleRunner(run_experiment.Runner):
    """BubbleRunner
    - customized for bubble runner
    
    Args:
      proc_queue: instance of `multiprocessing.Queue`
    """
    def __init__(self, base_dir, create_agent_fn, proc_queue = None, game_level=0):
        '''initialize bubble-runner'''
        print('! BubbleRunner(%s)' % (base_dir))
        BubbleRunner.init_logger(base_dir)
        super(BubbleRunner, self).__init__(base_dir, create_agent_fn)
        self.proc_queue = proc_queue
        self.game_level = game_level

    def _initialize_episode(self):
        '''override to reset w/ level dynamically'''
        # print('> initialize_episode(%d)' % (self.game_level))
        initial_observation = self._environment.reset(self.game_level)
        return self._agent.begin_episode(initial_observation)

    def _run_one_episode(self):
        '''override to post episode status'''
        # print('> run_one_episode(%d)' % (self.game_level))
        episode_length, episode_return = super(BubbleRunner, self)._run_one_episode()
        data = {'episode':{'length': episode_length, 'return': episode_return }}
        self.proc_queue.put(data) if self.proc_queue is not None else None
        return episode_length, episode_return

    @staticmethod
    def init_logger(base_dir):
        '''initialize logger to save into file'''
        import logging, os
        # get TF logger
        log = logging.getLogger('tensorflow')
        log.setLevel(logging.DEBUG)        
        # create file handler which logs even debug messages
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(os.path.join(base_dir, 'tensorflow.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        log.addHandler(fh)
        # print log header..
        tf.logging.info('---'*32)
        tf.logging.info('BubbleRunner() starts!!')
        tf.logging.info('---'*32)
