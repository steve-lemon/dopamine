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
class BubbleRunner(run_experiment.Runner):
    """BubbleRunner
    
    Args:
      proc_queue: instance of `multiprocessing.Queue`
    """
    def __init__(self, base_dir, create_agent_fn, proc_queue = None):
        '''initialize bubble-runner'''
        print('! BubbleRunner(%s)' % (base_dir))
        super(BubbleRunner, self).__init__(base_dir, create_agent_fn)
        self.proc_queue = proc_queue
        self._load_logger()
    def _run_one_episode(self):
        '''override to post episode status'''
        episode_length, episode_return = super(BubbleRunner, self)._run_one_episode()
        data = {'episode':{'length': episode_length, 'return': episode_return }}
        proc_queue.put(data) if proc_queue is not None else None
        return episode_length, episode_return
    def _load_logger(self):
        '''load logger to save into file'''
        import logging, os
        # get TF logger
        log = logging.getLogger('tensorflow')
        log.setLevel(logging.DEBUG)        
        # create file handler which logs even debug messages
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(os.path.join(self._base_dir, 'tensorflow.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        log.addHandler(fh)
