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
"""Library used by example_viz.py to generate visualizations.

This file illustrates the following:
  - How to subclass an existing agent to add visualization functionality.
    - For DQN we visualize the cumulative rewards and the Q-values for each
      action (MyDQNAgent).
    - For Rainbow we visualize the cumulative rewards and the Q-value
      distributions for each action (MyRainbowAgent).
  - How to subclass Runner to run in eval mode, lay out the different subplots,
    generate the visualizations, and compile them into a video (MyRunner).
  - The function `run()` is the main entrypoint for running everything.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment

import gin
import numpy as np
import pygame

import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim as contrib_slim

from dopamine.utils.example_viz_lib import MyDQNAgent, MyRunner
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from .retro_lib import create_retro_environment


class MyBubbleDQNAgent(MyDQNAgent):
    """Sample MyBubbleDQNAgent agent based on DQN"""

    def __init__(self, sess, num_actions, summary_writer=None):
        print('! MyBubbleDQNAgent(%s)' % (num_actions))
        self.scores = []
        super(MyBubbleDQNAgent, self).__init__(sess, num_actions, summary_writer=summary_writer)

    def step(self, reward, observation, info = None):
        #print('> info={}'.format(info))
        self.scores.append(int(info['score']) if info and 'score' in info else 0)
        return super(MyBubbleDQNAgent, self).step(reward, observation)

    def reload_checkpoint(self, checkpoint_path, use_legacy_checkpoint=False):
        print('DQN> reload_checkpoint()')
        if checkpoint_path is None:
            return
        return super(MyBubbleDQNAgent, self).reload_checkpoint(checkpoint_path, use_legacy_checkpoint)

    def get_rewards(self):
        ret = super(MyBubbleDQNAgent, self).get_rewards()
        ret.append(self.scores)
        return ret

class MyBubbleIQNAgent(implicit_quantile_agent.ImplicitQuantileAgent):
    """Sample MyBubbleIQNAgent agent based on IQN"""

    def __init__(self, sess, num_actions, summary_writer=None):
        print('! MyBubbleIQNAgent(%s)' % (num_actions))
        self.rewards = []
        self.scores = []
        super(MyBubbleIQNAgent, self).__init__(sess, num_actions, summary_writer=summary_writer)

    def step(self, reward, observation, info = None):
        self.rewards.append(reward)
        self.scores.append(int(info['score']) if info and 'score' in info else 0)
        return super(MyBubbleIQNAgent, self).step(reward, observation)

    def reload_checkpoint(self, checkpoint_path, use_legacy_checkpoint=False):
        print('IQN> reload_checkpoint()')
        if checkpoint_path is None:
            return
        # return super(MyBubbleIQNAgent, self).reload_checkpoint(checkpoint_path, use_legacy_checkpoint)
        if use_legacy_checkpoint:
            variables_to_restore = atari_lib.maybe_transform_variable_names(tf.all_variables(), legacy_checkpoint_load=True)
        else:
            global_vars = set([x.name for x in tf.global_variables()])
            ckpt_vars = [
                '{}:0'.format(name)
                for name, _ in tf.train.list_variables(checkpoint_path)
            ]
            include_vars = list(global_vars.intersection(set(ckpt_vars)))
            variables_to_restore = contrib_slim.get_variables_to_restore(include=include_vars)
        if variables_to_restore:
            reloader = tf.train.Saver(var_list=variables_to_restore)
            reloader.restore(self._sess, checkpoint_path)
            tf.logging.info('Done restoring from %s', checkpoint_path)
        else:
            tf.logging.info('Nothing to restore!')

    def get_probabilities(self):
        # return self._sess.run(tf.squeeze(self._net_outputs.probabilities), {self.state_ph: self.state})
        return self._sess.run(tf.squeeze(self._net_outputs.quantile_values), {self.state_ph: self.state})

    def get_rewards(self):
        return [np.cumsum(self.rewards), self.scores]

def create_bubble_dqn_agent(sess, environment, summary_writer=None):
    return MyBubbleDQNAgent(sess, num_actions=environment.action_space.n, summary_writer=summary_writer)

def create_bubble_iqn_agent(sess, environment, summary_writer=None):
    return MyBubbleIQNAgent(sess, num_actions=environment.action_space.n, summary_writer=summary_writer)

def create_runner(base_dir, trained_agent_ckpt_path, agent='dqn', use_legacy_checkpoint=False):
    from dopamine.utils.example_viz_lib import create_dqn_agent, create_rainbow_agent
    from . import bubble_runner
    create_agent = create_rainbow_agent if agent == 'rainbow' else create_dqn_agent
    create_agent = create_bubble_dqn_agent if agent == 'dqn' else create_agent
    create_agent = create_bubble_iqn_agent if agent == 'iqn' else create_agent
    create_agent = create_bubble_iqn_agent if agent == 'bubble' else create_agent
    return bubble_runner.VizBubbleRunner(base_dir = base_dir, 
                                        trained_agent_ckpt_path = trained_agent_ckpt_path, 
                                        create_agent_fn = create_agent, 
                                        use_legacy_checkpoint = use_legacy_checkpoint)


@gin.configurable
def run(agent, game, level, num_steps, root_dir, restore_ckpt, use_legacy_checkpoint, config=None, hello = 'Hello'):
    print('run(%s)....'%(hello))
    level = int(level) if level else 1
    # sample config if config = None.
    if config is None:
        config = """
        import bubble.retro_lib
        import bubble.bubble_agent

        retro_lib.create_retro_environment.game_name = '{}'
        retro_lib.create_retro_environment.level = {}
        Runner.create_environment_fn = @retro_lib.create_retro_environment
        RetroPreprocessing.wall_offset = 200
        DQNAgent.epsilon_eval = 0.1
        DQNAgent.tf_device = '/cpu:*'
        WrappedReplayBuffer.replay_capacity = 300
    """.format(game, level)
        gin.parse_config(config)
    # calc base directory.
    base_dir = os.path.join(root_dir, '{}_viz'.format(agent), game)
    print('! base_dir = {}'.format(base_dir))

    # python -m bubble.main --agent=hello
    if agent == 'hello':
        print('%s %s!!!!' % (hello,  game))
        exit(-1)

    # 1. create runner.
    runner = create_runner(base_dir, restore_ckpt, agent, use_legacy_checkpoint)

    # 2. exec visualize().
    runner.visualize(os.path.join(base_dir, 'images-{}'.format(level)), num_global_steps=num_steps)
