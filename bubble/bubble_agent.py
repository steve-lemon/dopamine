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
from dopamine.utils import agent_visualizer
from dopamine.utils import atari_plotter
from dopamine.utils import bar_plotter
from dopamine.utils import line_plotter
from dopamine.utils import plotter

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

    def begin_episode(self, observation):
        print('DQN> begin_episode()')
        return super(MyBubbleDQNAgent, self).begin_episode(observation)

    def end_episode(self, reward):
        print('DQN> end_episode(%s)'%reward)
        return super(MyBubbleDQNAgent, self).end_episode(reward)

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

    def begin_episode(self, observation):
        print('IQN> begin_episode()')
        return super(MyBubbleIQNAgent, self).begin_episode(observation)

    def end_episode(self, reward):
        print('IQN> end_episode(%s)'%reward)
        return super(MyBubbleIQNAgent, self).end_episode(reward)

    def get_probabilities(self):
        # return self._sess.run(tf.squeeze(self._net_outputs.probabilities), {self.state_ph: self.state})
        return self._sess.run(tf.squeeze(self._net_outputs.quantile_values), {self.state_ph: self.state})

    def get_rewards(self):
        return [np.cumsum(self.rewards), self.scores]

class MyObservationPlotter(plotter.Plotter):
    """MyObservationPlotter: plot observation via step()"""
    _defaults = { 'x': 0, 'y': 0 }
    def __init__(self, parameter_dict = {}, screen_size = 84):
        super(MyObservationPlotter, self).__init__(parameter_dict)
        self.width = self.parameters['width'] if 'width' in self.parameters else screen_size
        self.height = self.parameters['height'] if 'height' in self.parameters else screen_size
        self.game_surface = pygame.Surface((screen_size, screen_size))
        self.obs = None

    def setObservation(self, obs):
        self.obs = obs

    def draw(self):
        numpy_surface = np.frombuffer(self.game_surface.get_buffer(), dtype=np.int32)
        if self.obs is not None:
            obs = self.obs
            # obs = np.transpose(obs)
            # obs = np.swapaxes(obs, 1, 2)
            # obs = obs[0] | (obs[0] << 8) | (obs[0] << 16)   # must be grey-scale image (or single channel)
            np.copyto(numpy_surface, obs.ravel())
        return pygame.transform.scale(self.game_surface, (self.width, self.height))


class MyLinePlotter(line_plotter.LinePlotter):
  """MyLinePlotter: plot observation via step()"""
  def __init__(self, parameter_dict):
    myDef = {'font': {
               'family': 'DejaVu Sans',
               'weight': 'regular',
               'size': 26 },
             'figsize': (12, 9),
            }
    myDef.update(parameter_dict)
    super(MyLinePlotter, self).__init__(parameter_dict = myDef)
    #! use 2nd axes for score
    self.ax1 = self.plot.axes
    self.ax2 = self.ax1.twinx() if 1>0 else None
    self.ax2.set_ylabel('Score', color='b') if self.ax2 else None

  def draw(self):
    import pygame
    """Draw the line plot.

    If `parameter_dict` contains a 'legend' key pointing to a list of labels,
    this will be used as the legend labels in the plot.

    Returns:
      object to be rendered by AgentVisualizer.
    """
    self._setup_plot()   # draw 
    num_colors = len(self.parameters['colors'])
    max_xlim = 0
    line_data = self.parameters['get_line_data_fn']()
    for i in range(len(line_data)):
      plot_axes = self.ax2 if self.ax2 and i + 1 >= len(line_data) else self.ax1
      plot_axes.plot(line_data[i],
                     linewidth=self.parameters['linewidth'],
                     color=self.parameters['colors'][i % num_colors])
      max_xlim = max(max_xlim, len(line_data[i]))
    min_xlim = max(0, max_xlim - self.parameters['max_width'])
    self.plot.set_xlim(min_xlim, max_xlim)
    if 'legend' in self.parameters:
      self.plot.legend(self.parameters['legend'])
    self.fig.canvas.draw()
    # Now transfer to surface.
    width, height = self.fig.canvas.get_width_height()
    if self.plot_surface is None:
      self.plot_surface = pygame.Surface((width, height))
    plot_buffer = np.frombuffer(self.fig.canvas.buffer_rgba(), np.uint32)
    surf_buffer = np.frombuffer(self.plot_surface.get_buffer(),
                                dtype=np.int32)
    np.copyto(surf_buffer, plot_buffer)
    return pygame.transform.smoothscale(
        self.plot_surface,
        (self.parameters['width'], self.parameters['height']))

class MyBarPlotter(bar_plotter.BarPlotter):
  """MyBarPlotter: plot observation via step()"""
  def __init__(self, parameter_dict):
    myDef = {'font': {
               'family': 'DejaVu Sans',
               'weight': 'regular',
               'size': 26 },
            }
    myDef.update(parameter_dict)
    super(MyBarPlotter, self).__init__(parameter_dict = myDef)
  def draw(self):
      return super(MyBarPlotter, self).draw()

class MyBubbleRunner(MyRunner):
    """Custom MyRunner agent based on IQN"""
    def __init__(self, base_dir, trained_agent_ckpt_path, create_agent_fn, use_legacy_checkpoint = False):
        print('! MyBubbleRunner({})'.format(base_dir))
        super(MyBubbleRunner, self).__init__(base_dir, trained_agent_ckpt_path, create_agent_fn, use_legacy_checkpoint)

    def _run_one_step(self, action):
        observation, reward, is_terminal, info = self._environment.step(action)
        return observation, reward, is_terminal, info

    def _run_one_episode(self):
        step_number = 0
        total_reward = 0.
        action = self._initialize_episode()
        is_terminal = False
        # Keep interacting until we reach a terminal state.
        while True:
            observation, reward, is_terminal, info = self._run_one_step(action)
            total_reward += reward
            step_number += 1
            reward = np.clip(reward, -1, 1)
            if (self._environment.game_over or step_number == self._max_steps_per_episode):
                break
            elif is_terminal:
                self._agent.end_episode(reward)
                action = self._agent.begin_episode(observation)
            else:
                action = self._agent.step(reward, observation, info)
        self._end_episode(reward)
        return step_number, total_reward

    def visualize(self, record_path, num_global_steps=500):
        '''customize viz for bubble
        - origin from MyRunner.visualize()
        '''
        print('RUN> visualize(%s, %d)'%(record_path, num_global_steps))
        if not tf.gfile.Exists(record_path):
            tf.gfile.MakeDirs(record_path)
        self._agent.eval_mode = True

        # Set up the game playback rendering.
        atari_params = {'environment': self._environment,
                        'width': 240,
                        'height': 224 }

        atari_plot = atari_plotter.AtariPlotter(parameter_dict=atari_params)
        # Plot the rewards received next to it.
        reward_params = {'x': atari_plot.parameters['width'],
                         'xlabel': 'Timestep',
                         'ylabel': 'Reward',
                         'title': 'Rewards',
                         'get_line_data_fn': self._agent.get_rewards}
        #reward_plot = line_plotter.LinePlotter(parameter_dict=reward_params)
        reward_plot = MyLinePlotter(parameter_dict=reward_params)
        action_names = ['Action {}'.format(x) for x in range(self._agent.num_actions)]
        # Plot Observation at left-bottom
        obsrv_params = {
                'x': atari_plot.parameters['x'],
                'y': atari_plot.parameters['height'] - 10,
                'width': atari_plot.parameters['width'],
                'height': atari_plot.parameters['height'],
            }
        obsrv_plot = MyObservationPlotter(parameter_dict=obsrv_params)
        # Plot Q-values (DQN) or Q-value distributions (Rainbow).
        q_params = {'x': atari_plot.parameters['width'],
                    'y': atari_plot.parameters['height'],
                    'legend': action_names }
        if 'DQN' in self._agent.__class__.__name__:
            q_params['xlabel'] = 'Timestep'
            q_params['ylabel'] = 'Q-Value'
            q_params['title'] = 'Q-Values'
            q_params['get_line_data_fn'] = self._agent.get_q_values
            q_plot = MyLinePlotter(parameter_dict = q_params)
        else:
            q_params['xlabel'] = 'Return'
            q_params['ylabel'] = 'Return probability'
            q_params['title'] = 'Return distribution'
            q_params['get_bar_data_fn'] = self._agent.get_probabilities
            q_plot = MyBarPlotter(parameter_dict = q_params)
        # Screen Size
        screen_width = (atari_plot.parameters['width'] + reward_plot.parameters['width'])
        screen_height = (atari_plot.parameters['height'] + q_plot.parameters['height'])
        # Dimensions need to be divisible by 2:
        screen_width += 1 if screen_width % 2 > 0 else 0
        screen_height += 1 if screen_height % 2 > 0 else 0
        # build visualizer.
        visualizer = agent_visualizer.AgentVisualizer(
            record_path=record_path, plotters=[
                atari_plot, reward_plot, obsrv_plot, q_plot
            ],
            screen_width=screen_width, screen_height=screen_height)
        # run loop in global_step
        global_step = 0
        while global_step < num_global_steps:
            initial_observation = self._environment.reset()
            action = self._agent.begin_episode(initial_observation)
            while True:
                observation, reward, is_terminal, info = self._environment.step(action)
                global_step += 1
                obsrv_plot.setObservation(observation)
                visualizer.visualize()
                if self._environment.game_over or global_step >= num_global_steps:
                    break
                elif is_terminal:
                    self._agent.end_episode(reward)
                    action = self._agent.begin_episode(observation)
                else:
                    action = self._agent.step(reward, observation, info)
            self._end_episode(reward)
        visualizer.generate_video()


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
    # return bubble_runner.BubbleRunner(base_dir, create_agent)
    # return MyRunner(base_dir, trained_agent_ckpt_path, create_agent, use_legacy_checkpoint)
    return MyBubbleRunner(base_dir, trained_agent_ckpt_path, create_agent, use_legacy_checkpoint)


def run(agent, game, level, num_steps, root_dir, restore_ckpt, use_legacy_checkpoint):
    print('run....')
    level = int(level) if level else 1
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
    base_dir = os.path.join(root_dir, '{}_viz'.format(agent), game)
    gin.parse_config(config)
    print('! base_dir = {}'.format(base_dir))

    # 1. create runner.
    runner = create_runner(base_dir, restore_ckpt, agent, use_legacy_checkpoint)

    # 2. exec visualize().
    runner.visualize(os.path.join(base_dir, 'images'), num_global_steps=num_steps)
