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
from dopamine.utils.example_viz_lib import MyRunner
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.utils import agent_visualizer
from dopamine.utils import atari_plotter
from dopamine.utils import bar_plotter
from dopamine.utils import line_plotter
from dopamine.utils import plotter

import gin
import numpy as np
import tensorflow.compat.v1 as tf
import pygame
import gin.tf


@gin.configurable
def create_runner(base_dir, schedule='continuous_train_and_eval', level=0):
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
        return BubbleRunner(base_dir, create_agent, game_level=level)
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

    def __init__(self, base_dir, create_agent_fn, proc_queue=None, game_level=0):
        '''initialize bubble-runner'''
        print('! BubbleRunner(%s)' % (base_dir))
        assert create_agent_fn is not None
        BubbleRunner.init_logger(base_dir)
        super(BubbleRunner, self).__init__(base_dir, create_agent_fn)
        self.proc_queue = proc_queue
        self.game_level = game_level

    def post_message(self, data):
        self.proc_queue.put(data) if self.proc_queue is not None else None

    def current(self):
        import time
        return int(round(time.time() * 1000))

    def _initialize_episode(self):
        env = self._environment
        obs = env.reset(self.game_level) if self.game_level > 0 else env.reset()
        return self._agent.begin_episode(obs)

    def _run_one_step(self, action):
        observation, reward, is_terminal, info = self._environment.step(action)
        return observation, reward, is_terminal, info

    def _run_one_episode(self):
        step_number = 0
        total_reward = 0.
        agent_lives = 0
        action = self._initialize_episode()
        is_terminal = False
        is_death = False
        # Keep interacting until we reach a terminal state.
        while True:
            observation, reward, is_terminal, info = self._run_one_step(action)
            curr_lives = int(info['lives']) if 'lives' in info else 0
            total_reward += reward
            step_number += 1
            #! end the episode if death.
            is_death = True if curr_lives < agent_lives else is_death
            agent_lives = curr_lives
            #! determine terminal & EOE
            if (self.end_on_death and is_death):
                break
            # TODO(steve) - need to clip reward really?!!
            reward = np.clip(reward, -1, 1)
            if (self._environment.game_over or step_number == self._max_steps_per_episode):
                break
            elif is_terminal:
                self._agent.end_episode(reward)
                action = self._agent.begin_episode(observation)
            else:
                action = self._agent.step(reward, observation, info)
        self._end_episode(reward)
        #! report status and returns
        self.post_message({'episode': {'length': step_number, 'return': total_reward}})
        return step_number, total_reward, int(info['score']), int(info['level'])

    def _run_one_phase(self, min_steps, statistics, run_mode_str):
        step_count = 0
        num_episodes = 0
        sum_returns = 0.
        time_started = self.current()
        self.post_message({'phase': {'steps': min_steps, 'mode': run_mode_str, 'level':self.game_level }})
        while step_count < min_steps:
            episode_length, episode_return, episode_score, episode_level = self._run_one_episode()
            statistics.append({
              '{}_episode_lengths'.format(run_mode_str): episode_length,
              '{}_episode_returns'.format(run_mode_str): episode_return
            })
            step_count += episode_length
            sum_returns += episode_return
            num_episodes += 1
            sec_per_step = ((self.current() - time_started)/1000.0/step_count)
            sec_remained = int((min_steps - step_count)*sec_per_step)
            time_display = '{:1.1f}m'.format(sec_remained/60) if sec_remained > 60*5 else '{}s'.format(sec_remained)
            sys.stdout.write('Steps: {:6.0f} {:2.0f}% '.format(step_count, step_count/min_steps*100.) +
                           'Remains: {} '.format(time_display) +
                           'Episode[{}].len: {} '.format(num_episodes, episode_length) +
                           'Return: {:.1f} S:{} L:{}'.format(episode_return, episode_score, episode_level)+ 
                           '         \r')
            sys.stdout.flush()
        return step_count, sum_returns, num_episodes

    def _run_one_iteration(self, iteration):
        # print('! run_one_iteration({}) - L{}'.format(iteration, self.game_level))
        ret = super(BubbleRunner, self)._run_one_iteration(iteration)
        self.game_level = min(99, self.game_level + 1)
        return ret

    @staticmethod
    def init_logger(base_dir):
        '''initialize logger to save into file'''
        import logging, os
        # get TF logger
        log = logging.getLogger('tensorflow')
        log.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if os.path.exists(os.path.join(base_dir, 'tensorflow.log')):
            fh = logging.FileHandler(os.path.join(base_dir, 'tensorflow.log'))
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            log.addHandler(fh)
        # print log header..
        tf.logging.info('---'*32)
        tf.logging.info('BubbleRunner() starts!!')
        tf.logging.info('---'*32)

class VizBubbleRunner(BubbleRunner):
    """VizBubbleRunner: runner to visualize playing w/ checkpoint"""
    def __init__(self, base_dir, trained_agent_ckpt_path, create_agent_fn, use_legacy_checkpoint = False, game_level = 0):
        print('! VizBubbleRunner({})'.format(base_dir))
        self._trained_agent_ckpt_path = trained_agent_ckpt_path
        self._use_legacy_checkpoint = use_legacy_checkpoint
        super(VizBubbleRunner, self).__init__(base_dir, create_agent_fn, game_level=game_level)

    def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
        self._agent.reload_checkpoint(self._trained_agent_ckpt_path, self._use_legacy_checkpoint)
        self._start_iteration = 0

    def _run_one_iteration(self, iteration):
        from dopamine.discrete_domains import iteration_statistics
        statistics = iteration_statistics.IterationStatistics()
        tf.logging.info('Starting iteration %d', iteration)
        _, _ = self._run_eval_phase(statistics)
        return statistics.data_lists

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
