# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import gin
import gym
import retro

from gym.spaces.box import Box
import numpy as np
import tensorflow.compat.v1 as tf
import cv2


@gin.configurable
def create_retro_environment(game_name='BubbleBobble', sticky_actions=True, level=1):
    '''create retro game'''
    print('! create_retro_environment: %s/%d' % (game_name, level))
    env = RetroPreprocessing.createRetroGameEnv(game_name, sticky_actions, level)
    return RetroPreprocessing(env, game_level=1)


@gin.configurable
class RetroPreprocessing(object):
    '''RetroPreprocessing
    - wrapper of origin environment for pre-processing.
    '''
    _env = None             # old env created before (must be singleton of retro-env)
    @staticmethod
    def createRetroGameEnv(game_name='BubbleBobble', sticky_actions=True, level=1):
        assert game_name is not None
        rom_name = 'Nes' if sticky_actions else 'Nes'
        level = int(level) if level else 1
        full_game_name = '{}-{}'.format(game_name, rom_name)
        state = 'Level%02d' % level if level else retro.State.DEFAULT
        # print('! create-retro-game-env: %s/%s' % (full_game_name, state))
        tf.logging.info('Create RetroGame:%s w/ stage:%s' % (full_game_name, state))
        # close the previous one
        if RetroPreprocessing._env is not None:
            RetroPreprocessing._env.close()
            RetroPreprocessing._env = None
        # make env, then cache it.
        env = retro.make(game=full_game_name, state=state)
        RetroPreprocessing._env = (env if env else None)
        return env

    def __init__(self, environment, frame_skip=4, terminal_on_life_loss=True, screen_size=84, wall_offset=200, step_penalty=-0.001, game_level=0, reset_fire=0, score_bonus=0):
        """Constructor for an Atari 2600 preprocessor.

        Args:
           wall_offset: offset of wall position (0 means no wall-clearance)
           step_penalty: base penalty per each step.
           game_level: initial game-level on reset()
           reset_fire: index starts from 1 in order to reset fire press
           score_bonus: bonus reward if got +score.
        """
        if frame_skip <= 0:
            raise ValueError(
                'Frame skip should be strictly positive, got {}'.format(frame_skip))
        if screen_size <= 0:
            raise ValueError(
                'Target screen size should be strictly positive, got {}'.format(screen_size))

        self.environment = environment
        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.wall_offset = wall_offset                  # NOTE - X offset of WALL.
        self.step_penalty = step_penalty if 1 else 0
        self.game_level = game_level                    # current level of game. changeable by reset()
        self.reset_fire = reset_fire                    # flag to reset fire-press on last action trigger.
        self.score_bonus = score_bonus                  # bonus reward on score

        print('! RetroPreprocessing: wall_offset={}, step_penalty={}, game_level={}, reset_fire={}, score_bonus={}'.format(self.wall_offset, self.step_penalty, self.game_level, self.reset_fire, self.score_bonus))

        obs_dims = self.environment.observation_space
        # Stores temporary observations used for pooling over two successive
        # frames.
        self.screen_buffer = [
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
        ]

        self.game_over = False
        self.lives = 0          # Will need to be set by reset().
        self.last_score = 0     # last number of score
        self.last_level = 0     # last count of level
        self.last_lives = 0     # last count of lives
        self.last_enems = 0     # last count of enemies
        self.last_walls = []    # TODO - IMPROVE to replace wall-color for every level.

        # NOTE - core actions for BubbleBobble.
        self.mapping = {
            0: [0, 0, 0, 0, 0, 0, 0, 1, 0],  # RIGHT
            1: [0, 0, 0, 0, 0, 0, 1, 0, 0],  # LEFT
            2: [0, 0, 0, 0, 0, 0, 0, 0, 1],  # JUMP
            3: [1, 0, 0, 0, 0, 0, 0, 0, 0],  # FIRE
            4: [1, 0, 0, 0, 0, 0, 0, 1, 0],  # FIRE + RIGHT
            5: [1, 0, 0, 0, 0, 0, 1, 0, 0],  # FIRE + LEFT
        }

    @property
    def observation_space(self):
        # Return the observation space adjusted to match the shape of the processed
        return Box(low=0, high=255, shape=(self.screen_size, self.screen_size, 1), dtype=np.uint8)

    @property
    def action_space(self):
        # Return the discrete counts for actions
        return gym.spaces.Discrete(len(self.mapping.keys()))

    @property
    def reward_range(self):
        return self.environment.reward_range

    @property
    def metadata(self):
        return self.environment.metadata

    def close(self):
        return self.environment.close()

    def reset(self, game_level = 0):
        '''reset to init state
        Args:
           game_level: if >0, then should re-create retro env with game-level.
        '''
        # re-create the origin env if level has changed.
        if game_level > 0 and game_level != self.game_level:
            self.environment = RetroPreprocessing.createRetroGameEnv(level = game_level)
            self.game_level = game_level
        # reset to init-state
        self.environment.reset()
        # NOTE - catch initial state by one-step.
        obs, reward, game_over, info = self.environment.step([0])
        self.lives = info['lives']
        self.last_score = int(info['score']) if 'score' in info else 0
        self.last_level = int(info['level']) if 'level' in info else 0
        self.last_lives = int(info['lives']) if 'lives' in info else 0
        self.last_enems = int(info['enemies']) if 'enemies' in info else 0
        #print('! obs.shape={}'.format(np.shape(obs)))
        # NOTE - detect colors of wall for clearance.
        self.last_walls = []
        if self.wall_offset > 0:
            woff = self.wall_offset
            wall = obs[woff:woff+8,0:8,]      # find wall position.
            wall = np.reshape(wall, (64, 3))  # reshape to list of RGB
            wall = np.unique(wall, axis = 0)  # as [[240 120 248] [248 196 248]]
            #print('! wall({})={}'.format(np.shape(wall), wall))
            #self.last_walls = wall
            for w in wall:
                if not np.all(w == [0,0,0]):        
                    self.last_walls.append(w)
            self.last_walls = np.array(self.last_walls)
        #print('! wall({})={}'.format(np.shape(self.last_walls), self.last_walls))
        #! fill with initial screen
        self._fetch_grayscale_observation(obs, self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        return self._pool_and_resize()

    def render(self, mode):
        return self.environment.render(mode)

    def step(self, a):
        # action = np.identity(9, dtype=np.int32)[a]
        action = self.mapping.get(a)
        #print('> step(%s): %s'%(a, action))
        accumulated_reward = 0.

        curr_level = self.last_level
        curr_score = self.last_score
        curr_lives = self.last_lives
        curr_enems = self.last_enems
        for time_step in range(self.frame_skip):
            # We bypass the Gym observation altogether and directly fetch the
            # grayscale image from the ALE. This is a little faster.
            # NOTE(steve) - if holding to press fire-button, it would not fire actually. so clear fire on last frame-skip @200610
            if self.reset_fire > 0 and self.frame_skip > 2 and time_step + 2 >= self.frame_skip:
                action[self.reset_fire - 1] = 0           # clear fire press state.
            #! run step to origin game-env.
            obs, reward, game_over, info = self.environment.step(action)

            #! catch the last state of info + game
            curr_level = int(info['level']) if 'level' in info else curr_level
            curr_score = int(info['score']) if 'score' in info else curr_score
            curr_lives = int(info['lives']) if 'lives' in info else curr_lives
            curr_enems = int(info['enemies']) if 'enemies' in info else curr_enems

            #! determine if terminal
            if self.terminal_on_life_loss:
                new_lives = curr_lives
                is_terminal = game_over or new_lives < self.lives
                self.lives = new_lives
            else:
                is_terminal = game_over

            #! break if terminal
            if is_terminal:
                break

            #! We max-pool over the last two frames, in grayscale.
            elif time_step >= self.frame_skip - 2:
                t = time_step - (self.frame_skip - 2)
                self._fetch_grayscale_observation(obs, self.screen_buffer[t])

        # Pool the last two observations.
        observation = self._pool_and_resize()

        #! calculate the result reward..
        accumulated_reward = self._calculate_step_reward(curr_level, curr_score, curr_lives, curr_enems, game_over)

        # NOTE - save the latest status...
        self.last_level = curr_level
        self.last_score = curr_score
        self.last_lives = curr_lives
        self.last_enems = curr_enems
        self.game_over = game_over
        return observation, accumulated_reward, is_terminal, info

    def _fetch_grayscale_observation(self, obs, output):
        # clear walls
        for wall in self.last_walls:
            # masked = np.all(obs == wall, axis=-1)
            # obs[masked] = [255,32,32]
            obs[np.all(obs == wall, axis=2)] = [255,62,62]
        # use Green channel as grayscale (SIMPLE BUT FAST)
        obs = obs[:,:,1]
        np.copyto(output, obs)
        return output

    def _pool_and_resize(self):
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(
                self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[0])

        transformed_image = cv2.resize(self.screen_buffer[0],
                                       (self.screen_size, self.screen_size),
                                       interpolation=cv2.INTER_AREA)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=2)
    
    def asImage(self, img = None, tpy = None):
        from PIL import Image
        img = img if img is not None else self._pool_and_resize().squeeze()
        tpy = tpy if tpy is not None else 'P'
        return Image.fromarray(img, tpy)

    def _calculate_step_reward(self, curr_level, curr_score, curr_lives, curr_enems, game_over):
        """
        reward for score configuration:
        [objective]
        - survive as long as possible
        - achieve as mush as score
        """
        # init with base penalty per each step
        acc_rew = self.step_penalty
        # kill an enemy - with double reward along with score (1 kill -> 100 score)
        if self.last_enems > curr_enems:
            acc_rew += 1 * (self.last_enems - curr_enems)
        # case of life lost
        if self.last_lives > curr_lives:
            acc_rew += -5 * (self.last_lives - curr_lives)      # max 3x lives
        # get enhancement in log scale.
        if curr_score > (self.last_score + 1):
            acc_rew += math.log(curr_score - self.last_score, 100) + self.score_bonus
        # return with total.
        return acc_rew
