import cv2
import gym
import gym.spaces
import numpy as np
import collections

from source.wrapper.expander import make_env as mkenv
from source.wrapper.skipper import make_env as msenv

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    #
    def step(self, action):
        return self.env.step(action)
    #
    def reset(self):
        self.env.reset()
        obs, _, done, _, info = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _, info = self.env.step(2)
        obs, _, done, _, info = self.env.step(int(self.env.action_space.n/2))
        if done:
            self.env.reset()
        return obs, info

class FireResetSkipperEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetSkipperEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    #
    def step(self, action):
        return self.env.step(action)
    #
    def reset(self):
        self.env.reset()
        obs, _, done, _, info = self.env.step((0,1))
        if done:
            self.env.reset()
        obs, _, done, _, info = self.env.step((0,2))
        obs, _, done, _, info = self.env.step((0,2))
        if done:
            self.env.reset()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip
    #
    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, truncated, info
    #
    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
    #
    def observation(self, obs):
        return ProcessFrame84.process(obs)
    #
    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(
                np.float32)
        elif frame.size == 211 * 160 * 3:
            img = np.reshape(frame, [211, 160, 3]).astype(
                np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(
                np.float32)
        elif frame.size == 210 * 160 * 4:
            img = np.reshape(frame, [210, 160, 4]).astype(
                np.float32)
        elif frame.size == 250 * 160 * 4:
            img = np.reshape(frame, [250, 160, 4]).astype(
                np.float32)
        else:
            assert False, "Unknown resolution."
        if frame.shape[2] == 4:
            img = img[:, :, 0] * 0.199 + img[:, :, 1] * 0.487 + \
                img[:, :, 2] * 0.014 + img[:, :, 3] * 0.014
        else:
            img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + \
                    img[:, :, 2] * 0.114
        # resized_screen = cv2.resize(
        #     img, (84, 110), interpolation=cv2.INTER_AREA)
        if img.shape[0] == 211:
            measure_flag = img[210,:].reshape(1,img.shape[1])
            resized_screen = cv2.resize(img[:210,:], (84, 109), interpolation=cv2.INTER_AREA)      
            x_t = resized_screen[18:101, :]
            x_t = np.vstack((x_t, measure_flag[:,:84]))
        else:
            resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)      
            x_t = resized_screen[18:102, :]
        #
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32)
    #
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0), dtype=dtype)
    #
    def reset(self):
        self.buffer = np.zeros_like(
            self.observation_space.low, dtype=self.dtype)
        obs, info = self.env.reset()
        return self.observation(obs), info
    #
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_expander_env(env):
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)

def make_skipper_env(env):
    env = MaxAndSkipEnv(env)
    env = FireResetSkipperEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)


def make_env(env_name, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)


# from wrapper.expander import make_env
# from wrapper import wrappers
# from models import dqn_model
# import torch

# env = make_env("PongNoFrameskip-v4", 1.1, True, False, False, 0, max_episode_steps=200)
# env = wrappers.make_expander_env(env)
# o = env.reset()[0]

# env.observation_space.shape
# obs_size = env.observation_space.shape
# n_actions_behave = env.action_space.n

# net_behave = dqn_model.DQN_CNN(obs_size,n_actions_behave).to('cuda')

# net_behave(torch.tensor(o).to('cuda'))