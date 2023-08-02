# import gym
# import torch
import random
import collections
# from torch.autograd import Variable

import numpy as np

from collections import namedtuple, deque

from ptan.agent import BaseAgent
from ptan.common import utils

# one single experience step
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'ext_reward', 'int_reward', 'done', 'trunc'])

class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries
    """
    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        :param vectorized: support of vectorized envs from OpenAI universe
        """
        # assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.sum_of_measures = 0
        self.fullObs_rewards = 0
        self.total_rewards = []
        self.total_steps = []
        self.total_ext_rewards = []
        self.total_int_rewards = []
        self.total_fullObs_rewards = []
        self.total_sum_of_measures = []
        self.last_rewards = []
        self.vectorized = vectorized

    def __iter__(self):
        states, agent_states, histories, cur_rewards, cur_steps, cur_int_rewards, cur_ext_rewards, cur_fullObs_rewards, cur_sum_of_measures, cur_sum_of_noMeasures = [], [], [], [], [], [], [], [], [], []
        env_lens = []
        for env in self.pool:
            obs, _ = env.reset()
            # if the environment is vectorized, all it's output is lists of results.
            # Details are here: https://github.com/openai/universe/blob/master/doc/env_semantics.rst
            if self.vectorized:
                obs_len = len(obs)
                states.extend(obs)
            else:
                obs_len = 1
                states.append(obs)
            env_lens.append(obs_len)

            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count))
                cur_rewards.append(0.0)
                cur_int_rewards.append(0.0)
                cur_ext_rewards.append(0.0)
                cur_fullObs_rewards.append(0.0)
                cur_sum_of_measures.append(0.0)
                cur_steps.append(0)
                agent_states.append(self.agent.initial_state())

        iter_idx = 0
        while True:
            actions = [None] * len(states)
            states_input = []
            states_indices = []
            for idx, state in enumerate(states):
                if state is None:
                    actions[idx] = self.pool[0].action_space.sample()  # assume that all envs are from the same family
                else:
                    states_input.append(state)
                    states_indices.append(idx)
            if states_input:
                states_actions, new_agent_states = self.agent(states_input, agent_states)
                for idx, action in enumerate(states_actions):
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = new_agent_states[idx]
            grouped_actions = _group_list(actions, env_lens)

            global_ofs = 0
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                if self.vectorized:
                    next_state_n, r_n, is_done_n, is_trunc_n , info_n  = env.step(action_n)
                    # is_done_n = is_done_n or is_trunc_n # TODO fix handling of trunc  
                else:
                    next_state, r, is_done, is_trunc, info = env.step(action_n[0])
                    # is_done = is_done or is_trunc # TODO fix handling of trunc  
                    next_state_n, r_n, is_done_n, is_trunc_n, info_n = [next_state], [r], [is_done], [is_trunc], [info]
                
                for ofs, (action, next_state, r, is_done, is_trunc, info) in enumerate(zip(action_n, next_state_n, r_n, is_done_n, is_trunc_n, info_n)):
                    idx = global_ofs + ofs
                    state = states[idx]
                    history = histories[idx]

                    cur_rewards[idx] += r
                    cur_steps[idx] += info['tot_steps']
                    cur_int_rewards[idx] += info['int_reward']
                    cur_ext_rewards[idx] += info['ext_reward']
                    cur_fullObs_rewards[idx] += info['fullObs_reward']
                    if type(info['measure']) is list:
                        cur_sum_of_measures[idx] += 1
                    else:
                        cur_sum_of_measures[idx] += info['measure']
                    if state is not None:
                        history.append(Experience(state=state, action=action, reward=r, ext_reward=info['ext_reward'], int_reward=info['int_reward'], done=is_done, trunc=is_trunc))
                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    states[idx] = next_state
                    if is_done or is_trunc:
                        # in case of very short episode (shorter than our steps count), send gathered history
                        if 0 < len(history) < self.steps_count:
                            yield tuple(history)
                        # generate tail of history
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)
                        
                        self.total_rewards.append(cur_rewards[idx])
                        self.total_steps.append(cur_steps[idx])
                        self.total_ext_rewards.append(cur_ext_rewards[idx])
                        self.total_int_rewards.append(cur_int_rewards[idx])
                        self.total_fullObs_rewards.append(cur_fullObs_rewards[idx])
                        self.total_sum_of_measures.append(cur_sum_of_measures[idx])
                        self.last_rewards.append(r)
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        cur_int_rewards[idx] = 0.0
                        cur_ext_rewards[idx] = 0.0
                        cur_fullObs_rewards[idx] = 0.0
                        cur_sum_of_measures[idx] = 0.0 
                        # vectorized envs are reset automatically
                        states[idx], _ = env.reset() if not self.vectorized else None
                        agent_states[idx] = self.agent.initial_state()
                        history.clear()
                global_ofs += len(action_n)
            iter_idx += 1

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res
    
    def pop_verbose_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_ext_rewards, self.total_int_rewards, self.total_steps, self.last_rewards, self.total_fullObs_rewards, self.total_sum_of_measures))
        if res:
            self.total_rewards, self.total_ext_rewards, self.total_int_rewards, self.total_steps, self.last_rewards,self.total_fullObs_rewards, self.total_sum_of_measures = [],[], [], [],[], [], []
        return res


def _group_list(items, lens):
    """
    Unflat the list of items by lens
    :param items: list of items
    :param lens: list of integers
    :return: list of list of items grouped by lengths
    """
    res = []
    cur_ofs = 0
    for g_len in lens:
        res.append(items[cur_ofs:cur_ofs+g_len])
        cur_ofs += g_len
    return res


# those entries are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
ExperienceFirstLast = collections.namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    """
    def __init__(self, env, agent, gamma, steps_count=1, steps_delta=1, vectorized=False):
        assert isinstance(gamma, float)
        super(ExperienceSourceFirstLast, self).__init__(env, agent, steps_count+1, steps_delta, vectorized=vectorized)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)


class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)


class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, experience_source, buffer_size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(experience_source, buffer_size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = utils.SumSegmentTree(it_capacity)
        self._it_min = utils.MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def _add(self, *args, **kwargs):
        idx = self.pos
        super()._add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        samples = [self.buffer[idx] for idx in idxes]
        return samples, idxes, weights

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

