import gym
import numpy as np
from gym.spaces import Box

  
def make_env(env_name, int_reward, max_repeat, vanilla, full_ext_reward=True, render_mode=None,max_episode_steps = 200, verbose=False):
    """
    Make an environment, potentially wrapping it in MeasureWrapper.
    
    Args:
        env_name: Environment name
        int_reward: The intrinsic reward given for not measuring to use for the wrapper
        max_repeat: The maximum number of time that the agent can skip action selection cycle
        full_ext_reward: Whether the extrinsic rewards are added for intermediate steps
        vanilla: If True, uses the original environment without a wrapper. Ignores int_reward

    Returns:
        A gym environment.
    """
    env = gym.make(env_name, render_mode=render_mode)
    if vanilla:
        return VanillaWrapper(env)
    else:
        env = SkipperWrapper(env, int_reward=int_reward, full_ext_reward=full_ext_reward, max_repeat=max_repeat,max_episode_steps = max_episode_steps, render_mode=render_mode, verbose=verbose)
        return env

class VanillaWrapper(gym.Wrapper):
    def is_measure_action(self, _action):
        return False
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)

        info['int_reward'] = 0.0
        info['ext_reward'] = reward

        return state, reward, done, info

class SkipperWrapper(gym.Wrapper):
    """Augments environments to take a behaviour action and a repeat action.
    Stores the original reward in the info['ext_reward'] attribute, and the intrinsic reward for not measuring in info['int_reward'] attribute.

    "action_pair: (a,k)" the first element is the bahaviour action and the second 
    element is an integer between [1, max_repeat] specifying the number of times
    to repeat the action. The action, a, is applied k times. measurements are 
    skipped for the first k-1 time steps. The total intrinsic reward is k-1 x int_reward
    and the total extrinsic reward is sum of the rewards produced by the behaviour
    policy. 
    
    "a" is any legitiment behaviour actions

    "k" is an integer in the range [1, max_repeat]

    "total_reward" the total reward is the sum of the int_rewards and ext_rewards
    """
    def __init__(self, env, int_reward, max_repeat, full_ext_reward=True,max_episode_steps = 200, render_mode=None, verbose=False):
        super().__init__(env)
        self.verbose = verbose
        # self.render_mode = render_mode
        self.full_ext_reward=full_ext_reward
        self.env.spec.max_episode_steps = max_episode_steps
        self.global_step = 0
        self.int_reward = int_reward
        self.max_repeat = max_repeat
        self.continuous_action_space = True if env.action_space.shape else False
        
        #Action space becomes a tuple where the first element is the number of time to repeat and the second action is the behaviour action
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(self.max_repeat), env.action_space))

        self.observation_space = env.observation_space

    def step(self, action_pair):         
        skipper_num = action_pair[0]+1
        action = action_pair[1]
        global_steps = []
        actions = []
        measures = []
        int_reward = -self.int_reward
        ext_reward = 0
        fullObs_reward = 0
        if self.verbose:
            for _ in range(skipper_num):
                print("action %i don't measure" %(action))
            print("action %i measure" %(action))
        for stp in range(skipper_num):
            state, reward, done, trunc, info = self.env.step(action)
            #record fully observable reward (for evaluation)
            fullObs_reward += reward

            actions.append(action)
            measures.append(0)
            reward = reward
            int_reward += self.int_reward
            if self.full_ext_reward: #sum all extrinsic rewards (assumes we know the rewards without the state info)
                ext_reward += reward
            self.global_step += 1
            global_steps.append(self.global_step)

            #check if at max episode length or done
            if self.env.spec.max_episode_steps is not None:
                if self.global_step >= self.env.spec.max_episode_steps:
                    trunc = True
            
            if done or trunc:
                # print("done or truc at global setp %i" % self.global_step)
                ext_reward = reward
                info['fullObs_reward']   = fullObs_reward
                info['int_reward'] = int_reward
                info['ext_reward'] = ext_reward
                info['tot_steps']  = stp+1
                info['global_step']  = global_steps
                info['action']  = actions
                info['measure']  = measures
                self.global_step = 0
                return state, int_reward+ext_reward, done, trunc, info
            
        if not self.full_ext_reward: # take only the extrinsic reward from the last step. The step way pay to see the state (same as Shann Tzu-Yun)
            ext_reward = reward
        
        measures[-1] = 1
        info['fullObs_reward']   = fullObs_reward
        info['int_reward'] = int_reward
        info['ext_reward'] = ext_reward
        info['tot_steps']  = skipper_num
        info['global_step']  = global_steps
        info['action']  = actions
        info['measure']  = measures
        return state, int_reward+ext_reward, done, trunc, info
    
    # def render(self):
    #     self.render()
    
    def reset(self):
        self.global_step = 0
        state, info = self.env.reset()
        return state, info