import gym
import numpy as np
from gym.spaces import Box


def make_env(env_name, int_reward, obs_flag, prev_action_flag, vanilla, full_ext_reward=True, render_mode=None, max_episode_steps=200, verbose=False):
    """
    Make an environment, potentially wrapping it in MeasureWrapper.
    
    Args:
        env_name: Environment name
        int_reward: The intrinsic reward given for not measuring to use for the wrapper
        obs_flag: Whether to include a flag in the output to indicate if last action was observe
        prev_action: Whether to include the previously applied action in output observation
        full_ext_reward: Whether the extrinsic rewards are added for intermediate steps
        vanilla: If True, uses the original environment without a wrapper. Ignores obs_cost and obs_flag arguments.

    Returns:
        A gym environment.
    """
    env = gym.make(env_name, render_mode=render_mode)
    if vanilla:
        return VanillaWrapper(env)
    else:
        env = ExpanderWrapper(env, int_reward=int_reward, unknown_state='LAST_MEASURED', obs_flag=obs_flag, prev_action_flag=prev_action_flag,full_ext_reward=full_ext_reward,max_episode_steps = max_episode_steps, render_mode=render_mode, verbose=verbose)
        return env

class VanillaWrapper(gym.Wrapper):
    def is_measure_action(self, _action):
        return False
    
    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)

        info['int_reward'] = 0.0
        info['ext_reward'] = reward

        return state, reward, done, trunc, info

class ExpanderWrapper(gym.Wrapper):
    """Augments environments with an extra observe action and optionally an observe flag.
    Stores the original reward in the info['ext_reward'] attribute, and the reward for not measuring 
    in the info['int_reward'] attribute.

    "measure" either 0 or 1. If it is 1 then action (below) is carried
    out and an accurate state representation is returned without the intrinsic
    reward added to the entrinsic reward
    If 0, then the intrinsic reward is added to the entrinsic reward
    but the state returned cannot be trusted to be accurate.
    
    For discrete environments, measure action is encoded as:
    1.    0,   (n-1) - don't measure, just carry out the action
    2. (n+1), (2n-1) - measure, carry out the (action - n)

    So, for example, in CartPole there are two actions, so n=2:
    1. 0-1 - don't measure
    2. 2-3 - measure 

    In AcroBot, n=3, so:
    1. 0-2 - don't measure
    2. 3-5 - measure
    
    For continuous environments, a new bit is prepended to the action tensor, e.g.:

        action = [1,2,3]
        measure_action = [0,1,2,3]
    
    If obs_flag is True then state is augmented (prepended) 
    with observation flag, which is 1 if measure action was taken and 0 otherwise.

    If prev_action_flag is True then state is augmented (prepended) 
    with the previously applied action.

    Unknown state is either RANDOM, LAST_MEASURED or NOISY.
    RANDOM is a random sample from observation space.
    LAST_MEASURED returns the state that was last measured.
    NOISY_X samples noise with X stdev from a normal distribution with a mean of 0.
    """
    def __init__(self, env, int_reward, unknown_state, obs_flag=True, prev_action_flag=False, full_ext_reward=True,max_episode_steps = 200, render_mode=None, verbose=False):
        super().__init__(env)
        self.verbose = verbose
        self.full_ext_reward=full_ext_reward
        self.env.spec.max_episode_steps = max_episode_steps
        self.global_step = 0
        self.obs_flag = obs_flag
        self.prev_action_flag = prev_action_flag
        self.continuous_action_space = True if env.action_space.shape else False
                
        if self.continuous_action_space:
            la = np.concatenate(([0], env.unwrapped.action_space.low))
            ha = np.concatenate(([1], env.unwrapped.action_space.high))        
            self.action_space = Box(low=la, high=ha, 
                dtype=env.unwrapped.action_space.dtype)
        else:
            self.action_space = gym.spaces.Discrete(2*env.action_space.n)

        self.state_low = env.observation_space.low
        self.state_high = env.observation_space.high

        self.unknown_state = unknown_state
        self.last_state = None
        self.int_reward = int_reward    

        l = env.unwrapped.observation_space.low
        h = env.unwrapped.observation_space.high  
        if obs_flag | prev_action_flag:
            if self.obs_flag:
                obs_shape = env.unwrapped.observation_space.shape
                if len(obs_shape) == 3:
                    # l = np.dstack((l, np.zeros((obs_shape[0],obs_shape[1],1))))
                    # h = np.dstack((h, np.ones((obs_shape[0],obs_shape[1],1))))
                    meaure_flag_l = np.min(self.env.unwrapped.observation_space.low)
                    meaure_flag_h = np.max(self.env.unwrapped.observation_space.low)
                    l = np.vstack((l, np.repeat(meaure_flag_l,obs_shape[1]*obs_shape[2]).reshape((1,obs_shape[1],obs_shape[2]))))
                    h = np.vstack((h, np.repeat(meaure_flag_h,obs_shape[1]*obs_shape[2]).reshape((1,obs_shape[1],obs_shape[2]))))
                else:
                    l = np.concatenate(([0], l))
                    h = np.concatenate(([1], h))        
            if self.prev_action_flag:
                if self.continuous_action_space:
                    l = np.concatenate((env.unwrapped.action_space.low, l))
                    h = np.concatenate((env.unwrapped.action_space.high, h))   
                else:
                    l = np.concatenate(([0], l))
                    h = np.concatenate(([2*env.action_space.n], h))   
            self.observation_space = Box(low=l, high=h)

    def is_measure_action(self, action):
        if self.continuous_action_space:
            return action[0] >= 0.5
        else:
            return action >= self.env.action_space.n
        
    def step(self, action):
        self.global_step += 1
        # determin measure behaviour        
        if self.is_measure_action(action):
            measure = 1
            
            if not self.continuous_action_space:
                action = action - self.env.action_space.n
        else:
            measure = 0
        
        if self.continuous_action_space:
            action = action[1:]

        if self.verbose:
            print("action %i applied, measure: %i" %(action, measure))

        #apply actions
        state, reward, done, trun, info = self.env.step(action)
        
        #record fully observable reward (for evaluation)
        fullObs_reward = reward
        #record extrinsic reward
        ext_reward = reward

        #check if at max episode length or done
        if self.env.spec.max_episode_steps is not None:
            if self.global_step == self.env.spec.max_episode_steps:
                trun = True

        #apply action 
        if 'Pong' in str(self.env) and (reward == 1 or reward == -1): #just for Pong so the agent gets the first state in a new game start
            self.last_state = state
            measure = 1
            int_reward = 0
        elif measure == 1: # measure next state 
            self.last_state = state
            int_reward = 0
        elif measure == 0: # don't measure next state
            int_reward = self.int_reward
            if self.full_ext_reward or done: #sum all extrinsic rewards (assumes we know the rewards without the state info) or if done show final reward
                reward += int_reward
            else: # take only the extrinsic reward from the last step. The step way pay to see the state (same as Shann Tzu-Yun)
                reward = int_reward
                ext_reward = 0
            if self.unknown_state == 'RANDOM':
                state = self.unwrapped.observation_space.sample()
            elif self.unknown_state == 'LAST_MEASURED':
                state = self.last_state
            elif self.unknown_state.startswith('NOISY'):
                std = self.parse_std(self.unknown_state)
                noise = np.random.normal(loc=state, scale=std)
                state = np.clip(state + noise, self.state_low, self.state_high)            
            else:
                raise ValueError(f"Unknown state invalid: {self.unknown_state}")
        
        #for next observation
        if self.obs_flag:
            obs_shape = state.shape
            if len(obs_shape) == 3:
                    if measure == 0:
                        meaure_flag = np.min(self.env.unwrapped.observation_space.low)
                        # state = np.dstack((state, np.repeat(meaure_flag,obs_shape[0]*obs_shape[1]).reshape((obs_shape[0],obs_shape[1],1))))
                        state = np.vstack((state, np.repeat(meaure_flag,obs_shape[1]*obs_shape[2]).reshape((1,obs_shape[1],obs_shape[2]))))
                    else: 
                        meaure_flag = np.max(self.env.unwrapped.observation_space.high)
                        # state = np.dstack((state, np.repeat(meaure_flag,obs_shape[0]*obs_shape[1]).reshape((obs_shape[0],obs_shape[1],1))))
                        state = np.vstack((state, np.repeat(meaure_flag,obs_shape[1]*obs_shape[2]).reshape((1,obs_shape[1],obs_shape[2]))))
            else:
                state = np.concatenate(([measure], state))
        
        if self.prev_action_flag:
            if self.continuous_action_space:
                state = np.concatenate((action, state))
            else:
                state = np.concatenate(([action], state))

        info['fullObs_reward']   = fullObs_reward
        info['int_reward']   = int_reward
        info['ext_reward']   = ext_reward
        info['tot_steps']    = 1
        info['global_step']  = self.global_step
        info['action']   = action
        info['measure']  = measure
        return state.astype(np.float32), reward, done, trun, info

    def parse_std(self, unknown_state):
        return float(unknown_state.split('-')[1])


    def reset(self):
        state, info = self.env.reset()
        self.last_state = state
        self.global_step = 0

        if self.obs_flag:
            obs_shape = state.shape
            if len(obs_shape) == 3:
                meaure_flag = np.max(self.env.unwrapped.observation_space.high)
                # state = np.dstack((state, np.repeat(meaure_flag,obs_shape[0]*obs_shape[1]).reshape((obs_shape[0],obs_shape[1],1))))
                state = np.vstack((state, np.repeat(meaure_flag,obs_shape[1]*obs_shape[2]).reshape((1,obs_shape[1],obs_shape[2]))))
            else:
                state = np.concatenate(([1], state))

        if self.prev_action_flag:
            if self.continuous_action_space:
                state = np.concatenate(([0], state))
            else:
                state = np.concatenate(([0], state))
        
        return state.astype(np.float32), info