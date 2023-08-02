"""
Various logging utilities.
"""
import wandb
import pandas as pd
import os
from source.helpers.eval import eval_policy_skipper_agent, eval_policy_expander_agent
import numpy as np

# def log_training_summary(result):
#     wandb.log(result)
    
# def log_best_rewards(config, result):
#     """Log best Rewards using a wandb Table"""

#     df = pd.DataFrame({'seed': config.seed, 
#         'sweep_id': [config.sweep_id], 'env_name': [config.env_name], 'obs_cost': [config.obs_cost], 
#         'best_reward': [result['best_reward']]})
#     df.to_csv(os.path.join(config.run_dir, 'rewards.csv'), index=False)    
        
# def log_eval_expander(config, env, policy_behave,detailed_name="lastTrained_"):    
#     """Run episodes using our policy and log them."""
#     data = {        
#         'action':  [],
#         'reward':  [],
#         'obs_cost': [],
#         'int_reward': [],
#         'ext_reward': [],
#         'measure': [], # Was action a measure action?
#         'step' :   [],
#         'episode': [],
#         'detailed_name': [] #best model or last trained
#     }        
#     for episode in range(config.log_eval_episodes):
#         actions, rewards, int_rewards, ext_rewards = eval_policy_expander(policy_behave, env, device=config.device)
#         R = range(len(actions))
        
#         data['action'].extend(actions)
#         data['reward'].extend(rewards)
#         data['int_reward'].extend(int_rewards)
#         data['ext_reward'].extend(ext_rewards)
#         data['detailed_name'].extend([detailed_name]*len(actions))
#         data['step'].extend([i for i in R])
#         data['episode'].extend([episode for _ in R])
#         data['obs_cost'].extend(np.repeat(config.obs_cost,len(actions)))

#         measure = [env.is_measure_action(action) for action in actions]
#         data['measure'].extend(measure)

#     df = pd.DataFrame(data=data)
#     print(config.run_dir)
#     df.to_csv(os.path.join(config.run_dir, detailed_name+'episodes_expander.csv'), index=False)
  

# def log_eval_skipper(config, env, policy_behave, policy_skip,detailed_name="lastTrained_"):    
#     """Run episodes using our policy and log them."""
#     data = {        
#         'action':  [],
#         'reward':  [],
#         'obs_cost': [],
#         'int_reward': [],
#         'ext_reward': [],
#         'measure': [], # Was action a measure action?
#         'step' :   [],
#         'episode': [],
#         'detailed_name': [] #best model or last trained
#     }        

#     for episode in range(config.log_eval_episodes):
#         actions, rewards, int_rewards, ext_rewards = eval_policy_skipper(policy_behave, policy_skip, env, device=config.device)
#         R = range(len(actions))
        
        
#         data['action'].extend(actions)
#         data['reward'].extend(rewards)
#         data['int_reward'].extend(int_rewards)
#         data['ext_reward'].extend(ext_rewards)
#         data['detailed_name'].extend([detailed_name]*len(actions))
#         data['step'].extend([i for i in R])
#         data['episode'].extend([episode for _ in R])
#         data['obs_cost'].extend(np.repeat(config.obs_cost,len(actions)))

#         measure = [action[0] for action in actions]
#         data['measure'].extend(measure)

#     df = pd.DataFrame(data=data)
#     df.to_csv(os.path.join(config.run_dir, detailed_name+'episodes_skipper.csv'), index=False)

def log_eval_skipper_agent(config, env, skipper_agent, detailed_name="lastTrained_"):    
    """Run episodes using our policy and log them."""
    data = {        
        'action':  [],
        'reward':  [],
        'obs_cost': [],
        'int_reward': [],
        'ext_reward': [],
        'measure': [], # Was action a measure action?
        'step' :   [],
        'episode': [],
        'detailed_name': [] #best model or last trained
    }        

    for episode in range(config.log_eval_episodes):
        actions, rewards, int_rewards, ext_rewards = eval_policy_skipper_agent(skipper_agent, env)
        R = range(len(actions))
        
        
        data['action'].extend(actions)
        data['reward'].extend(rewards)
        data['int_reward'].extend(int_rewards)
        data['ext_reward'].extend(ext_rewards)
        data['detailed_name'].extend([detailed_name]*len(actions))
        data['step'].extend([i for i in R])
        data['episode'].extend([episode for _ in R])
        data['obs_cost'].extend(np.repeat(config.obs_cost,len(actions)))

        measure = [action[0] for action in actions]
        data['measure'].extend(measure)

    df = pd.DataFrame(data=data)
    df.to_csv(os.path.join(config.run_dir, detailed_name+'episodes_skipper.csv'), index=False)

def log_eval_expander_agent(config, env, expander_agent, detailed_name="lastTrained_"):    
    """Run episodes using our policy and log them."""
    data = {        
        'action':  [],
        'reward':  [],
        'obs_cost': [],
        'int_reward': [],
        'ext_reward': [],
        'measure': [], # Was action a measure action?
        'step' :   [],
        'episode': [],
        'detailed_name': [] #best model or last trained
    }        

    for episode in range(config.log_eval_episodes):
        actions, rewards, int_rewards, ext_rewards = eval_policy_expander_agent(expander_agent, env)
        R = range(len(actions))
        
        
        data['action'].extend(actions)
        data['reward'].extend(rewards)
        data['int_reward'].extend(int_rewards)
        data['ext_reward'].extend(ext_rewards)
        data['detailed_name'].extend([detailed_name]*len(actions))
        data['step'].extend([i for i in R])
        data['episode'].extend([episode for _ in R])
        data['obs_cost'].extend(np.repeat(config.obs_cost,len(actions)))
        #
        measure = [env.is_measure_action(action) for action in actions]
        data['measure'].extend(measure)

    df = pd.DataFrame(data=data)
    df.to_csv(os.path.join(config.run_dir, detailed_name+'episodes_expander.csv'), index=False)

