import torch
import numpy as np

"""
Evaluate a trained policy
"""
# @torch.no_grad()
# def eval_policy_expander(net_behave, env, epsilon=0.0, device="cpu"):
#     done_reward = None
#     rewards = []
#     actions = []
#     int_rewards = []
#     ext_rewards = []
#     obs, _ = env.reset()
#     done = False
#     trunc = False
#     while not done and not trunc:
#         if np.random.random() < epsilon:
#             action = env.action_space.sample()
#         else:
#             #Get the behaviour action
#             obs_a = np.array(obs, copy=False)
#             obs_v = torch.FloatTensor(np.expand_dims(obs_a, axis=0)).to(device)
#             q_vals_v = net_behave(obs_v)
#             _, act_v = torch.max(q_vals_v, dim=1)
#             action = int(act_v.item())

#         actions.append(action)

#         # do step in the environment
#         new_obs, reward, done, trunc, info = env.step(action)
#         rewards.append(reward)
#         int_rewards.append(info['int_reward'])
#         ext_rewards.append(info['ext_reward'])

#         obs = new_obs
    
#     return actions, rewards, int_rewards, ext_rewards

@torch.no_grad()
def eval_policy_expander_agent(agent, env):
    rewards = []
    actions = []
    int_rewards = []
    ext_rewards = []
    obs, _ = env.reset()
    done = False
    trunc = False
    while not done and not trunc:
        action = agent([obs])
        actions.append(action[0][0])
        # do step in the environment
        new_obs, reward, done, trunc, info = env.step(action[0][0])
        rewards.append(reward)
        int_rewards.append(info['int_reward'])
        ext_rewards.append(info['ext_reward'])
        obs = new_obs
    #
    return actions, rewards, int_rewards, ext_rewards

@torch.no_grad()
def eval_policy_skipper_agent(agent, env):
    rewards = []
    actions = []
    int_rewards = []
    ext_rewards = []
    obs, _ = env.reset()
    done = False
    trunc = False
    while not done and not trunc:
        action = agent([obs])
        actions.append(action[0][0])
        # do step in the environment
        new_obs, reward, done, trunc, info = env.step(action[0][0])
        rewards.append(reward)
        int_rewards.append(info['int_reward'])
        ext_rewards.append(info['ext_reward'])
        obs = new_obs
    #
    return actions, rewards, int_rewards, ext_rewards




# @torch.no_grad()
# def eval_policy_expander_agent(agent, env):
#     rewards = []
#     actions = []
#     int_rewards = []
#     ext_rewards = []
#     obs, _ = env.reset()
#     done = False
#     trunc = False
#     while not done and not trunc:
#         action = agent([obs])
#         actions.append(action[0][0])
#         # do step in the environment
#         new_obs, reward, done, trunc, info = env.step(action[0][0])
#         rewards.append(reward)
#         int_rewards.append(info['int_reward'])
#         ext_rewards.append(info['ext_reward'])
#         obs = new_obs
#     #
#     return actions, rewards, int_rewards, ext_rewards


# @torch.no_grad()
# def eval_policy_skipper(net_behave, net_skip, env, epsilon=0.0, device="cpu"):
#     done_reward = None
#     rewards = []
#     actions = []
#     int_rewards = []
#     ext_rewards = []
#     obs, _ = env.reset()
#     done = False
#     trunc = False
#     while not done and not trunc:
#         if np.random.random() < epsilon:
#             action = env.action_space.sample()
#         else:
#             #Get the behaviour action
#             obs_a = np.array(obs, copy=False)
#             obs_v = torch.FloatTensor(np.expand_dims(obs_a, axis=0)).to(device)
#             q_vals_v = net_behave(obs_v)
#             _, act_v = torch.max(q_vals_v, dim=1)
#             action = int(act_v.item())

#             #Get the skipper action
#             obs_skip_a = np.array(obs, copy=False)
#             obs_v_sk_np = np.append(obs_skip_a,[action])
#             obs_v_sk = torch.FloatTensor(np.expand_dims(obs_v_sk_np, axis=0)).to(device)
#             q_vals_v = net_skip(obs_v_sk)
#             _, act_v = torch.max(q_vals_v, dim=1)
#             rep = int(act_v.item())
#             action = (rep, action)

#         actions.append(action)

#         # do step in the environment
#         new_obs, reward, done, trunc, info = env.step(action)
#         rewards.append(reward)
#         int_rewards.append(info['int_reward'])
#         ext_rewards.append(info['ext_reward'])

#         obs = new_obs
    
#     return actions, rewards, int_rewards, ext_rewards


