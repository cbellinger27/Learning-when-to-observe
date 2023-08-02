import os
import sys
sys.path.append('../')
sys.path.append('.')

import torch
import numpy as np
import gym
from source.wrapper import skipper
from source.wrapper import expander
from source.wrapper import atari
import argparse
from source.models import dqn_model
import time
import ptan

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_name', type=str, default='LunarLander-v2')
    parser.add_argument('--policy_prefix', type=str, default='lunar_')
    parser.add_argument('--agent_style', type=str, choices= ['skipper', 'expander'], default='skipper')
    parser.add_argument('--obs_cost', type=float, default=0)
    parser.add_argument('--full_ext_reward', type=int, choices=[0,1], default=0)
    parser.add_argument('--skip_steps', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--path', type=str, default='policies/')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--max_episode_steps', type=int, default=900)
    
    return parser

def load_skipper_policy(policy_behave, policy_skip, run_dir, device='cuda', detailed_name="best_"):
    if device == 'cpu':
        policy_behave.load_state_dict(torch.load(os.path.join(run_dir, detailed_name+'dqn_behave_policy.torch'), map_location=torch.device('cpu')))
        policy_skip.load_state_dict(torch.load(os.path.join(run_dir, detailed_name+'dqn_skip_policy.torch'), map_location=torch.device('cpu')))
    else:
        policy_behave.load_state_dict(torch.load(os.path.join(run_dir, detailed_name+'dqn_behave_policy.torch')))
        policy_skip.load_state_dict(torch.load(os.path.join(run_dir, detailed_name+'dqn_skip_policy.torch')))

def load_expander_policy(policy_behave, run_dir, device='cuda', detailed_name="best_"):
    if device == 'cpu':
        policy_behave.load_state_dict(torch.load(os.path.join(run_dir, detailed_name+'dqn_expander_policy.torch'), map_location=torch.device('cpu')))
    else:
        policy_behave.load_state_dict(torch.load(os.path.join(run_dir, detailed_name+'dqn_expander_policy4.torch')))
 
@torch.no_grad()
def eval_policy_skipper(agent, env):

    done_reward = None
    rewards = []
    actions = []
    int_rewards = []
    ext_rewards = []

    for e in range(10):
        done = False
        trunc = False
        rewards_tmp = []
        int_rewards_tmp = []
        ext_rewards_tmps = []
        obs, _ = env.reset()
        step = 0
        while not done and not trunc:
            step+=1
            env.render()
            time.sleep(0.6)
            action = agent([obs])
            # print("repeat x %i times" %rep)
            actions.append(action)

            # do step in the environment
            new_obs, reward, done, trunc, info = env.step(action[0][0])
            rewards_tmp.append(reward)
            int_rewards_tmp.append(info['int_reward'])
            ext_rewards_tmps.append(info['ext_reward'])
            step += info['tot_steps']

            obs = new_obs
            if done:
                print("Done with reward: " + str(reward) + " at step " + str(step))
            if trunc:
                print("Truc at step " + str(step))
        rewards.append(np.mean(rewards_tmp))
        int_rewards.append(np.mean(int_rewards_tmp))
        ext_rewards.append(np.mean(ext_rewards_tmps))
        print(rewards)
    
    return actions, rewards, int_rewards, ext_rewards

@torch.no_grad()
def eval_policy_expander(net_behave, path, env, epsilon=0.0, device="cpu", detailed_name="best_"):
    load_expander_policy(net_behave, path, device=device, detailed_name=detailed_name)

    done_reward = None
    rewards = []
    actions = []
    int_rewards = []
    ext_rewards = []

    for e in range(10):
        done = False
        trunc = False
        rewards_tmp = []
        int_rewards_tmp = []
        ext_rewards_tmps = []
        obs, _ = env.reset()
        step = 0
        while not done and not trunc:
            env.render()
            step+=1
            time.sleep(1)
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                #Get the behaviour action
                obs_a = np.array(obs, copy=False)
                obs_v = torch.FloatTensor(np.expand_dims(obs_a, axis=0)).to(device)
                q_vals_v = net_behave(obs_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                action = int(act_v.item())

            # if action < env.env.action_space.n/2:
            #     print("don't measure")
            # else:
            #     print('measure')
            actions.append(action)

            # do step in the environment
            new_obs, reward, done, trunc, info = env.step(action)
            rewards_tmp.append(reward)
            int_rewards_tmp.append(info['int_reward'])
            ext_rewards_tmps.append(info['ext_reward'])
            # print(info)
            obs = new_obs
            if done:
                print("Done with reward: " + str(reward) + " at step " + str(step))
            if trunc:
                print("Truc at step " + str(step))
        rewards.append(np.mean(rewards_tmp))
        int_rewards.append(np.mean(int_rewards_tmp))
        ext_rewards.append(np.mean(ext_rewards_tmps))
        print(rewards)
    
    return actions, rewards, int_rewards, ext_rewards

def main(args):
    if args.agent_style == 'skipper':
        env = skipper.make_env(args.env_name, args.obs_cost, args.skip_steps, False, args.full_ext_reward, render_mode='human', max_episode_steps=args.max_episode_steps, verbose=True)
        obs_size = env.observation_space.shape[0]
        n_actions_behave = env.action_space.spaces[1].n
        n_actions_skipper = env.action_space.spaces[0].n
        if args.env_name == 'PongNoFrameskip-v4':
            env = atari.make_skipper_env(env)
            obs_size = env.observation_space.shape
            net_behave = dqn_model.DQN_CNN(obs_size,n_actions_behave).to(args.device)
            net_skip = dqn_model.DQN_CNN((obs_size[0], obs_size[1]+1, obs_size[2]),n_actions_skipper).to(args.device)
            print(net_behave)
        else:
            net_behave = dqn_model.DQN_MLP(obs_size,n_actions_behave,args.hidden_size).to(args.device)
            net_skip = dqn_model.DQN_MLP(obs_size+1,n_actions_skipper,args.hidden_size).to(args.device)
        
        selector = ptan.actions.ArgmaxActionSelector()
        load_skipper_policy(net_behave, net_skip, args.path, device=args.device, detailed_name=args.policy_prefix+"best_")
        agent = dqn_model.SkipperAgent(net_behave, net_skip, selector, args.device)
        eval_policy_skipper(agent, env)
    else: 
        env = expander.make_env(args.env_name, args.obs_cost, True, False, False,args.full_ext_reward, render_mode='human', max_episode_steps=args.max_episode_steps, verbose=True)
        obs_size = env.observation_space.shape[0]
        n_actions_behave = env.action_space.n
        if args.env_name == 'PongNoFrameskip-v4':
            env = atari.make_expander_env(env)
            obs_size = env.observation_space.shape
            net_behave = dqn_model.DQN_CNN(obs_size,n_actions_behave).to('cuda')
            print(net_behave)
        else:
            net_behave = dqn_model.DQN_MLP(obs_size,n_actions_behave,args.hidden_size).to(args.device)

        eval_policy_expander(net_behave, args.path, env, epsilon=0, device=args.device, detailed_name=args.policy_prefix+"best_")



if __name__ == '__main__':
    args = create_parser().parse_args()
    # args.env_name = 'Acrobot-v1'
    # args.policy_prefix = 'acro_'
    # args.env_name = 'CartPole-v1'
    # args.policy_prefix = 'cart_'
    # args.policy_prefix = 'lunar_'
    args.env_name = 'PongNoFrameskip-v4'
    args.policy_prefix = 'pong_'
    args.max_episode_steps =100000
    args.agent_style = 'skipper'
    args.device = 'cuda'
    main(args)