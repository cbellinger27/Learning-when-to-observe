# %%
#!/usr/bin/env python3
import gym
import sys
import os
sys.path.append('../')
sys.path.append('.')
                
from source.wrapper.expander import make_env
from source.wrapper import atari

from source.models import dqn_model
from source.helpers import experience
from source.helpers import common

import argparse
import numpy as np
# import collections

import ptan
import torch
import torch.nn as nn
import torch.optim as optim


import wandb
from source.helpers.cli import add_wandb_args, add_device_args, add_amrl_args
from source.helpers.log import  log_eval_expander_agent

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_wandb_args(parser)
    add_device_args(parser)
    add_amrl_args(parser)

    parser.add_argument('--env_name', default='CartPole-v1', help='Which environment to use?')
    parser.add_argument('--solved_score', default=220, type=int, help='Solved score for early stopping')
    parser.add_argument('--wrapper', default='Expander', type=str, help="Leave as default. Used for sorting results.")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--win_reward', type=float)
    
    # Dqn special
    parser.add_argument('--total_timesteps', type=int, default=65000)
    parser.add_argument('--batch_size', type=int, default=64, help="Batch of the sample data.")
    parser.add_argument('--eps_test', type=float, default=0.01)
    parser.add_argument('--eps_train', type=float, default=1)
    parser.add_argument('--eps_decay', type=int, default=15000)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--train_start', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target_update_freq', type=int, default=100)
    parser.add_argument('--n_step', type=int, default=1)
    parser.add_argument('--step_per_collect', type=int, default=1)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--doubleQ', type=int, choices=[0,1], default=1)
    parser.add_argument('--noisyNet', type=int, choices=[0,1], default=0)
    parser.add_argument('--prioReplay', type=int, choices=[0,1], default=0)
    parser.add_argument('--prioAlpha', type=float, default=0.6, help="Larger value puts more emphasis higher priority tuples.")
    parser.add_argument('--prioBeta', type=float, default=0.4, help="Componsates for the bias introduce due non-iid sampling. beta=1 fully componsates for the bias")
    parser.add_argument('--prioBetaFrames', type=int, default=30000, help="Number of frames to increment beta over")
    parser.add_argument('--recurrent', type=int, choices=[0,1], default=0, help="Recurrent or not")

    # parser.add_argument('--run_dir', type=str, default="./")
    return parser


def main(args):
    with wandb.init(project=args.project_name, entity='cbellinger', mode=args.wandb_mode, config=args) as run:
        wandb.config['run_dir'] = run.dir
        wandb.config['sweep_id'] = run.sweep_id
        wandb.config['id'] = run.id
        
        results, expander_agent, env = train(wandb.config)
        after_training(wandb.config, results, expander_agent, env, "lastTrained_")
        load_policy(expander_agent.behave_model, run.dir, detailed_name="best_")
        after_training(wandb.config, results, expander_agent, env, "best_")

def after_training(args, result, expander_agent, env,detailed_name="lastTrained_"):
    log_eval_expander_agent(args, env, expander_agent, detailed_name)

def save_fn(policy_behave, run_dir, detailed_name="best_"):
    torch.save(policy_behave.state_dict(), os.path.join(run_dir, detailed_name+'dqn_behave_policy.torch'))

def load_policy(policy_behave, run_dir, detailed_name="best_"):
    policy_behave.load_state_dict(torch.load(os.path.join(run_dir, detailed_name+'dqn_behave_policy.torch')))

def train(args):
    print(args.env_name)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.env_name == "PongNoFrameskip-v4":
        env = make_env(args.env_name, args.obs_cost, True, False, False, args.full_ext_reward, max_episode_steps=None)
        env = atari.make_expander_env(env)
        obs_size = env.observation_space.shape
    else:
        env = make_env(args.env_name, args.obs_cost, True, False, False, args.full_ext_reward, max_episode_steps=args.max_episode_steps)   
        obs_size = env.observation_space.shape[0]
    
    n_actions_behave = env.action_space.n

    if args.noisyNet == 1:
        net_behave = dqn_model.NoisyMLP_DQN(obs_size,n_actions_behave,args.hidden_size).to(args.device)
        tgt_net_behave = ptan.agent.TargetNet(net_behave)
        selector = ptan.actions.ArgmaxActionSelector()
    else:
        if args.recurrent == 1:
            net_behave = dqn_model.Recurrent(1,obs_size,n_actions_behave, args.hidden_size, device=args.device).to(args.device)
        else:
            if args.env_name == "PongNoFrameskip-v4":
                net_behave = dqn_model.DQN_CNN(obs_size,n_actions_behave).to(args.device)
            else:
                net_behave = dqn_model.DQN_MLP(obs_size,n_actions_behave,args.hidden_size).to(args.device)
        tgt_net_behave = ptan.agent.TargetNet(net_behave)
        selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=args.eps_train)
        epsilon_tracker = common.EpsilonTracker(selector, args.eps_train, args.eps_test, args.eps_decay)
    agent = dqn_model.ExpanderAgent(net_behave, selector, args.device)

    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=args.gamma, steps_count=args.n_step) #n_step dqn
    if args.prioReplay == 1:
        buffer = experience.PrioritizedReplayBuffer(exp_source, buffer_size=args.buffer_size, alpha=args.prioAlpha)
    else:
        buffer = experience.ExperienceReplayBuffer(exp_source, buffer_size=args.buffer_size)

    optimizer_behave = optim.Adam(net_behave.parameters(), args.lr)

    total_rewards = []
    mean_rewards = []
    total_ext_rewards = []
    total_int_rewards = []
    total_measures = []
    total_noMeasures = []
    frame_idx = 0
    best_ep_reward = None
    steps_in_episode = []
    total_landings = 0
    episode_num = 0

    # load a previous policy
    if args.resume_path:
        load_policy(agent.behave_model, args.resume_path, detailed_name="")
        print("Loaded agent from: ", args.resume_path)

    beta = 0
    while frame_idx < args.total_timesteps:
        frame_idx += 1
        buffer.populate(args.step_per_collect) #steps to collect per policy update 
        if args.prioReplay == 1:
            beta = min(1.0, args.prioBeta + frame_idx * (1.0 - args.prioBeta) / args.prioBetaFrames)
        for reward, ext_rewards, int_rewards, steps, last_rewards, fullObs_rewards, measures in exp_source.pop_verbose_rewards_steps():
            episode_num += 1
            total_rewards.append(reward)
            total_ext_rewards.append(ext_rewards) 
            total_int_rewards.append(int_rewards) 
            steps_in_episode.append(steps) 
            total_measures.append(measures)
            total_noMeasures.append(steps-measures)
            mean_rewards.append(np.mean(total_rewards[-100:]))
            wandb.log({'int_reward': int_rewards, 'episode':episode_num, 'env_step':frame_idx})
            wandb.log({'ext_reward': ext_rewards, 'episode':episode_num, 'env_step':frame_idx})
            wandb.log({'reward': reward, 'episode':episode_num, 'env_step':frame_idx})
            wandb.log({'fullObs_rewards': fullObs_rewards, 'episode':episode_num, 'env_step':frame_idx})
            wandb.log({'measures': measures, 'episode':episode_num, 'env_step':frame_idx})
            wandb.log({'no measures': steps-measures, 'episode':episode_num, 'env_step':frame_idx})
            wandb.log({'epsiode_length': steps, 'episode':episode_num, 'env_step':frame_idx})
            wandb.log({'beta': beta, 'episode':episode_num, 'env_step':frame_idx})
            wandb.log({'epsilon': epsilon_tracker.selector.epsilon, 'episode':episode_num, 'env_step':frame_idx})
            if args.win_reward is not None:
                if last_rewards >= args.win_reward:
                    total_landings += 1
            if args.env_name == 'LunarLander-v2' or args.env_name == 'PongNoFrameskip-v4':
                wandb.log({'sum_of_wins': total_landings, 'episode':episode_num, 'env_step':frame_idx})
            if episode_num % 5 == 0:
                eps = 0
                if args.noisyNet == 0:
                    eps = selector.epsilon
                print("%d: episode %d steps %d done, reward=%.3f, epsilon=%.2f, mean reward=%.2f" % (frame_idx, episode_num, steps, reward, eps, np.mean(total_rewards[-50:])))
            if best_ep_reward is None or best_ep_reward < reward:
                save_fn(net_behave, args.run_dir, "best_")
                best_ep_reward = reward
        if len(buffer) < args.train_start:
            continue

        optimizer_behave.zero_grad()
        if args.prioReplay == 1:
            batch, batch_indices, batch_weights = buffer.sample(args.batch_size, beta)
            loss_v_behave, sample_prios_bh_v = common.calc_loss_prio_expander_dqn(batch, batch_weights, net_behave, tgt_net_behave, args.gamma**args.n_step, device=args.device, doubleQ=args.doubleQ)
        else:
            batch = buffer.sample(args.batch_size)
            loss_v_behave = common.calc_loss_expander_dqn(batch, net_behave, tgt_net_behave, args.gamma**args.n_step, device=args.device, doubleQ=args.doubleQ)
        
        loss_v_behave.backward()
        optimizer_behave.step()
        if args.prioReplay == 1:
            buffer.update_priorities(batch_indices, sample_prios_bh_v)
        #
        if frame_idx % args.target_update_freq == 0:
            tgt_net_behave.sync()
        #
        if args.noisyNet == 0:
            epsilon_tracker.frame(frame_idx)
    save_fn(net_behave, args.run_dir,detailed_name="lastTrained_")
    wandb.log({'best_episode_reward': best_ep_reward})
    wandb.log({'total measures': np.sum(total_measures)})
    wandb.log({'total no measures': np.sum(total_noMeasures)})
    if args.env_name == 'LunarLander-v2' or args.env_name == 'PongNoFrameskip-v4':
        wandb.log({'total wins': total_landings})
    wandb.log({'mean episode length': np.sum(steps_in_episode)/episode_num})
    return dict({'rewards':total_rewards, 'mean_rewards':mean_rewards, 'best_ep_reward':best_ep_reward}), agent, env


if __name__ == '__main__':
    args = create_parser().parse_args()
    # args.max_episode_steps = 200
    # args.win_reward = 210
    main(args)