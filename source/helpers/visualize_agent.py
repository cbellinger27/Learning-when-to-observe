import gym
import pygame
from pygame.locals import *
import os
import sys
sys.path.append('../')
sys.path.append('.')

import torch
import numpy as np
import gym
from source.wrapper import skipper
from source.wrapper import expander
from wrapper import atari
import argparse
from models import dqn_model
import time
import ptan

WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 800
FPS = 1000
text_colour = (1, 1, 1)
#for lunar
# text_colour = (255, 255, 255)

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
        policy_behave.load_state_dict(torch.load(os.path.join(run_dir, detailed_name+'dqn_expander_policy2.torch')))

@torch.no_grad()
def eval_policy_skipper(agent, env, pygame, window, clock, font):

    done_reward = None
    rewards = []
    actions = []
    int_rewards = []
    ext_rewards = []
    font_score = pygame.font.Font(None, 30)

    for e in range(10):
        done = False
        trunc = False
        rewards_tmp = []
        int_rewards_tmp = []
        ext_rewards_tmps = []
        obs, _ = env.reset()
        step = 0
        tot_step = 0
        tot_skip = 0
        while not done and not trunc:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    exit()
            
            # env.render()
            time.sleep(0.3)
            action = agent([obs])
            # print("repeat x %i times" %rep)
            actions.append(action)
            
            tot_step += action[0][0][0] + 1
            tot_skip += action[0][0][0]

            # do step in the environment
            new_obs, reward, done, trunc, info = env.step(action[0][0])

            rewards_tmp.append(reward)
            int_rewards_tmp.append(info['int_reward'])
            ext_rewards_tmps.append(info['ext_reward'])
            step += info['tot_steps']

            # Render the environment
            window.fill((0, 0, 0))

            # Convert the environment's rendered image to a Pygame surface
            rendered_image = pygame.surfarray.make_surface(env.render())
            rendered_image = pygame.transform.scale(rendered_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
            # rendered_image = pygame.transform.rotate(rendered_image, -90)

            # Draw the rendered image on the window
            window.blit(rendered_image, (0, 0))

            # Display the total reward as text on the window
            text = font.render('Action %i: do not measure x %i, measure x %i' %((action[0][0][1]), (action[0][0][0]), 1), True, text_colour)
            temp_surface = pygame.Surface(text.get_size())
            temp_surface.fill((255, 255, 255))
            # window.blit(text, (200, 10))
            window.blit(text, (20, 10))

            text2 = font_score.render('Total No Measure: %i:' %tot_skip, True, text_colour)
            temp_surface = pygame.Surface(text2.get_size())
            temp_surface.fill((255, 255, 255))
            window.blit(text2, (10, 150))

            text3 = font_score.render('Total Steps: %i:' %tot_step, True, text_colour)
            temp_surface = pygame.Surface(text3.get_size())
            temp_surface.fill((255, 255, 255))
            window.blit(text3, (10, 100))

            if done:
                if reward >= 100:
                    text4 = font.render('Successful Landing!', True, 'red')
                    temp_surface = pygame.Surface(text4.get_size())
                    temp_surface.fill((255, 255, 255))
                    window.blit(text4, (250, 550))
                else:    
                    text4 = font.render('Episode done!', True, 'red')
                    temp_surface = pygame.Surface(text4.get_size())
                    temp_surface.fill((255, 255, 255))
                    window.blit(text4, (250, 550))
            

            pygame.display.update()
            clock.tick(FPS)
            if done:
                time.sleep(1)
            if reward >= 100:
                time.sleep(5)

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
def eval_policy_expander(agent, env,  pygame, window, clock, font):
    # load_expander_policy(net_behave, path, device=device, detailed_name=detailed_name)

    done_reward = None
    rewards = []
    actions = []
    int_rewards = []
    ext_rewards = []
    font_score = pygame.font.Font(None, 30)

    for e in range(10):
        done = False
        trunc = False
        rewards_tmp = []
        int_rewards_tmp = []
        ext_rewards_tmps = []
        obs, _ = env.reset()
        step = 0

        tot_step = 0
        tot_skip = 0
        while not done and not trunc:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    exit()
            
            # env.render()
            time.sleep(0.2)

            action = agent([obs])[0]

            actions.append(action)

            # do step in the environment
            print(action)
            new_obs, reward, done, trunc, info = env.step(action[0])
            step += 1
            tot_skip += info['measure']
            tot_step +=1

            rewards_tmp.append(reward)
            
            # Render the environment
            window.fill((0, 0, 0))

            # Convert the environment's rendered image to a Pygame surface
            rendered_image = pygame.surfarray.make_surface(env.render())
            rendered_image = pygame.transform.scale(rendered_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
            # rendered_image = pygame.transform.rotate(rendered_image, -90)

            # Draw the rendered image on the window
            window.blit(rendered_image, (0, 0))

            # Display the total reward as text on the window
            text = font.render('Action %i: do not measure (0) / measure (1) =   %i' %((action[0]), info['measure']), True, text_colour)
            temp_surface = pygame.Surface(text.get_size())
            temp_surface.fill((255, 255, 255))
            # window.blit(text, (200, 10))
            window.blit(text, (20, 10))

            text2 = font_score.render('Total No Measure: %i:' %tot_skip, True, text_colour)
            temp_surface = pygame.Surface(text2.get_size())
            temp_surface.fill((255, 255, 255))
            window.blit(text2, (10, 150))

            text3 = font_score.render('Total Steps: %i:' %tot_step, True, text_colour)
            temp_surface = pygame.Surface(text3.get_size())
            temp_surface.fill((255, 255, 255))
            window.blit(text3, (10, 100))

            if done:
                if reward >= 100:
                    text4 = font.render('Successful Landing!', True, 'red')
                    temp_surface = pygame.Surface(text4.get_size())
                    temp_surface.fill((255, 255, 255))
                    window.blit(text4, (250, 550))
                else:    
                    text4 = font.render('Episode done!', True, 'red')
                    temp_surface = pygame.Surface(text4.get_size())
                    temp_surface.fill((255, 255, 255))
                    window.blit(text4, (250, 550))
            

            pygame.display.update()
            clock.tick(FPS)
            if done:
                time.sleep(1)
            if reward >= 100:
                time.sleep(5)

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
    # Set up the Pygame window
    pygame.init()
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    # Set up the font
    font = pygame.font.Font(None, 50)
        
    if args.agent_style == 'skipper':
        env = skipper.make_env(args.env_name, args.obs_cost, args.skip_steps, False, args.full_ext_reward, render_mode='rgb_array', max_episode_steps=args.max_episode_steps, verbose=False)
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
        eval_policy_skipper(agent, env, pygame, window, clock, font)
    else: 
        env = expander.make_env(args.env_name, args.obs_cost, True, False, False,args.full_ext_reward, render_mode='rgb_array', max_episode_steps=args.max_episode_steps, verbose=True)
        obs_size = env.observation_space.shape[0]
        n_actions_behave = env.action_space.n
        if args.env_name == 'PongNoFrameskip-v4':
            env = atari.make_expander_env(env)
            obs_size = env.observation_space.shape
            net_behave = dqn_model.DQN_CNN(obs_size,n_actions_behave).to(args.device)
            print(net_behave)
        else:
            net_behave = dqn_model.DQN_MLP(obs_size,n_actions_behave,args.hidden_size).to(args.device)
        
        selector = ptan.actions.ArgmaxActionSelector()
        load_expander_policy(net_behave, args.path, device=args.device, detailed_name=args.policy_prefix+"best_")
        agent = dqn_model.ExpanderAgent(net_behave, selector, args.device)

        eval_policy_expander(agent, env, pygame, window, clock, font)



if __name__ == '__main__':
    args = create_parser().parse_args()
    # args.env_name = 'Acrobot-v1'
    # args.policy_prefix = 'acro_'
    # args.env_name = 'CartPole-v1'
    # args.policy_prefix = 'cart_'
    # args.env_name = 'LunarLander-v2'
    # args.policy_prefix = 'lunar_'
    args.env_name = 'PongNoFrameskip-v4'
    args.policy_prefix = 'pong_'
    args.max_episode_steps =100000
    args.agent_style = 'skipper'
    args.device = 'cuda'
    main(args)