import torch
import ptan
from source.models import dqn_model
from source.helpers import experience
import gym
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple, List



@torch.no_grad()
def unpack_expander_batch(batch, tgt_net_behave, gamma, device='cpu', net_behave=None):
    behave_states = []
    behave_actions = []
    rewards = []
    done_masks = []
    last_behave_states = []
    for exp in batch:
        behave_states.append(exp.state)
        behave_actions.append(exp.action)
        rewards.append(exp.reward)
        done_masks.append(exp.last_state is None)
        if exp.last_state is None:
            last_behave_states.append(exp.state)
        else:
            last_behave_states.append(exp.last_state)
    behave_states_v = torch.tensor(np.array(behave_states)).to(device)
    behave_actions_v = torch.tensor(np.array(behave_actions)).to(device)
    rewards_v = torch.tensor(np.array(rewards),dtype=torch.float32).to(device)
    last_states_v = torch.tensor(np.array(last_behave_states)).to(device)
    if net_behave is not None:
        last_state_q_v = net_behave(last_states_v)
        best_last_q_a = torch.argmax(last_state_q_v, dim=1)
        best_last_q_v = tgt_net_behave(last_states_v).gather(1, best_last_q_a.unsqueeze(dim=1)).squeeze(-1)
    else: 
        last_state_q_v = tgt_net_behave(last_states_v)
        best_last_q_v = torch.max(last_state_q_v, dim=1)[0]
    best_last_q_v[done_masks] = 0.0
    #
    return behave_states_v, behave_actions_v, best_last_q_v.detach() * gamma + rewards_v


@torch.no_grad()
def unpack_skipper_batch(batch, tgt_net_behave, tgt_net_skip, gamma, device='cpu', net_behave=None, net_skip=None):
    behave_states = []
    behave_actions = []
    skip_states = []
    skip_actions = []
    rewards = []
    done_masks = []
    last_behave_states = []
    last_skip_states = []
    for exp in batch:
        behave_states.append(exp.state)
        behave_actions.append(exp.action[1])
        #
        skip_actions.append(exp.action[0])
        rewards.append(exp.reward)
        done_masks.append(exp.last_state is None)
        if exp.last_state is None:
            last_behave_states.append(exp.state)
        else:
            last_behave_states.append(exp.last_state)
    behave_states_v = torch.tensor(np.array(behave_states)).to(device)
    behave_actions_v = torch.tensor(np.array(behave_actions)).to(device)
    rewards_v = torch.tensor(np.array(rewards),dtype=torch.float32).to(device)
    last_states_v = torch.tensor(np.array(last_behave_states)).to(device)
    if net_behave is not None and net_skip is not None:
        last_state_q_v = net_behave(last_states_v)
        best_last_q_a = torch.argmax(last_state_q_v, dim=1)
        best_last_q_v = tgt_net_behave(last_states_v).gather(1, best_last_q_a.unsqueeze(dim=1)).squeeze(-1)
    else:
        last_state_q_v = tgt_net_behave(last_states_v)
        best_last_q_v = torch.max(last_state_q_v, dim=1)[0]
        best_last_q_a = torch.argmax(last_state_q_v, dim=1)
    best_last_q_v[done_masks] = 0.0
    #
    skip_states_v = torch.tensor(np.array(skip_states)).to(device)
    skip_actions_v = torch.tensor(np.array(skip_actions)).to(device)
    obs_shape = behave_states_v.shape
    if len(obs_shape) == 4: # for atari (image)
        size = obs_shape[3]
        N = obs_shape[0]
        a = torch.tensor(behave_actions_v, dtype=torch.float32).view(N, 1, 1, 1)
        skip_states_v = torch.cat((behave_states_v, a.expand(N, 4, 1, size)), dim=2)    
        a = torch.tensor(best_last_q_a, dtype=torch.float32).view(N, 1, 1, 1)
        last_skip_states = torch.cat((last_states_v, a.expand(N, 4, 1, size)), dim=2)    
    else: # vector observation
        skip_states_v = torch.cat((behave_states_v, torch.reshape(behave_actions_v, (behave_states_v.shape[0],1))), dim=1)
        last_skip_states = torch.cat((last_states_v, torch.reshape(best_last_q_a, (last_states_v.shape[0],1))), dim=1)
    if net_behave is not None and net_skip is not None:
        last_state_q_v = net_skip(last_skip_states)
        best_last_q_a = torch.argmax(last_state_q_v, dim=1)
        best_skip_last_q_v = tgt_net_skip(last_skip_states).gather(1, best_last_q_a.unsqueeze(dim=1)).squeeze(-1)
    else:
        last_state_q_v = tgt_net_skip(last_skip_states)
        best_skip_last_q_v = torch.max(last_state_q_v, dim=1)[0]
    best_skip_last_q_v[done_masks] = 0.0

    return behave_states_v, behave_actions_v, best_last_q_v.detach() * gamma + rewards_v, skip_states_v, skip_actions_v,  best_skip_last_q_v.detach() * gamma + rewards_v



def calc_loss_skipper_dqn(batch, net_behave, tgt_net_behave, net_skip, tgt_net_skip, gamma, doubleQ=0, device="cpu"):
    if doubleQ == 1:
        behave_states_v, behave_actions_v, behave_tgt_q_v, skip_states_v, skip_actions_v, skip_tgt_q_v  = unpack_skipper_batch(batch, tgt_net_behave.target_model, tgt_net_skip.target_model, gamma, device, net_behave, net_skip)
    else:
        behave_states_v, behave_actions_v, behave_tgt_q_v, skip_states_v, skip_actions_v, skip_tgt_q_v  = unpack_skipper_batch(batch, tgt_net_behave.target_model, tgt_net_skip.target_model, gamma, device)
    #
    q_v_behave = net_behave(behave_states_v)
    q_v_behave = q_v_behave.gather(1, behave_actions_v.unsqueeze(-1)).squeeze(-1)
    #
    q_v_skip = net_skip(skip_states_v)
    q_v_skip = q_v_skip.gather(1, skip_actions_v.unsqueeze(-1)).squeeze(-1)
    #
    return nn.MSELoss()(q_v_behave, behave_tgt_q_v), nn.MSELoss()(q_v_skip, skip_tgt_q_v)

def calc_loss_prio_skipper_dqn(batch, batch_weights, net_behave, tgt_net_behave, net_skip, tgt_net_skip,  gamma,  doubleQ=0, device="cpu"):
    if doubleQ == 1:
        behave_states_v, behave_actions_v, behave_tgt_q_v, skip_states_v, skip_actions_v, skip_tgt_q_v  = unpack_skipper_batch(batch, tgt_net_behave.target_model, tgt_net_skip.target_model, gamma, device, net_behave, net_skip)
    else:
        behave_states_v, behave_actions_v, behave_tgt_q_v, skip_states_v, skip_actions_v, skip_tgt_q_v  = unpack_skipper_batch(batch, tgt_net_behave.target_model, tgt_net_skip.target_model, gamma, device)
    
    batch_weights_v = torch.tensor(batch_weights).to(device)
    #
    q_v_behave = net_behave(behave_states_v)
    q_v_behave = q_v_behave.gather(1, behave_actions_v.unsqueeze(-1)).squeeze(-1)
    #
    q_v_skip = net_skip(skip_states_v)
    q_v_skip = q_v_skip.gather(1, skip_actions_v.unsqueeze(-1)).squeeze(-1)
    #
    bh_losses_v = batch_weights_v * (q_v_behave - behave_tgt_q_v) ** 2
    sk_losses_v = batch_weights_v * (q_v_skip - skip_tgt_q_v) ** 2
    return bh_losses_v.mean(), (bh_losses_v + 1e-5).data.cpu().numpy(), sk_losses_v.mean(), (sk_losses_v + 1e-5).data.cpu().numpy()

def calc_loss_expander_dqn(batch, net_behave, tgt_net_behave,  gamma, doubleQ=0, device="cpu"):
    if doubleQ == 1:
        behave_states_v, behave_actions_v, behave_tgt_q_v,  = unpack_expander_batch(batch, tgt_net_behave.target_model, gamma, device, net_behave)
    else:
        behave_states_v, behave_actions_v, behave_tgt_q_v,  = unpack_expander_batch(batch, tgt_net_behave.target_model, gamma, device)
    #
    q_v_behave = net_behave(behave_states_v)
    q_v_behave = q_v_behave.gather(1, behave_actions_v.unsqueeze(-1)).squeeze(-1)
    #
    return nn.MSELoss()(q_v_behave, behave_tgt_q_v)

def calc_loss_prio_expander_dqn(batch, batch_weights, net_behave, tgt_net_behave, gamma,  doubleQ=0, device="cpu"):
    if doubleQ == 1:
        behave_states_v, behave_actions_v, behave_tgt_q_v  = unpack_expander_batch(batch, tgt_net_behave.target_model, gamma, device, net_behave)
    else:
        behave_states_v, behave_actions_v, behave_tgt_q_v  = unpack_expander_batch(batch, tgt_net_behave.target_model, gamma, device)
    
    batch_weights_v = torch.tensor(batch_weights).to(device)
    #
    q_v_behave = net_behave(behave_states_v)
    q_v_behave = q_v_behave.gather(1, behave_actions_v.unsqueeze(-1)).squeeze(-1)
    #
    bh_losses_v = batch_weights_v * (q_v_behave - behave_tgt_q_v) ** 2
    return bh_losses_v.mean(), (bh_losses_v + 1e-5).data.cpu().numpy()


class EpsilonTracker:
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector,
                 epsilon_start: float,
                 epsilon_final: float,
                 epsilon_frames: int):
        self.selector = selector
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_frames = epsilon_frames
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.epsilon_start - frame_idx / self.epsilon_frames
        self.selector.epsilon = max(self.epsilon_final, eps)

def batch_generator(buffer: experience.ExperienceReplayBuffer,
                    initial: int, 
                    batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)

