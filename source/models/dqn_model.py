from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)

import torch
import torch.nn as nn
import numpy as np
import ptan
import math

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],Sequence[Dict[Any, Any]]]

def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)


def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)


class SkipperAgent(ptan.agent.BaseAgent):
    """
    Skipper Agent is a memoryless DQN agent which calculates Q values and skip steps
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, behave_model, skip_model, action_selector, device="cpu", preprocessor=default_states_preprocessor):
        self.behave_model = behave_model
        self.skip_model = skip_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        #behaviour model
        q_v = self.behave_model(states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)

        #Get the skipper action
        # obs_skip_a = np.array(states, copy=False)
        obs_skip_a = states.data.cpu().numpy()
        obs_shape = obs_skip_a.shape
        if len(obs_shape) == 4: # for atari
            size = obs_shape[3]
            N = obs_shape[0]
            a = torch.tensor(actions, dtype=torch.float32).view(N, 1, 1, 1).to(self.device)
            obs_v_sk = torch.cat((states, a.expand(N, 4, 1, size)), dim=2)  
            q_vals_v = self.skip_model(obs_v_sk)
            q = q_vals_v.data.cpu().numpy()
            skips = self.action_selector(q)
            action = np.array([skips, actions]).T
        else: #for classic control
            obs_v_sk_np = np.concatenate((obs_skip_a,np.expand_dims(actions,axis=1)),axis=1)
            obs_v_sk = torch.FloatTensor(obs_v_sk_np).to(self.device)
            q_vals_v = self.skip_model(obs_v_sk)
            q = q_vals_v.data.cpu().numpy()
            skips = self.action_selector(q)
            action = np.array([skips, actions]).T

        return action, agent_states

class ExpanderAgent(ptan.agent.BaseAgent):
    """
    Expander Agent is a memoryless DQN agent which calculates Q values and skip steps
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, behave_model, action_selector, device="cpu", preprocessor=default_states_preprocessor):
        self.behave_model = behave_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device
    
    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        agent_state = None
        if self.behave_model.isRecurrent:
            agent_state = {'hidden':torch.zeros((1,1,self.net_behave.nn.hidden_size)).to('cuda'), 'cell':torch.zeros((1,1,self.net_behave.nn.hidden_size)).to('cuda')}
        return agent_state


    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        #behaviour model
        q_v = self.behave_model(states)
        q = q_v.data.cpu().numpy()
        action = self.action_selector(q)

        return action, agent_states


class DQN_CNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN_CNN, self).__init__()
        self.isRecurrent = False
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class DQN_MLP(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size=128):
        super(DQN_MLP, self).__init__()
        
        self.isRecurrent = False
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)
    
class DQN_MLP_L3(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size=128):
        super(DQN_MLP_L3, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)
    

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features,
                 sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * \
                   self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + \
            self.weight
        
        return nn.functional.linear(input, v, bias)
    
class NoisyCNN_DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyCNN_DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.noisy_layers = [
            NoisyLinear(conv_out_size, 512),
            NoisyLinear(512, n_actions)
        ]
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]

class NoisyMLP_DQN(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size=128):
        super(NoisyMLP_DQN, self).__init__()

        self.enc = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, n_actions)
        )

        enc_out_size = self._get_enc_out(obs_size)
        self.noisy_layers = [
            NoisyLinear(enc_out_size, hidden_size),
            NoisyLinear(hidden_size, n_actions)
        ]
        self.dec = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def _get_enc_out(self, shape):
        o = self.enc(torch.zeros(1, shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        enc_out = self.enc(x).view(x.size()[0], -1)
        return self.dec(enc_out)

    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]
