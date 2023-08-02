"""
Helper functions for creating a CLI
"""
from argparse import ArgumentParser
import torch



def add_tianshou_args(parser:ArgumentParser):
    group = parser.add_argument_group('Tianshou')
    group.add_argument('--max_epoch', type=int, default=10, help="Maximum epochs to run training for (training may stop earlier).")
    group.add_argument('--batch_size', type=int, default=64, help="Batch of the sample data that is going to feed into policy network. In Reinforce/PG case, batch-size=1 is equivalent to per step update, while batch-size > steps-per-collect is equivalent to single update per collect.")
    group.add_argument('--train_num', type=int, default=16, help="Number of parallel training environments.")
    group.add_argument('--test_num', type=int, default=100, help="Number of parallel testing environments.")
    

def add_device_args(parser:ArgumentParser):
    group = parser.add_argument_group('Device')
    group.add_argument('--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')

def add_wandb_args(parser:ArgumentParser):
    """
    Takes a argumgent parser object, and adds a group for the wandb arguments.
    """
    # wandb special
    group = parser.add_argument_group('Wandb')
    group.add_argument('--project_name', '-p', type=str, default='amrl', help="Project name, used for loging to WandB.")
    group.add_argument('--wandb_mode', '-w', choices=['online', 'disabled', 'offline'], default='online', help="WandB mode. Online means log data. Offline means store data locally and push it later. Disabled means don't log data at all")

def add_amrl_args(parser:ArgumentParser):
    group = parser.add_argument_group('AMRL')
    group.add_argument('--full_ext_reward', default=1, type=int, choices=[0,1], help="Fully observable rewards.")
    group.add_argument('--vanilla', default=0, type=int, choices=[0,1], help="Use non-wrapped environment.")
    group.add_argument('--obs_flag', default=0, type=int, choices=[0,1], help="Include a flag in the observation space to indicate if last action was observe?")
    group.add_argument('--prev_action_flag', default=0, type=int, choices=[0,1], help="Include a the previous action in the observation space?")
    group.add_argument('--obs_cost', type=float, default=0.2)
    group.add_argument('--max_episode_steps', type=int, default=400)
    parser.add_argument('--log_eval_episodes', default=50, type=int, help='How many episodes to run for evaluating the policy at the end of training.')
    