# %% Retrieves the data from the sweep and makes plots. Currently tested with PPO experiments only.
import pandas as pd
import argparse
from pandas.api import types
import altair as alt
import os

alt.data_transformers.disable_max_rows()


def create_parser() -> argparse.ArgumentParser:    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--entity', type=str, default='rl-team', help='Entity inside wandb.')
    parser.add_argument('--project', type=str, help='Project on wandb.')
    parser.add_argument('--sweep-id', type=str, help='Sweep id.')
    parser.add_argument('--run-id', type=str, help='Run id.')
    parser.add_argument('--num-base-actions', type=int, help='Number of actions in the base environment. E.g. 2 for CartPole.')
    
    return parser

def plot_episodes(args):   
    sweep_id = args.sweep_id 
    df = get_episodes_df(args.entity, args.project, args.sweep_id, args.run_id)
    
    assert types.is_float_dtype(df['int_reward'])
    assert types.is_int64_dtype(df['step'])
    assert types.is_int64_dtype(df['action'])
    assert types.is_float_dtype(df['reward'])
    assert types.is_int64_dtype(df['episode'])
    
    df_a = df[df['episode']==0]
    # Actions plots
    chart_actions = alt.Chart(df_a).mark_rect(
    ).encode(
        alt.X('step:O', title='Step'),
        alt.Y('seed:O', title='',  axis=alt.Axis(labels=False)),
        alt.Color('measure:N', legend=alt.Legend(title='Measure')),
        row=alt.Facet('obs_cost:Q', title='Observation Cost')
    ).properties(
        width=600)
    
    chart_actions.save(f'sweeps/{sweep_id}/plots/{args.run_id}/chart_actions.png')
    
    
    # Rewards plot
    df_r = df[['obs_cost', 'reward', 'int_reward',  'ext_reward', 'episode']]
    df_r = df_r.groupby(['obs_cost', 'episode']).sum()
    df_r['costed_reward'] = df_r['reward']
    df_r = df_r.groupby(['obs_cost']).mean().reset_index()
    
    chart_rewards = alt.Chart(df_r).mark_rect().encode(
        alt.X('reward:Q', title='Reward'),
        alt.Y('seed:O', title='', axis=alt.Axis(labels=False)),
        alt.Color('reward:Q', legend=alt.Legend(title='Reward'), scale=alt.Scale(scheme="redyellowblue")),
        row=alt.Facet('obs_cost:Q', title='Observation Cost')
    ).properties(
        width=600)
    
    chart_rewards.save(f'sweeps/{sweep_id}/plots/{args.run_id}/chart_rewards.png')
    
    chart_costed_rewards = alt.Chart(df_r).mark_rect().encode(
        alt.X('costed_reward:Q', title='Costed Reward'),
        alt.Y('seed:O', title='', axis=alt.Axis(labels=False)),
        alt.Color('reward:Q', legend=alt.Legend(title='Costed Reward'), scale=alt.Scale(scheme="redyellowblue")),
        row=alt.Facet('obs_cost:Q', title='Observation Cost')
    ).properties(
        width=600)
    
    chart_costed_rewards.save(f'sweeps/{sweep_id}/plots/{args.run_id}/chart_costed_rewards.png')
    return chart_actions & chart_rewards & chart_costed_rewards


def get_episodes_df(entity, project, sweep_id, run_id):
    """Downloads all the episode files and concatenates them into single dataframe."""
    df = []
    
    if os.path.exists(f'sweeps/{sweep_id}/plots/{run_id}/lastTrained_episodes_expander.csv'):
        df = pd.read_csv(f"sweeps/{sweep_id}/plots/{run_id}/lastTrained_episodes_expander.csv")
    else:
        print('file does not exist')
    return df
    
# %%
if __name__ == '__main__':
    args = create_parser().parse_args()
    plot_episodes(args)
# %%

#python plots.py --sweep ws179cdc --entity rl-team --project DualDRQN_CartPole --num-base-actions 2 --run-id 3y0nd2t5
#
# rl-team/DualDRQN_CartPole/3y0nd2t5
# rl-team/DualDRQN_CartPole/g0l0qnrf

# args = create_parser().parse_args()
# args.sweep_id = "1bpt44v5"
# args.entity = "rl-team"
# args.project = "DualDRQN_CartPole"
# args.num_base_actions = 2
# args.run_id = "g0l0qnrf"
# sweep_id = args.sweep_id 

# plot_episodes(args)


# rl-team/DualDQN_CartPole/yt9ys2te
# args = create_parser().parse_args()
# args.sweep_id = "ws179cdc"
# args.entity = "rl-team"
# args.project = "DualDQN_CartPolee"
# args.num_base_actions = 2
# args.run_id = "yt9ys2te"

# sweep_id = args.sweep_id 

# plot_episodes(args)

# df = get_episodes_df(args.entity, args.project, args.sweep_id, args.run_id)


# df_r = df[['obs_cost', 'cost', 'seed', 'episode', 'reward']]
# df_r = df_r.groupby(['obs_cost', 'seed', 'episode']).sum()
# df_r['costed_reward'] = df_r['reward']-df_r['cost']
# df_r = df_r.groupby(['obs_cost', 'seed']).mean().reset_index()

# df_m = df[['obs_cost', 'cost', 'seed', 'episode', 'reward','action']]
# df_m['action'] = df_m['action'] > (args.num_base_actions - 1)
# df_m.groupby(['obs_cost', 'seed', 'episode'])['action'].value_counts()

# df_m.groupby(['obs_cost', 'seed', 'episode'])['action'].sum().mean()
# df_m.groupby(['obs_cost', 'seed', 'episode'])['action'].sum().std()

# DQN
# mean=100
# std=0

# DRQN
# mean=92.8
# std=4.22