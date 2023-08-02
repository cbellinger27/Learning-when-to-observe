# %% Retrieves the data from the episodes artifact,
# and creates plots for each combination of hyperparameters, averaging across obs-costs and seeds.
# Plots are logged to wandb.
import pandas as pd
import argparse
import altair as alt
import wandb
import itertools
from altair_saver import save
import os

alt.data_transformers.disable_max_rows()


def _create_parser() -> argparse.ArgumentParser:    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--entity', type=str, default='cbellinger', help='Entity inside wandb.')
    parser.add_argument('--project', type=str, help='Project on wandb.')
    parser.add_argument('--sweep_id', type=str, help='Sweep id.')
    parser.add_argument('--run_id', type=str, default=None, help='Run id. (optional)')
    parser.add_argument('--matches', type=int, default=None, help='Plot Pong matches. (optional)')
    parser.add_argument('--agent_style', type=str, default='expander', help='expander, cost_expander or skipper agent style.')
    parser.add_argument('--csv_name', type=str, default='lastTrained_episodes_expander', help='name of csv file.')
    
    return parser


def plot_experiment(args, experiment_id, experiment_caption, df):
    plot_actions(args, experiment_id, experiment_caption, df)
    plot_rewards(args, experiment_id, experiment_caption, df)


def plot_actions(args, experiment_id, experiment_caption, df):
    if args.matches is not None:
        df_a = df[df['episode']==1]
        seed = df_a.iloc[0]['seed']
        df_a = df_a[df_a['seed']==seed]
        df_a['matches'] = 0
        match = 0
    elif args.run_id is None:
        df_a = df[df['episode']==10]
    else:
        df_a = df[df['episode']<10]

    max_steps = df_a['step'].max()
    ticks = int(max_steps/10) # how often to put ticks
    max_steps = 10_000 # make sure this is way more than needed
    
    for obs_cost, df_o in df_a.groupby(['obs_cost']):
        print(f'Length of {obs_cost} df! {df_o.shape[0]}')
        
        if args.matches is not None:
            print("plot matches")
            if args.agent_style == 'skipper':
                df_o = df_o.loc[df_o.index.repeat(df_o.skips+1)].reset_index(drop=True)
                cur_step = df_o.loc[0]['step']
                counter = 0 
                for i in range(df_o.shape[0]-1):
                    df_o.loc[i,'matches'] = match
                    if df_o.loc[i]['ext_reward']  >= 1 or df_o.loc[i]['ext_reward']  <= -1: #end of one match of pong
                        match += 1
                        counter = 0 
                    if cur_step == df_o.loc[i+1]['step']:
                        df_o.loc[i,'measure'] = False
                    else:
                        df_o.loc[i,'measure'] = True
                        cur_step = df_o.loc[i+1]['step']
                    df_o.loc[i,'step'] = counter
                    counter += 1
                df_o.loc[i+1,'measure'] = False
                df_o.loc[i+1,'step'] = counter
                df_o = df_o[df_o['matches']>0]
                print(df_o)
            elif args.agent_style == 'expander':
                cur_step = df_o.iloc[0]['step']
                counter = 0 
                for i in range(df_o.shape[0]-1):
                    df_o.iloc[i]['matches'] = match
                    if df_o.iloc[i]['ext_reward']  == 1 or df_o.iloc[i]['ext_reward']  == -1: #end of one match of pong
                        match += 1
                        counter = 0 
                    df_o.iloc[i]['step'] = counter
                    counter += 1
                df_o.iloc[i+1]['measure'] = False
                df_o.iloc[i+1]['step'] = counter
                df_o = df_o[df_o['matches']>0]
            print(df_o)
        elif args.run_id is None:
            if args.agent_style == 'skipper':
                df_o = df_o.loc[df_o.index.repeat(df_o.skips+1)].reset_index(drop=True)
                cur_step = df_o.loc[0]['step']
                cur_seed = df_o.loc[0]['seed']
                counter = 0 
                for i in range(df_o.shape[0]-1):
                    if cur_seed != df_o.loc[i]['seed']:
                        cur_seed = df_o.loc[i]['seed']
                        counter = 0 
                    if cur_step == df_o.loc[i+1]['step']:
                        df_o.loc[i,'measure'] = False
                    else:
                        df_o.loc[i,'measure'] = True
                        cur_step = df_o.loc[i+1]['step']
                    df_o.loc[i,'step'] = counter
                    counter += 1
                df_o.loc[i+1,'measure'] = False
                df_o.loc[i+1,'step'] = counter
        
        if args.matches is not None:
            chart = alt.Chart(df_o).mark_rect(
                ).encode(
                    alt.X('step:O', title='Step', axis=alt.Axis(values=list(range(0, max_steps, ticks)))),
                    alt.Y('matches:O', title='Match',  axis=alt.Axis(labels=True)),
                    alt.Color('measure:N', legend=alt.Legend(title='Measure'))
                ).properties(width=600).configure_axis(labelFontSize=18,titleFontSize=22)
        elif args.run_id is None:
            chart = alt.Chart(df_o).mark_rect(
                ).encode(
                    alt.X('step:O', title='Step', axis=alt.Axis(values=list(range(0, max_steps, ticks)))),
                    alt.Y('seed:O', title='seed',  axis=alt.Axis(labels=True)),
                    alt.Color('measure:N', legend=alt.Legend(title='Measure'))
                ).properties(width=600).configure_axis(labelFontSize=18,titleFontSize=22)
        else:
            if args.agent_style == 'skipper':
                df_o = df_o.loc[df_o.index.repeat(df_o.skips+1)].reset_index(drop=True)
                cur_step = df_o.loc[0]['step']
                cur_seed = df_o.loc[0]['episode']
                counter = 0 
                i = 0
                while i < df_o.shape[0]-1:
                    idx = df_o.index
                    if cur_seed != df_o.loc[idx[i]]['episode']:
                        cur_seed = df_o.loc[idx[i]]['episode']
                        counter = 0 
                    if cur_step == df_o.loc[idx[i+1]]['step']:
                        df_o.loc[i,'measure'] = False
                    else:
                        df_o.loc[i,'measure'] = True
                        cur_step = df_o.loc[idx[i+1]]['step']
                    df_o.loc[idx[i],'step'] = counter
                    counter += 1
                    i += 1
                df_o.loc[idx[i],'measure'] = False
                df_o.loc[idx[i],'step'] = counter
            #
            chart = alt.Chart(df_o).mark_rect(
                ).encode(
                    alt.X('step:O', title='Step', axis=alt.Axis(values=list(range(0, max_steps, ticks)))),
                    alt.Y('episode:O', title='Episode',  axis=alt.Axis(labels=True)),
                    alt.Color('measure:N', legend=alt.Legend(title='Measure'))
                ).properties(width=600).configure_axis(labelFontSize=18,titleFontSize=22)

        
        log_obs_experiment_chart("actions", args, experiment_id, experiment_caption, chart, obs_cost)
        

def plot_rewards(args, experiment_id, experiment_caption, df):
    # Rewards plot
    df['costed_reward'] = df['reward']
    df_r = df[['obs_cost', 'seed', 'episode', 'reward', 'int_reward', 'ext_reward', 'costed_reward']]

    if args.run_id is None:
        df_r = df_r.groupby(['obs_cost', 'seed', 'episode']).sum()
        df_r = df_r.groupby(['obs_cost', 'seed']).mean().reset_index()
        for obs_cost, df_o in df_r.groupby('obs_cost'):
            chart = alt.Chart(df_o).mark_rect().encode(
                alt.X('reward:Q', title='Costed Reward'),
                alt.Y('seed:O', title='Seed', axis=alt.Axis(labels=True))
            ).properties(
                width=600).configure_axis(labelFontSize=18,titleFontSize=22)
    else:    
        df_r = df_r.groupby(['obs_cost', 'episode']).sum().reset_index()
        df_r = df_r[df_r['episode']<10]
        for obs_cost, df_o in df_r.groupby('obs_cost'):
            chart = alt.Chart(df_o).mark_rect().encode(
                alt.X('costed_reward:Q', title='Costed Reward'),
                alt.Y('episode:O', title='Episde', axis=alt.Axis(labels=True))
            ).properties(
                width=600).configure_axis(labelFontSize=18,titleFontSize=22)
        
    log_obs_experiment_chart("rewards", args, experiment_id, experiment_caption, chart, obs_cost)
        

def log_obs_experiment_chart(plot_name, args, experiment_id, experiment_caption, chart, obs_cost):
    plot_path = f'plots/{args.project}/{args.sweep_id}/{plot_name}_{experiment_id}_{obs_cost}.png'
    
    chart.save(plot_path)
    # Only caption the image with the hyperparameters that vary
    chart_caption = f'{experiment_caption}_obs-cost:{obs_cost}'
    wandb.log({f"{plot_name}/obs-cost:{obs_cost} experiment:{experiment_id}": wandb.Image(plot_path, caption=chart_caption)})

def experiments(hyperparams):
    # Given a dict of  `{'col', [vals], 'col2': [vals]}`
    # returns all possible combinations
    keys, values = zip(*hyperparams.items())
    return [list(zip(keys, v)) for v in itertools.product(*values)]

def main(args):
    df = load_episodes_df(entity=args.entity, project=args.project, sweep_id=args.sweep_id, csv_name=args.csv_name)
    h = hyperparams(df)
    exps = experiments(h)
    varying_params = [k for (k,v) in h.items() if len(v) > 1]
    plot_experiments(args, exps, df, varying_params)

def plot_experiments(args, exps, df, varying_params):
    os.makedirs(f'plots/{args.project}/{args.sweep_id}', exist_ok=True)

    with wandb.init(entity=args.entity, project=args.project) as run:
        for i, exp in enumerate(exps):
            df_exp = select_experiment(exp, df)

            exp_params = dict(exp)
            # wandb.config.update(exp_params)

            experiment_caption = exp_caption(exp_params, varying_params)
            plot_experiment(args, i, experiment_caption, df_exp)


def exp_caption(exp_params, varying_params):
    """Only caption the image with the hyperparameters that vary"""
    caption_dict = {k:v for (k,v) in exp_params.items() if k in varying_params}
    return str(caption_dict)            

def select_experiment(exp, df):
    df_exp = df
    for param, value in exp:
        df_exp = df_exp[df_exp[param]==value]
    return df_exp
    
def hyperparams(df):
    exclude = ['action', 'id', 'action_shape', 'measure', 'cost', 'reward', 'episode', 'obs_cost', 'step', 'seed', 'int_reward','ext_reward', 'skips', 'action_pair', 'detailed_name']
    h = {}
    for col in df:
        if col in exclude:
            continue
        vals = df[col].unique().tolist()
        h[col] = vals
    return h

def load_episodes_df(*, entity, project, sweep_id, csv_name='lastTrained_episodes_expander'):
    api = wandb.Api()
    artifact = api.artifact(f'{entity}/{project}/episodes_{sweep_id}:latest')
    artifact_dir = artifact.download()
    csv_path = f'{artifact_dir}/{csv_name}.csv.zip'
    df = pd.read_csv(csv_path)
    df = df.dropna(axis=1)
    return df



# %%
if __name__ == '__main__':
    args = _create_parser().parse_args()
    # args.agent_style = 'skipper'
    # args.csv_name = 'lastTrained_episodes_skipper'
    print(args.csv_name)
    main(args)

