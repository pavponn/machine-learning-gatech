import os
import re
from typing import List, Dict

import numpy as np
import pandas as pd
import json

import hiive.mdptoolbox
import hiive.mdptoolbox.mdp
import hiive.mdptoolbox.example

DEFAULT_GAMMAS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]


def write_stats_for_problem_sizes(algo_class, main_size, state_sizes, problem_function, not_use_span_vi=False,
                                  collect_changes=False, policies0=None, max_iter=1000, gammas=None):
    dfs = []
    policies_to_return = None
    for state_size in state_sizes:
        p, r = problem_function(state_size)
        cur_stat_df, policies = write_stats(p, r, algo_class, not_use_span_vi=not_use_span_vi,
                                            collect_changes=collect_changes, max_iter=max_iter, gammas=gammas,
                                            policy0=None if policies0 is None else policies0[state_size])
        if state_size == main_size:
            policies_to_return = policies
        cur_stat_df['States'] = state_size
        dfs.append(cur_stat_df)
    return pd.concat(dfs, ignore_index=True).reset_index(drop=True), policies_to_return


def write_stats(p, r, algo_class, gammas=None, max_iter: int = 1000, not_use_span_vi: bool = False,
                collect_changes=False, policy0=None):
    if gammas is None:
        gammas = DEFAULT_GAMMAS
    gs = []
    iters = []
    mean_vs = []
    diffs = []
    times = []
    total_times = []
    total_iters = []
    changes = []
    policies = {}
    for gamma in gammas:
        if policy0 is not None:
            alg = algo_class(p, r, gamma, max_iter=max_iter, skip_check=True, policy0=policy0)
        else:
            alg = algo_class(p, r, gamma, max_iter=max_iter, skip_check=True)
        if not_use_span_vi:
            alg.max_iter = max_iter
        alg.run()
        stats = alg.run_stats
        iters.extend(s['Iteration'] for s in stats)
        mean_vs.extend(s['Mean V'] for s in stats)
        diffs.extend(s['Error'] for s in stats)
        times.extend(s['Time'] for s in stats)
        total_times.extend([alg.time] * len(stats))
        total_iters.extend([alg.iter] * len(stats))
        if collect_changes:
            changes.extend(s['Changes'] for s in stats)
        else:
            changes.extend([0] * len(stats))
        gs.extend([gamma] * len(stats))
        policies[gamma] = alg.policy

    return pd.DataFrame(
        {
            'Gamma': gs,
            'Iteration': iters,
            'Mean V': mean_vs,
            'Difference': diffs,
            'Times': times,
            'Total Time': total_times,
            'Total Iter': total_iters,
            'Changes': changes,
        }
    ), policies


def save_df_as_csv(df: pd.DataFrame, folder_path: str, file_name: str):
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, file_name)
    df.to_csv(save_path, index=False)


def read_csv(folder_path: str, file_name: str):
    csv_path = os.path.join(folder_path, file_name)
    return pd.read_csv(csv_path)


def save_policies(policies: Dict[float, List[int]], folder_path: str, file_name: str):
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, file_name)
    with open(save_path, 'w') as jsonfile:
        json.dump(policies, jsonfile)


def read_policies(folder_path: str, file_name: str):
    policy_path = os.path.join(folder_path, file_name)
    with open(policy_path, 'r') as jsonfile:
        loaded_policies = json.load(jsonfile)
    return loaded_policies


# TODO
def convert_p_r(env):
    rows = env.nrow
    cols = env.ncol
    r = np.zeros((4, rows * cols, rows * cols))
    p = np.zeros((4, rows * cols, rows * cols))
    env_P = env.unwrapped.P
    old_state = np.inf
    for state in env_P:
        for action in env_P[state]:
            for idx in range((len(env_P[state][action]))):
                trans_prob = env_P[state][action][idx][0]
                next_state = env_P[state][action][idx][1]
                reward = env_P[state][action][idx][2]
                if next_state == old_state:
                    p[action][state][next_state] = p[action][state][old_state] + trans_prob
                    r[action][state][next_state] = r[action][state][old_state] + reward
                else:
                    p[action][state][next_state] = trans_prob
                    r[action][state][next_state] = reward
                old_state = next_state
            p[action, state, :] /= np.sum(p[action, state, :])
    return p, r


def convert_p_r_v2(env):
    transitions = env.P
    actions = int(re.findall(r'\d+', str(env.action_space))[0])
    states = int(re.findall(r'\d+', str(env.observation_space))[0])
    P = np.zeros((actions, states, states))
    R = np.zeros((states, actions))
    for state in range(states):
        for action in range(actions):
            for i in range(len(transitions[state][action])):
                tran_prob = transitions[state][action][i][0]
                state_ = transitions[state][action][i][1]
                R[state][action] += tran_prob * transitions[state][action][i][2]
                P[action, state, state_] += tran_prob
    return P, R
