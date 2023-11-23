import os
import re
from typing import List, Dict

import numpy as np
import pandas as pd
import json

from qlearning_custom import QLearningCustom

DEFAULT_GAMMAS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

EP_LENGTH = []


def q_learning_stats_gammas_problem_sizes(
        state_sizes,
        problem_function,
        epsilon,
        epsilon_decay,
        epsilon_min,
        alpha,
        alpha_min,
        alpha_decay,
        episode_length,
        td_error_threshold,
        overall_stat_freq,
        iter_callback,
        n_iter,
        gammas=[0.99]):
    dfs = []
    for state_size in state_sizes:
        print(f"Processing size {state_size}")
        p, r = problem_function(state_size)
        cur_stat_df, _, policies = q_learning_stats_gammas(p, r,
                                                           epsilon,
                                                           epsilon_decay,
                                                           epsilon_min,
                                                           alpha,
                                                           alpha_min,
                                                           alpha_decay,
                                                           episode_length,
                                                           td_error_threshold,
                                                           overall_stat_freq,
                                                           iter_callback,
                                                           n_iter=n_iter,
                                                           gammas=gammas,
                                                           )
        cur_stat_df['States'] = state_size
        dfs.append(cur_stat_df)
    return pd.concat(dfs, ignore_index=True).reset_index(drop=True)


def q_learning_stats_gammas(p,
                            r,
                            epsilon,
                            epsilon_decay,
                            epsilon_min,
                            alpha,
                            alpha_min,
                            alpha_decay,
                            episode_length,
                            td_error_threshold,
                            overall_stat_freq,
                            iter_callback,
                            n_iter: int = 1e7,
                            gammas=None,
                            ):
    if gammas is None:
        gammas = DEFAULT_GAMMAS
    gs = []
    iters = []
    mean_vs = []
    diffs = []
    times = []
    total_times = []
    total_iters = []

    gammas_v2 = []
    errors_v2 = []
    errors_v2_max = []
    iters_v2 = []
    policies = {}
    for gamma in gammas:
        print(f"Running for gamma={gamma}")
        alg = QLearningCustom(p,
                              r,
                              gamma=gamma,
                              epsilon=epsilon,
                              epsilon_decay=epsilon_decay,
                              epsilon_min=epsilon_min,
                              alpha=alpha,
                              alpha_decay=alpha_decay,
                              alpha_min=alpha_min,
                              iter_callback=iter_callback,
                              n_iter=n_iter,
                              episode_length=episode_length,
                              td_error_threshold=td_error_threshold,
                              overall_stat_freq=overall_stat_freq,
                              )
        alg.run()
        stats = alg.run_stats
        iters.extend(s['Iteration'] for s in stats)
        mean_vs.extend(s['Mean V'] for s in stats)
        diffs.extend(s['Error'] for s in stats)
        times.extend(s['Time'] for s in stats)
        total_times.extend([alg.time] * len(stats))
        total_iters.extend([np.max([s['Iteration'] for s in stats])] * len(stats))
        gs.extend(s['Gamma'] for s in stats)

        iters_v2.extend(alg.stat_iters)
        errors_v2.extend(alg.stat_error_mean)
        errors_v2_max.extend(alg.stat_error_max)
        gammas_v2.extend([gamma] * len(alg.stat_iters))
        policies[gamma] = alg.policy

    return pd.DataFrame(
        {
            'Gamma': gs,
            'Iteration': iters,
            'Mean V': mean_vs,
            'Error': diffs,
            'Times': times,
            'Total Time': total_times,
            'Total Iter': total_iters,

        }
    ), pd.DataFrame(
        {'Gamma': gammas_v2, 'Error': errors_v2, 'Error Max': errors_v2_max, 'Iteration': iters_v2}), policies


def q_learning_stats_epsilons(p,
                              r,
                              epsilons,
                              alpha,
                              alpha_min,
                              alpha_decay,
                              episode_length,
                              td_error_threshold,
                              overall_stat_freq,
                              gamma,
                              iter_callback,
                              n_iter: int = 1e7,
                              ):
    eps = []
    iters = []
    mean_vs = []
    diffs = []
    times = []
    total_times = []
    total_iters = []

    eps_v2 = []
    errors_v2 = []
    errors_v2_max = []
    iters_v2 = []
    policies = {}
    for epsilon in epsilons:
        print(f"Running for epsilon={epsilon}")
        alg = QLearningCustom(p,
                              r,
                              gamma=gamma,
                              epsilon=epsilon,
                              epsilon_decay=1,
                              epsilon_min=epsilon,
                              alpha=alpha,
                              alpha_decay=alpha_decay,
                              alpha_min=alpha_min,
                              iter_callback=iter_callback,
                              n_iter=n_iter,
                              episode_length=episode_length,
                              td_error_threshold=td_error_threshold,
                              overall_stat_freq=overall_stat_freq,
                              )
        alg.run()
        stats = alg.run_stats
        iters.extend(s['Iteration'] for s in stats)
        mean_vs.extend(s['Mean V'] for s in stats)
        diffs.extend(s['Error'] for s in stats)
        times.extend(s['Time'] for s in stats)
        total_times.extend([alg.time] * len(stats))
        total_iters.extend([np.max([s['Iteration'] for s in stats])] * len(stats))
        eps.extend([epsilon] * len(stats))

        iters_v2.extend(alg.stat_iters)
        errors_v2.extend(alg.stat_error_mean)
        errors_v2_max.extend(alg.stat_error_max)
        eps_v2.extend([epsilon] * len(alg.stat_iters))
        policies[gamma] = alg.policy

    return pd.DataFrame(
        {
            'Epsilon': eps,
            'Iteration': iters,
            'Mean V': mean_vs,
            'Error': diffs,
            'Times': times,
            'Total Time': total_times,
            'Total Iter': total_iters,

        }
    ), pd.DataFrame(
        {'Epsilon': eps_v2, 'Error': errors_v2, 'Error Max': errors_v2_max, 'Iteration': iters_v2}), policies
