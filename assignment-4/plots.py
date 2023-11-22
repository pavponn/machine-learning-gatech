import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib


def setup_plots():
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (8, 6)


def create_convergence_and_state_plots(default_size,
                                       result_df,
                                       folder_path,
                                       algo,
                                       log_scale_x_one=False,
                                       log_scale_y_one=False,
                                       log_scale_x=False,
                                       log_scale_y=False,
                                       marker='.',
                                       marker_size=10,
                                       plot_changes: bool = False):
    os.makedirs(folder_path, exist_ok=True)
    df_def_size = result_df[result_df['States'] == default_size]

    if plot_changes:
        for gamma_value, group in df_def_size.groupby('Gamma'):
            plt.plot(group['Iteration'], group['Changes'], marker=marker, markersize=marker_size,
                     label=f'Gamma {gamma_value}')
        plt.title('Changes vs. Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Policy Changes')
        if log_scale_x_one:
            plt.xscale('log')
        if log_scale_y_one:
            plt.yscale('log')
        plt.grid(True)
        plt.legend(frameon=True)

        mean_v_plot_path = os.path.join(folder_path, f'{algo}_policy_changes.png')
        plt.savefig(mean_v_plot_path)
        plt.close()

    ###########

    for gamma_value, group in df_def_size.groupby('Gamma'):
        plt.plot(group['Iteration'], group['Mean V'], marker=marker, markersize=marker_size,
                 label=f'Gamma {gamma_value}')
    plt.title('Mean V vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Mean V')
    if log_scale_x_one:
        plt.xscale('log')
    if log_scale_y_one:
        plt.yscale('log')
    plt.grid(True)
    plt.legend(frameon=True)

    mean_v_plot_path = os.path.join(folder_path, f'{algo}_convergence_mean_v.png')
    plt.savefig(mean_v_plot_path)
    plt.close()

    ###########

    for gamma_value, group in df_def_size.groupby('Gamma'):
        plt.plot(group['Iteration'], group['Difference'], marker=marker, markersize=marker_size,
                 label=f'Gamma {gamma_value}',
                 )

    plt.title('Difference vs. Iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Difference')
    if log_scale_x_one:
        plt.xscale('log')
    if log_scale_y_one:
        plt.yscale('log')
    plt.grid(True)
    plt.legend(frameon=True)

    difference_plot_path = os.path.join(folder_path, f'{algo}_convergence_difference.png')
    plt.savefig(difference_plot_path)
    plt.close()

    ####################
    ####################
    ####################
    ####################

    data = result_df.groupby(['States', 'Gamma'])['Total Iter'].max().reset_index()
    for i, gamma in enumerate(set(result_df['Gamma'])):
        gamma_data = data[data['Gamma'] == gamma]
        states = gamma_data['States']
        iters = gamma_data['Total Iter']
        plt.plot(states, iters, marker=marker, markersize=marker_size, linestyle='-', label=f"Gamma = {gamma}")
    plt.title('Iterations vs. States')
    plt.xlabel('States')
    plt.ylabel('Iterations')
    plt.grid(True)
    plt.legend(frameon=True)
    if log_scale_x:
        plt.xscale('log')
    if log_scale_y:
        plt.yscale('log')
    states_iterations_plot_path = os.path.join(folder_path, f'{algo}_state_analysis_iters.png')
    plt.savefig(states_iterations_plot_path)

    plt.close()

    ################

    data = result_df.groupby(['States', 'Gamma'])['Total Time'].max().reset_index()
    for i, gamma in enumerate(set(result_df['Gamma'])):
        gamma_data = data[data['Gamma'] == gamma]
        states = gamma_data['States']
        times = gamma_data['Total Time']
        plt.plot(states, times, marker=marker, markersize=marker_size, linestyle='-', label=f"Gamma = {gamma}")

    plt.title('Time vs. States')
    plt.xlabel('States')
    plt.ylabel('Time')
    if log_scale_x:
        plt.xscale('log')
    if log_scale_y:
        plt.yscale('log')
    plt.grid(True)
    plt.legend(frameon=True)

    states_time_plot_path = os.path.join(folder_path, f'{algo}_state_analysis_time.png')
    plt.savefig(states_time_plot_path)
    plt.close()
