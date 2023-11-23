import os
import re
from typing import List, Dict

import numpy as np
import pandas as pd
import json

from matplotlib import pyplot as plt

from qlearning_custom import QLearningCustom


def create_stat_plot(df,
                     folder_path: str,
                     file_name: str,
                     hue_col='Gamma',
                     x_axis='Iteration',
                     y_axis='Error',
                     log_scale_x=False,
                     log_scale_y=False,
                     marker='.',
                     marker_size=2,
                     title_additional='',

                     ):
    os.makedirs(folder_path, exist_ok=True)

    for hue_value, group in df.groupby(hue_col):
        plt.plot(group[x_axis], group[y_axis], marker=marker, markersize=marker_size,
                 label=f'{hue_col} {hue_value}')
    plt.title(f'{y_axis} vs. {x_axis} {title_additional}')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    if log_scale_x:
        plt.xscale('log')
    if log_scale_y:
        plt.yscale('log')
    plt.grid(True)
    plt.legend(frameon=True)

    plot_path = os.path.join(folder_path, file_name)
    plt.savefig(plot_path)
    plt.close()
