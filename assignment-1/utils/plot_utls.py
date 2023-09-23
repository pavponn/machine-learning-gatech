from typing import List, Union, Optional

import os
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path


def setup_plots():
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (10, 6)


def label_distribution_chart(
        df: pd.DataFrame,
        label_column: Union[str, List[str]],
        title: str = 'Label Distribution',
        save_path: Optional[str] = None):
    label_counts = df[label_column].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(label_counts))
    bars_counts = ax.bar(x, label_counts, width=0.4, align='center', label='Counts')

    # Set x-axis labels and tick positions
    ax.set_xticks([i for i in x])
    ax.set_xticklabels(label_counts.index)

    # Set labels and title
    ax.set_xlabel('Labels')
    ax.set_ylabel('Count')
    ax.set_title(title)

    for bar_count in bars_counts:
        height = bar_count.get_height()
        percentage_label = (height / len(df)) * 100
        ax.text(bar_count.get_x() + bar_count.get_width() / 2., height, "{:.2f}%".format(percentage_label), ha='center',
                va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path is not None:
        path_to_create = Path(save_path).parent.absolute()
        os.makedirs(path_to_create, exist_ok=True)
        plt.savefig(save_path)
    plt.show()
