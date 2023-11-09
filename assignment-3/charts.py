import time
from typing import List, Union, Optional

import os

import numpy as np
import pandas as pd
from kneed import KneeLocator
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick import set_aesthetic
from GaussianMixtureCluster import GaussianMixtureCluster
from scipy.stats import kurtosis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import Isomap
from constants import SEEDS


def setup_plots():
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (8, 6)


def pair_plot(df: pd.DataFrame, output_path: str, hue=None, vars=None):
    path_to_create = Path(output_path).parent.absolute()
    os.makedirs(path_to_create, exist_ok=True)
    this_plot = sns.pairplot(df, hue=hue, vars=vars)
    this_plot.savefig(output_path)
    return this_plot


def k_means_visualize(X_data: pd.DataFrame,
                      output_path: str,
                      seed: int,
                      k_max: int = 20,
                      metric: str = 'distortion',
                      locate_elbow: bool = True,
                      timings: bool = False):
    set_aesthetic()
    setup_plots()
    model = KMeans(random_state=seed)
    visualizer = KElbowVisualizer(
        model, k=(2, k_max + 1), metric=metric, timings=timings, locate_elbow=locate_elbow
    )

    # Define a custom formatter to display integer ticks
    visualizer.ax.set_xticks(range(2, k_max + 1))
    visualizer.ax.set_xlim(left=0, right=k_max + 1)

    visualizer.fit(X_data)
    path_to_create = Path(output_path).parent.absolute()
    os.makedirs(path_to_create, exist_ok=True)
    visualizer.show(outpath=output_path)


def gmm_visualize(X_data: pd.DataFrame,
                  output_path: str,
                  seed: int,
                  k_max: int = 20,
                  metric: str = 'distortion',
                  n_init: int = 10,
                  locate_elbow: bool = True,
                  timings=False):
    # setup_plots()
    # set_aesthetic()
    model = GaussianMixtureCluster(random_state=seed, n_init=n_init)

    visualizer = KElbowVisualizer(
        model, k=(2, k_max + 1), metric=metric, timings=timings, locate_elbow=locate_elbow,
        force_model=True,
    )

    # Define a custom formatter to display integer ticks
    visualizer.ax.set_xticks(range(2, k_max + 1))
    visualizer.ax.set_xlim(left=0, right=k_max + 1)

    visualizer.fit(X_data)
    path_to_create = Path(output_path).parent.absolute()
    os.makedirs(path_to_create, exist_ok=True)
    visualizer.show(outpath=output_path)


def gmm_visualize_aic_bic(X_data: pd.DataFrame,
                          output_path: str,
                          seed: int,
                          k_max: int = 20,
                          n_init: int = 10,
                          ):
    set_aesthetic()
    setup_plots()

    fig, ax = plt.subplots()

    # Set Yellowbrick-like style
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', colors='black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')

    aic = []
    bic = []
    time_list = []
    k_range = list(range(2, k_max + 1))
    for k in k_range:
        start = time.time()
        gmm = GaussianMixture(n_components=k, random_state=seed, n_init=n_init)
        gmm.fit(X_data)
        end = time.time()
        time_list.append(end - start)
        aic.append(gmm.aic(X_data))
        bic.append(gmm.bic(X_data))

    plt.plot(k_range, aic, color='#1f77b4', label='AIC')
    plt.plot(k_range, bic, color='#2ca02c', label='BIC')

    plt.scatter(k_range, aic, color='#1f77b4', marker='D', s=50, zorder=5)
    plt.scatter(k_range, bic, color='#2ca02c', marker='D', s=50, zorder=5)
    plt.ylabel('AIC/BIC score')
    plt.xlabel('k')
    plt.title('AIC/BIC Score for GaussianMixture')
    plt.xticks(k_range)

    plt.tight_layout()
    plt.legend(frameon=True)
    plt.savefig(output_path)
    plt.show()


def visualize_clusters_with_tsne(model, X_data: pd.DataFrame, labels: Optional[pd.DataFrame], output_path: str,
                                 seed: int, labels_name='Ground Truth Labels', s=5):
    clusters = model.fit_predict(X_data)

    tsne = TSNE(n_components=2, random_state=seed)
    X_tsne = tsne.fit_transform(X_data)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='plasma', label='Predicted Clusters', alpha=1.)
    if labels is not None:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', label=labels_name, alpha=1.0,
                    marker='D', s=s, linewidths=1)

    plt.legend()
    plt.title('t-SNE Visualization of Clusters')
    plt.savefig(output_path)
    plt.show()


def pca_visualize_explained_variance_ratio(X_data: pd.DataFrame, output_path: str):
    set_aesthetic()
    setup_plots()
    pca = PCA(n_components=X_data.shape[1])
    pca.fit(X_data)

    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    x_axis = range(1, len(cumulative_variance_ratio) + 1)

    plt.plot(x_axis, cumulative_variance_ratio, color='#1f77b4', label='Explained Variance Ratio')
    plt.scatter(x_axis, cumulative_variance_ratio, color='#1f77b4', marker='D', s=50, zorder=5)
    plt.xticks(x_axis)
    thresholds_with_colour = [(0.8, 'g'), (0.85, 'b'), (0.9, 'purple'), (0.95, 'orange')]
    for threshold, colour in thresholds_with_colour:
        plt.axhline(y=threshold, linestyle='--', color=colour, label=f'{threshold} Explained Variance Ratio')

    plt.title('Explained Variance Ratio for PCA vs number of components')
    plt.xlabel('n_components')
    plt.ylabel('explained variance ratio')
    plt.grid(True)
    plt.legend(frameon=True)
    plt.savefig(output_path)
    plt.show()


def pca_visualize_eigenvalues(X_data: pd.DataFrame, output_path: str):
    set_aesthetic()
    setup_plots()
    pca = PCA(n_components=X_data.shape[1])
    pca.fit(X_data)
    eigen_values = pca.explained_variance_
    x_axis = range(1, len(eigen_values) + 1)
    print(f"Eigen values: {eigen_values}")

    plt.plot(x_axis, eigen_values, color='#1f77b4')
    plt.scatter(x_axis, eigen_values, color='#1f77b4', marker='D', s=50, zorder=5)
    plt.xticks(x_axis)
    plt.yticks(eigen_values, fontsize=6)

    plt.title('Eigenvalues for Principal Components')
    plt.xlabel('n-th principal component')
    plt.ylabel('eigenvalues (λ)')
    plt.savefig(output_path)
    plt.show()


def ica_visualize_absolute_mean_kurtosis(X_data: pd.DataFrame, output_path: str, seed: int, max_iter: int = 500):
    n_features = X_data.shape[1]
    k_range = range(1, n_features + 1)

    ica = FastICA(n_components=n_features, random_state=seed, max_iter=max_iter)
    X_data_reduced = ica.fit_transform(X_data)
    kurtosis_ica = np.absolute(kurtosis(X_data_reduced))
    kurtosis_ica = np.sort(kurtosis_ica)[::-1]
    cum_kurtosis_means = np.cumsum(kurtosis_ica) / np.arange(1, len(kurtosis_ica) + 1)

    plt.plot(k_range, cum_kurtosis_means, color='#1f77b4', label='Absolute Average Kurtosis')

    elbow_locator = KneeLocator(k_range, cum_kurtosis_means, curve='convex', direction='decreasing')
    plt.axvline(x=elbow_locator.knee, color='black', label=f'Elbow point', linestyle='--')

    plt.scatter(k_range, cum_kurtosis_means, color='#1f77b4', marker='D', s=50, zorder=5)
    plt.xticks(k_range)
    plt.xlabel('n_components')
    plt.ylabel('absolute average kurtosis')
    plt.title("Absolute Average Kurtosis vs Number of Independent Components")
    plt.grid(True)
    plt.legend(frameon=True)
    plt.savefig(output_path)

    plt.show()


def ica_visualize_absolute_kurtosis_distribution(X_data: pd.DataFrame, output_path: str, seed: int,
                                                 max_iter: int = 500):
    k = X_data.shape[1]
    ica = FastICA(n_components=k, random_state=seed, max_iter=max_iter)
    X_data_reduced = ica.fit_transform(X_data)
    print(len(kurtosis(X_data_reduced)))
    kurtosis_values = np.abs(kurtosis(X_data_reduced))
    kurtosis_values = np.sort(kurtosis_values)[::-1]
    x_axis = range(1, k + 1)
    plt.bar(x_axis, kurtosis_values, color='#1f77b4')
    for i, value in enumerate(kurtosis_values):
        plt.text(i + 1, value + 1, '{:.1f}'.format(value), ha='center', va='bottom')

    plt.xticks(x_axis)
    plt.title('Absolute Kurtosis Values for Independent Components')
    plt.xlabel('independent component')
    plt.ylabel('absolute kurtosis')
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()


def rp_reconstruction_error(X_data: pd.DataFrame, output_path: str):
    num_features = X_data.shape[1]
    k_range = range(1, num_features + 1)
    mean_errors = []
    std_errors = []
    for k in k_range:
        errors = []
        for seed in SEEDS:
            rand_proj = GaussianRandomProjection(n_components=k, compute_inverse_components=True, random_state=seed)
            X_data_reduced = rand_proj.fit_transform(X_data)
            X_data_recon = rand_proj.inverse_transform(X_data_reduced)
            errors.append(mean_squared_error(X_data, X_data_recon))

        mean_errors.append(np.mean(errors))
        std_errors.append(np.std(errors))

    mean_errors = np.array(mean_errors)
    std_errors = np.array(std_errors)

    plt.plot(k_range, mean_errors, color='#1f77b4')
    plt.fill_between(k_range, mean_errors - std_errors, mean_errors + std_errors,
                     color='#1f77b4', alpha=0.2)
    plt.scatter(k_range, mean_errors, color='#1f77b4', marker='D', s=50, zorder=5)
    plt.title('Reconstruction Error vs Number of Components (RP)')
    plt.xlabel('n_components')
    plt.ylabel('error')
    plt.grid(True)
    plt.legend(frameon=True)
    plt.savefig(output_path)
    plt.show()


def isomap_reconstruction_error(X_data: pd.DataFrame, output_path: str):
    num_features = X_data.shape[1]
    k_range = range(1, num_features + 1)
    errors = []
    for k in k_range:
        isomap = Isomap(n_components=k)
        isomap.fit(X_data)
        errors.append(isomap.reconstruction_error())

    elbow_locator = KneeLocator(k_range, errors, curve='convex', direction='decreasing')
    plt.axvline(x=elbow_locator.knee, color='black', label=f'Elbow point', linestyle='--')
    plt.plot(k_range, errors, color='#1f77b4', label='Isomap Reconstruction Error')
    plt.scatter(k_range, errors, color='#1f77b4', marker='D', s=50, zorder=5)
    plt.title('Reconstruction Error vs Number of Components (Isomap)')
    plt.xlabel('n_components')
    plt.ylabel('error')

    plt.grid(True)
    plt.legend(frameon=True)
    plt.savefig(output_path)
    plt.show()
