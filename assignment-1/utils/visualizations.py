import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import sklearn.model_selection as sk_ms
from sklearn.neural_network import MLPClassifier
from yellowbrick import set_aesthetic
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.model_selection import ValidationCurve, LearningCurve
from yellowbrick.classifier import ConfusionMatrix

from utils.metrics import recall_at_precision_90_scoring
from utils.plot_utls import setup_plots


def precision_recall_curve(clf, X_tr, y_tr, X_t, y_t, title: str, output_path: str):
    set_aesthetic()
    setup_plots()
    viz = PrecisionRecallCurve(clf, title=title, is_fitted=True, ap_score=False)
    viz.fit(X_tr, y_tr)
    viz.score(X_tr, y_tr)
    viz.score(X_t, y_t)
    viz.finalize()
    ax = viz.ax
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["Precision-Recall Curve (train)", "Precision-Recall Curve (test)"]
    ax.legend(handles, new_labels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        handles,
        new_labels,
        frameon=True,
        # bbox_to_anchor=(1.02, 1),
        # borderaxespad=0.0,
        loc='lower left',
    )
    path_to_create = Path(output_path).parent.absolute()
    os.makedirs(path_to_create, exist_ok=True)
    plt.savefig(output_path)
    plt.show()


def validation_curve(clf, X_tr, y_tr, param_name, param_range, output_path, cv,
                     scoring=recall_at_precision_90_scoring,
                     n_jobs=-1, logx=False):
    set_aesthetic()
    setup_plots()
    viz = ValidationCurve(
        clf,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        logx=logx
    )
    viz.fit(X_tr, y_tr)
    viz.show(outpath=output_path)


def learning_curve(clf, X_tr, y_tr, cv, output_path, scoring=recall_at_precision_90_scoring):
    set_aesthetic()
    setup_plots()
    viz = LearningCurve(clf, scoring=scoring, cv=cv, train_sizes=np.arange(0.1, 1.001, 0.1))
    viz.fit(X_tr, y_tr)

    viz.finalize()

    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int(100 * x / (X_tr.shape[0] - int(X_tr.shape[0] / cv)))}%')
    )

    path_to_create = Path(output_path).parent.absolute()
    os.makedirs(path_to_create, exist_ok=True)
    plt.savefig(output_path)
    plt.show()


def validation_bar_chart(clf,
                         X_tr,
                         y_tr,
                         cv,
                         param_name,
                         param_range,
                         output_path,
                         scoring=recall_at_precision_90_scoring,

                         n_jobs=-1):
    train_scores, cv_scores = sk_ms.validation_curve(clf, X_tr, y_tr, param_name=param_name, param_range=param_range,
                                                     cv=cv, scoring=scoring,
                                                     n_jobs=n_jobs)
    set_aesthetic()
    setup_plots()
    param_positions = np.arange(len(param_range))

    fig, ax = plt.subplots()

    # Set Yellowbrick-like style
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', colors='black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')

    bar_width = 0.4
    train_means = np.mean(train_scores, axis=1)
    cv_means = np.mean(cv_scores, axis=1)

    train_variances = np.std(train_scores, axis=1)
    cv_variances = np.std(cv_scores, axis=1)

    # Set position of bar on X axis

    plt.bar(param_positions,
            train_means,
            yerr=train_variances,
            color='#1f77b4',
            width=bar_width,
            error_kw={
                'capsize': 5,
                'capthick': 2,
                'elinewidth': 1,
                'markeredgewidth': 1,
                'color': '#1f77b4',
            },
            align='center',
            label='Train Score',
            alpha=0.5)

    plt.bar(param_positions + bar_width,
            cv_means,
            color='#ff7f0e',
            width=bar_width,
            yerr=cv_variances,
            error_kw={
                'capsize': 5,
                'capthick': 2,
                'elinewidth': 1,
                'markeredgewidth': 1,
                'color': '#ff7f0e',
            },
            label='Cross Validation Score',
            alpha=0.5)

    for r, train_score, test_score, train_variance, cv_variance in zip(param_positions, train_means, cv_means,
                                                                       train_variances, cv_variances):
        plt.text(r, train_score + train_variance + 0.01, f'{train_score:.2f}', ha='center', va='bottom', color='black')
        plt.text(r + bar_width, test_score + cv_variance + 0.01, f'{test_score:.2f}', ha='center', va='bottom',
                 color='black')

    plt.xlabel(param_name)
    plt.ylabel('score')
    plt.title(f'Validation Bar Chart for {clf.__class__.__name__}')

    plt.xticks(param_positions + bar_width / 2, param_range)
    plt.legend(frameon=True, loc='lower left')
    plt.savefig(output_path)

    # Show the plot
    plt.show()


def validation_curve_complex(clf,
                             X_tr,
                             y_tr,
                             cv,
                             param_name,
                             param_name_print,
                             param_range,
                             param_range_print,
                             output_path,
                             scoring=recall_at_precision_90_scoring,
                             legend_loc='lower left',
                             n_jobs=-1):
    train_scores, cv_scores = sk_ms.validation_curve(clf, X_tr, y_tr, param_name=param_name, param_range=param_range,
                                                     cv=cv, scoring=scoring,
                                                     n_jobs=n_jobs)

    set_aesthetic()
    setup_plots()
    fig, ax = plt.subplots()

    # Set Yellowbrick-like style
    # ax.spines['top'].set_color('none')
    # ax.spines['right'].set_color('none')
    # ax.spines['left'].set_color('#d9d9d9')
    # ax.spines['bottom'].set_color('#d9d9d9')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', colors='black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')

    train_means = np.mean(train_scores, axis=1)
    cv_means = np.mean(cv_scores, axis=1)

    train_variances = np.std(train_scores, axis=1)
    cv_variances = np.std(cv_scores, axis=1)

    plt.plot(param_range_print, train_means, color='#1f77b4', label='Train Score')
    plt.fill_between(param_range_print, train_means - np.sqrt(train_variances), train_means + np.sqrt(train_variances),
                     color='#1f77b4', alpha=0.2)
    plt.plot(param_range_print, cv_means, color='#2ca02c', label='Cross Validation Score')
    plt.fill_between(param_range_print, cv_means - np.sqrt(cv_variances), cv_means + np.sqrt(cv_variances),
                     color='#2ca02c', alpha=0.2)

    plt.scatter(param_range_print, train_means, color='#1f77b4', marker='D', s=50, zorder=5)
    plt.scatter(param_range_print, cv_means, color='#2ca02c', marker='D', s=50, zorder=5)

    plt.xlabel(param_name_print)
    plt.ylabel('score')
    plt.title(f'Validation Curve for {clf.__class__.__name__}')

    plt.xticks(param_range_print)
    plt.legend(frameon=True, loc=legend_loc)
    plt.savefig(output_path)

    # Show the plot
    plt.show()


def loss_curve(clf: MLPClassifier,
               output_path,
               legend_loc='lower left'):
    set_aesthetic()
    setup_plots()
    fig, ax = plt.subplots()

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', colors='black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')

    plt.plot(clf.loss_curve_, color='#1f77b4', label='Train Loss')

    plt.ylabel('score')
    plt.title(f'Loss Curve for {clf.__class__.__name__}')

    plt.legend(frameon=True, loc=legend_loc)
    plt.savefig(output_path)

    # Show the plot
    plt.show()


def confusion_matrix(clf, X, y, classes=None):
    if classes is None:
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    cm = ConfusionMatrix(clf, classes=classes, is_fitted=True)
    cm.score(X, y)
    plt.show()
