from typing import List, Union, Optional, Tuple

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import rand_score, homogeneity_completeness_v_measure, mutual_info_score, fowlkes_mallows_score
from constants import SEEDS


def evaluate_models_with_transformers(models: List[Tuple[str, TransformerMixin, ClusterMixin]], X_data, labels):
    dfs = {}
    for model_name, transformer, model in models:
        X_data_transformed = transformer.fit_transform(X_data)
        dfs[model_name] = evaluate_clustering(model, X_data_transformed, labels)
    combined_df = pd.concat(dfs.values(), axis=0, keys=dfs.keys())
    return combined_df


def evaluate_models(models: List[Tuple[str, ClusterMixin]], X_data, labels):
    dfs = {}
    for model_name, model in models:
        dfs[model_name] = evaluate_clustering(model, X_data, labels)
    combined_df = pd.concat(dfs.values(), axis=0, keys=dfs.keys())
    return combined_df


def evaluate_clustering(model: ClusterMixin, X_data, labels):
    predictions = model.fit_predict(X_data)
    rand_index = rand_score(labels, predictions)
    homogeneity, completeness, v_measure_score = homogeneity_completeness_v_measure(labels, predictions)
    mutual_information = mutual_info_score(labels, predictions)
    fowlkes_mallows_index = fowlkes_mallows_score(labels, predictions)
    results_df = pd.DataFrame({
        'Rand Index': [rand_index],
        # 'Homogeneity': [homogeneity],
        # 'Completeness': [completeness],
        'V Measure': [v_measure_score],
        'Mutual Information': [mutual_information],
        'Fowlkes-Mallows index': [fowlkes_mallows_index]
    })
    return results_df


def evaluate_dim_reduction(transformers: List[Tuple[str, TransformerMixin]], X_data, labels, scoring: str):
    dfs = {}
    for transformer_name, transformer in transformers:
        dfs[transformer_name] = evaluate_transformer(transformer, X_data, labels, scoring)
    combined_df = pd.concat(dfs.values(), axis=0, keys=dfs.keys())
    return combined_df


def evaluate_transformer(transformer: TransformerMixin, X_data, y_labels, scoring: str):
    X_data_transformed = transformer.fit_transform(X_data)
    models = [
        DecisionTreeClassifier(max_depth=8),
        RandomForestClassifier(max_depth=10),
        BaggingClassifier(),
        LinearSVC(),
        SGDClassifier(max_iter=100, tol=1e-3),
        LogisticRegression()
    ]
    dfs = []
    for model in models:
        cross_vals = cross_val_score(model, X_data_transformed, y_labels, scoring=scoring)
        df = pd.DataFrame({
            f'{model.__class__.__name__}': [np.mean(cross_vals)]
        })
        dfs.append(df)
    return pd.concat(dfs, axis=1)
