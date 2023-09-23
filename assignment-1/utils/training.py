import time

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier

from utils.metrics import recall_at_precision_90_scoring


def perform_hyperparameter_tuning(clf,
                                  X_tr,
                                  y_tr,
                                  param_space,
                                  cv,
                                  scoring=recall_at_precision_90_scoring,
                                  n_jobs=-1):
    param_search = GridSearchCV(
        estimator=clf,
        param_grid=param_space,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs
    )

    param_search.fit(X_tr, y_tr)
    return param_search.best_params_


def learning_time_stats(clf, X_tr, y_tr):
    start_time = time.time()
    clf.fit(X_tr, y_tr)
    end_time = time.time()
    train_time = end_time - start_time
    print(f"Training time, total: {train_time:.4f} seconds")
    print(f"Training time, normalized (per 1000 samples): {(train_time * 1000 / len(X_tr)):.4f} seconds")


def inference_time_stats(clf, X_pr, proba=True, dataset_name='all'):
    start_time = time.time()
    if proba:
        clf.predict_proba(X_pr)
    else:
        clf.predict(X_pr)
    end_time = time.time()
    inf_time = end_time - start_time
    print(f"Inference time ({dataset_name}), total: {inf_time:.4f} seconds")
    print(
        f"Inference time ({dataset_name}), normalized (per 1000 samples): {(inf_time * 1000 / len(X_pr)):.4f} seconds")


def mlp_epochs_training(clf: MLPClassifier, X_tr, y_tr, epochs, random_state, scoring=recall_at_precision_90_scoring):
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=random_state)

    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    train_score = np.zeros(epochs)
    val_score = np.zeros(epochs)

    for epoch in range(epochs):
        clf.fit(X_tr, y_tr)

        train_score[epoch] = scoring(clf, X_train, y_train)
        train_loss[epoch] = clf.loss_

        val_score[epoch] = scoring(clf, X_val, y_val)
        val_loss[epoch] = clf.val
