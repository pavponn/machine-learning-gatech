from functools import partial

import numpy as np
from sklearn.metrics import precision_recall_curve


def max_recall_at_precision_k(k, y_true, probas_pred) -> float:
    if k < 0 or k > 1:
        raise ValueError("Value of k must be in range 0 and 1")
    precisions, recalls, _thresholds = precision_recall_curve(
        y_true, probas_pred
    )
    valid_positions = precisions >= k
    valid_recalls = recalls[valid_positions]
    value = 0.0
    if valid_recalls.shape[0] > 0:
        value = np.max(valid_recalls)
    return value


def recall_at_precision_stats(k, clf, X_tr, X_t, y_tr, y_t):
    y_train_proba = clf.predict_proba(X_tr)
    y_test_proba = clf.predict_proba(X_t)
    train_r_at_pr_k = max_recall_at_precision_k(k, y_tr.values, y_train_proba[:, 1])
    test_r_at_pr_k = max_recall_at_precision_k(k, y_t.values, y_test_proba[:, 1])

    print(f"Recall @ Precision 0.9 (train): {train_r_at_pr_k:.4f}")
    print(f"Recall @ Precision 0.9 (test) : {test_r_at_pr_k:.4f}")


def recall_at_precision_k_scoring(k, clf, X_m, y_m):
    y_m_pred = clf.predict_proba(X_m)
    return max_recall_at_precision_k(k, y_m, y_m_pred[:, 1])


recall_at_precision_90_scoring = partial(recall_at_precision_k_scoring, 0.9)
