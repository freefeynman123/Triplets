import numpy as np
from sklearn.metrics import average_precision_score


def mean_average_precision(y_true, y_score, **kwargs):
    classes = np.unique(y_true).astype(int)
    results = []
    for index in classes:
        indices = np.where(y_true == index)[0]
        y_true_score = np.zeros(len(y_true))
        y_true_score[indices] = 1
        results.append(average_precision_score(y_true_score, y_score[:, index]))
    return np.mean(results)
