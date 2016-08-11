#!python2.7
# -*- coding: utf-8 -*-
"""
Created by kun on 2016/8/11.

This module contains basic implementations that encapsulate
    retrieval-related statistics about the quality of the recommender's
    recommendations.

"""

import numpy as np
from ..utils import check_arrays, unique_labels

__author__ = 'kun'


def root_mean_square_error(y_real, y_pred):
    """
    It computes the root mean squared difference (RMSE)
    between predicted and actual ratings for users.

    Positive floating point value: the best value is 0.0.

    :param y_real: array-like
    :param y_pred: array-like
    :return: the mean square error
    """
    y_real, y_pred = check_arrays(y_real, y_pred)

    return np.sqrt((np.sum((y_pred - y_real) ** 2)) / y_real.shape[0])


def mean_absolute_error(y_real, y_pred):
    y_real, y_pred = check_arrays(y_real, y_pred)

    return np.sum(np.abs(y_pred - y_real)) / y_real.size


def normalized_mean_absolute_error(y_real, y_pred, max_rating, min_rating):
    y_real, y_pred = check_arrays(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    return mae / (max_rating - min_rating)


def evaluation_error(y_real, y_pred, max_rating, min_rating):
    mae = mean_absolute_error(y_real, y_pred)
    nmae = normalized_mean_absolute_error(y_real, y_pred,
                                          max_rating, min_rating)
    rmse = root_mean_square_error(y_real, y_pred)

    return mae, nmae, rmse


def precision_score(y_real, y_pred):
    p, _, _ = precision_recall_fscore(y_real, y_pred)
    return np.average(p)


def recall_score(y_real, y_pred):
    _, r, _ = precision_recall_fscore(y_real, y_pred)
    return np.average(r)


def f1_score(y_real, y_pred):
    return fbeta_score(y_real, y_pred, 1)


def fbeta_score(y_real, y_pred, beta):
    _, _, f = precision_recall_fscore(y_real, y_pred, beta=beta)

    return np.average(f)


def precision_recall_fscore(y_real, y_pred, beta=1.0):
    y_real, y_pred = check_arrays(y_real, y_pred)
    assert (beta > 0)

    n_users = y_real.shape[0]
    precision = np.zeros(n_users, dtype=np.double)
    recall = np.zeros(n_users, dtype=np.double)
    fscore = np.zeros(n_users, dtype=np.double)

    try:
        # oddly, we may get an "invalid" rather than a "divide" error here
        old_err_settings = np.seterr(divide='ignore', invalid='ignore')

        for i, y_items_pred in enumerate(y_pred):
            intersection_size = np.intersect1d(y_items_pred, y_real[i]).size
            precision[i] = (intersection_size / float(len(y_real[i]))) \
                if len(y_real[i])  else 0.0
            recall[i] = (intersection_size / float(len(y_items_pred))) \
                if len(y_items_pred) else 0.0

        # handle division by 0.0 in precision and recall
        precision[np.isnan(precision)] = 0.0
        recall[np.isnan(precision)] = 0.0

        # fbeta Score
        beta2 = beta ** 2
        fscore = (1 + beta2) * (precision * recall) \
                 / (beta2 * precision + recall)

        # handle division by 0.0 in fscore
        fscore[(precision + recall) == 0.0] = 0.0

    finally:
        np.seterr(**old_err_settings)

    return precision, recall, fscore


def evaluation_report(y_real, y_pred, labels=None, target_names=None):
    if labels is None:
        labels = unique_labels(y_real)
    else:
        labels = np.asarray(labels, dtype=np.int)

    last_line_heading = 'avg / total'

    if target_names is None:
        width = len(last_line_heading)
        target_names = ['%d' % l for l in labels]
    else:
        width = max(len(cn) for cn in target_names)
        width = max(width, len(last_line_heading))

    headers = ["precision", "recall", "f1-score"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'
    p, r, f1 = precision_recall_fscore(y_real, y_pred)
    for i, label in enumerate(labels):
        values = [target_names[i]]
        for v in (p[i], r[i], f1[i]):
            values += ["%0.2f" % float(v)]
        report += fmt % tuple(values)

    report += '\n'

    # compute averages
    values = [last_line_heading]
    for v in (np.average(p),
              np.average(r),
              np.average(f1)):
        values += ["%0.2f" % float(v)]
    report += fmt % tuple(values)
    return report
