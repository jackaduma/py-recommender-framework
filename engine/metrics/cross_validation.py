#!python2.7
# -*- coding: utf-8 -*-
"""
Created by kun on 2016/8/11.
"""

import numpy as np
from ..utils.extmath import factorial, combinations
from ..utils import check_random_state
from math import ceil

__author__ = 'kun'


class LeaveOneOut(object):
    def __init__(self, n, indices=False):
        self.n = n
        self.indices = indices

    def __iter__(self):
        n = self.n
        for i in xrange(n):
            test_index = np.zeros(n, dtype=np.bool)
            test_index[i] = True
            train_index = np.logical_not(test_index)
            if self.indices:
                ind = np.arange(n)
                train_index = ind[train_index]
                test_index = ind[test_index]
            yield train_index, test_index

    def __repr__(self):
        return '%s.%s(n=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
        )

    def __len__(self):
        return self.n


class LeavePOut(object):
    def __init__(self, n, p, indices=False):
        self.n = n
        self.p = p
        self.indices = indices

    def __iter__(self):
        n = self.n
        p = self.p
        comb = combinations(range(n), p)
        for idx in comb:
            test_index = np.zeros(n, dtype=np.bool)
            test_index[np.array(idx)] = True
            train_index = np.logical_not(test_index)
            if self.indices:
                ind = np.arange(n)
                train_index = ind[train_index]
                test_index = ind[test_index]
            yield train_index, test_index

    def __repr__(self):
        return '%s.%s(n=%i, p=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.p,
        )

    def __len__(self):
        return (factorial(self.n) / factorial(self.n - self.p)
                / factorial(self.p))


class KFold(object):
    def __init__(self, n, k, indices=False):
        assert k > 0, ValueError('Cannot have number of folds k below 1.')
        assert k <= n, ValueError('Cannot have number of folds k=%d, '
                                  'greater than the number '
                                  'of samples: %d.' % (k, n))
        self.n = n
        self.k = k
        self.indices = indices

    def __iter__(self):
        n = self.n
        k = self.k
        j = ceil(n / k)

        for i in xrange(k):
            test_index = np.zeros(n, dtype=np.bool)
            if i < k - 1:
                test_index[i * j:(i + 1) * j] = True
            else:
                test_index[i * j:] = True
            train_index = np.logical_not(test_index)
            if self.indices:
                ind = np.arange(n)
                train_index = ind[train_index]
                test_index = ind[test_index]
            yield train_index, test_index

    def __repr__(self):
        return '%s.%s(n=%i, k=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.k,
        )

    def __len__(self):
        return self.k


class ShuffleSplit(object):
    def __init__(self, n, n_iterations=10, test_fraction=0.1,
                 indices=False, random_state=None):
        self.n = n
        self.n_iterations = n_iterations
        self.test_fraction = test_fraction
        self.random_state = random_state
        self.indices = indices

    def __iter__(self):
        rng = self.random_state = check_random_state(self.random_state)
        n_test = ceil(self.test_fraction * self.n)
        for i in range(self.n_iterations):
            # random partition
            permutation = rng.permutation(self.n)
            ind_train = permutation[:-n_test]
            ind_test = permutation[-n_test:]
            if self.indices:
                yield ind_train, ind_test
            else:
                train_mask = np.zeros(self.n, dtype=np.bool)
                train_mask[ind_train] = True
                test_mask = np.zeros(self.n, dtype=np.bool)
                test_mask[ind_test] = True
                yield train_mask, test_mask

    def __repr__(self):
        return ('%s(%d, n_iterations=%d, test_fraction=%s, indices=%s, '
                'random_state=%d)' % (
                    self.__class__.__name__,
                    self.n,
                    self.n_iterations,
                    str(self.test_fraction),
                    self.indices,
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iterations
