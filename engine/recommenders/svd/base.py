#!python2.7
# -*- coding: utf-8 -*-
"""
Created by kun on 2016/8/11.

Generalized Recommender models amd utility classes.

This module contains basic memory recommender interfaces used throughout
the whole scikit-crab package as also utility classes.

The interfaces are realized as abstract base classes (ie., some optional
functionality is provided in the interface itself, so that the interfaces
can be subclassed).

"""

from ..base import MemoryBasedRecommender

__author__ = 'kun'


class SVDRecommender(MemoryBasedRecommender):
    def factorize(self):
        '''
        Factorize the ratings matrix with a factorization
         technique implemented in this method.

        Parameters
        -----------

        Returns
        -----------
        '''
        raise NotImplementedError("ItemRecommender is an abstract class.")

    def train(self):
        '''
        Train the recommender with the matrix factorization method chosen.

        Parameters
        -----------

        Returns
        ----------

        '''
        raise NotImplementedError("ItemRecommender is an abstract class.")
