#!python2.7
# -*- coding: utf-8 -*-
"""
Created by kun on 2016/7/29.
"""

from scikits.learn.base import BaseEstimator

__author__ = 'kun'


class BaseRecommender(BaseEstimator):
    def __init__(self, model, with_preference=False):
        self.model = model
        self.with_preference = with_preference

    def recommend(self, user_id, how_many, **params):
        raise NotImplementedError("BaseRecommender is an abstract class.")
