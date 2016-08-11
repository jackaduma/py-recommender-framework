#!python2.7
# -*- coding: utf-8 -*-
"""
Created by kun on 2016/7/29.
"""

__author__ = 'kun'


class RecommenderEvaluator(object):
    """
        Basic Interface which is responsible to evaluate the quality of Recommender
        recommendations. The range of values that may be returned depends on the
        implementation. but lower values must mean better recommendations, with 0
        being the lowest / best possible evaluation, meaning a perfect match.
    """

    def evaluate(self, recommender, metrics=None, **kwargs):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def evaluate_online(self, metrics=None, **kwargs):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def evaluate_on_split(self, metrics=None, **kwargs):
        raise NotImplementedError("cannot instantiate Abstract Base Class")
