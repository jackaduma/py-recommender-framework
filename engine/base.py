#!python2.7
# -*- coding: utf-8 -*-
"""
Created by kun on 2016/7/29.
Base Recommender Models.
"""

from scikits.learn.base import BaseEstimator

__author__ = 'kun'


class BaseRecommender(BaseEstimator):
    u"""
    为用户推荐物品的基类

    Attributes
    ----------
     model:  DataModel
            物品的data model

     with_preference: bool (default= False)
            推荐算法是否有 评估参数
    """

    def __init__(self, model, with_preference=False):
        self.model = model
        self.with_preference = with_preference

    def recommend(self, user_id, how_many, **params):
        u"""
        :param user_id:  int or string
            推荐计算的用户对象
        :param how_many: int
            需要推荐物品的个数
        :param params: function, optional
            Rescoring function to apply before final list of recommendations.
        :return:
            Return a list of recommended items, ordered from most strongly recommend to least.
        """
        raise NotImplementedError("BaseRecommender is an abstract class.")

    def estimate_preference(self, user_id, item_id, **params):
        u"""
        :param user_id: int or string
                 User for which recommendations are to be computed.
        :param item_id: int or string
                Item for which recommendations are to be computed.
        :param params:
        :return:
                Return an estimated preference if the user has not expressed a
                preference for the item, or else the user's actual preference for the
                item. If a preference cannot be estimated, returns None.
        """
        raise NotImplementedError("BaseRecommender is an abstract class.")

    def all_other_items(self, user_id, **params):
        u"""
        :param user_id: int or string
                 User for which recommendations are to be computed.
        :param params:
        :return:
                Return all items in the `model` for which the user has not expressed
                the preference and could possibly be recommended to the user.
        """
        raise NotImplementedError("BaseRecommender is an abstract class.")

    def set_preference(self, user_id, item_id, value):
        u"""
        Set a new preference of a user for a specific item with a certain
        magnitude.
        :param user_id: int or string
                 User for which the preference will be updated.
        :param item_id: int or string
                 Item that will be updated.
        :param value: The new magnitude for the preference of a item_id from a
                user_id.
        :return:
        """
        self.model.set_preference(user_id, item_id, value)

    def remove_preference(self, user_id, item_id):
        u"""
        Remove a preference of a user for a specific item
        :param user_id: int or string
                 User for which recommendations are to be computed.
        :param item_id: int or string
                 Item that will be removed the preference for the user_id.
        :return:
        """
        self.model.remove_preference(user_id, item_id)
