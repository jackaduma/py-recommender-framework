#!python2.7
# -*- coding: utf-8 -*-
"""
Created by kun on 2016/8/11.

Generalized Recommender models.

This module contains basic memory recommender interfaces used throughout
the whole scikit-crab package.

The interfaces are realized as abstract base classes (ie., some optional
functionality is provided in the interface itself, so that the interfaces
can be subclassed).

"""

from base import ItemRecommender, UserRecommender
from item_strategies import ItemsNeighborhoodStrategy
from neighborhood_strategies import NearestNeighborsStrategy
import numpy as np

__author__ = 'kun'


class ItemBasedRecommender(ItemRecommender):
    """
    Item Based Collaborative Filtering Recommender.


    Parameters
    -----------
    data_model: The data model instance that will be data source
         for the recommender.

    similarity: The Item Similarity instance that will be used to
        score the items that will be recommended.

    items_selection_strategy: The item candidates strategy that you
     can choose for selecting the possible items to recommend.
     default = ItemsNeighborhoodStrategy

    capper: bool (default=True)
        Cap the preferences with maximum and minimum preferences
        in the model.
    with_preference: bool (default=False)
        Return the recommendations with the estimated preferences if True.

    Attributes
    -----------
    `model`: The data model instance that will be data source
         for the recommender.

    `similarity`: The Item Similarity instance that will be used to
        score the items that will be recommended.

    `items_selection_strategy`: The item candidates strategy that you
         can choose for selecting the possible items to recommend.
         default = ItemsNeighborhoodStrategy

    `capper`: bool (default=True)
        Cap the preferences with maximum and minimum preferences
        in the model.
    `with_preference`: bool (default=False)
        Return the recommendations with the estimated preferences if True.

    Examples
    -----------

    Notes
    -----------
    This ItemBasedRecommender does not yet provide
    suppot for rescorer functions.

    References
    -----------
    Item-based collaborative filtering recommendation algorithms by Sarwar
    http://portal.acm.org/citation.cfm?id=372071

    """

    def __init__(self, model, similarity, items_selection_strategy=None,
                 capper=True, with_preference=False):
        ItemRecommender.__init__(self, model, with_preference)
        self.similarity = similarity
        self.capper = capper
        if items_selection_strategy is None:
            self.items_selection_strategy = ItemsNeighborhoodStrategy()
        else:
            self.items_selection_strategy = items_selection_strategy

    def recommend(self, user_id, how_many=None, **params):
        '''
        Return a list of recommended items, ordered from most strongly
        recommend to least.

        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.
        how_many: int
                 Desired number of recommendations (default=None ALL)

        '''
        self._set_params(**params)

        candidate_items = self.all_other_items(user_id)

        recommendable_items = self._top_matches(user_id, \
                                                candidate_items, how_many)

        return recommendable_items

    def estimate_preference(self, user_id, item_id, **params):
        '''
        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.

        item_id:  int or string
            ID of item for which wants to find the estimated preference.

        Returns
        -------
        Return an estimated preference if the user has not expressed a
        preference for the item, or else the user's actual preference for the
        item. If a preference cannot be estimated, returns None.
        '''
        preference = self.model.preference_value(user_id, item_id)

        if not np.isnan(preference):
            return preference

        # TODO: It needs optimization
        prefs = self.model.preferences_from_user(user_id)

        if not self.model.has_preference_values():
            prefs = [(pref, 1.0) for pref in prefs]

        similarities = \
            np.array([self.similarity.get_similarity(item_id, to_item_id) \
                      for to_item_id, pref in prefs if to_item_id != item_id]).flatten()

        prefs = np.array([pref for it, pref in prefs])
        prefs_sim = np.sum(prefs[~np.isnan(similarities)] *
                           similarities[~np.isnan(similarities)])
        total_similarity = np.sum(similarities)

        # Throw out the estimate if it was based on no data points,
        # of course, but also if based on
        # just one. This is a bit of a band-aid on the 'stock'
        # item-based algorithm for the moment.
        # The reason is that in this case the estimate is, simply,
        # the user's rating for one item
        # that happened to have a defined similarity.
        # The similarity score doesn't matter, and that
        # seems like a bad situation.
        if total_similarity == 0.0 or \
                not similarities[~np.isnan(similarities)].size:
            return np.nan

        estimated = prefs_sim / total_similarity

        if self.capper:
            max_p = self.model.maximum_preference_value()
            min_p = self.model.minimum_preference_value()
            estimated = max_p if estimated > max_p else min_p \
                if estimated < min_p else estimated
        return estimated

    def all_other_items(self, user_id, **params):
        '''
        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.

        Returns
        ---------
        Return items in the `model` for which the user has not expressed
        the preference and could possibly be recommended to the user.

        '''
        return self.items_selection_strategy.candidate_items(user_id, \
                                                             self.model)

    def _top_matches(self, source_id, target_ids, how_many=None, **params):
        '''
        Parameters
        ----------
        target_ids: array of shape [n_target_ids]

        source_id: int or string
                item id to compare against.

        how_many: int
            Desired number of most top items to recommend (default=None ALL)

        Returns
        --------
        Return the top N matches
        It can be user_ids or item_ids.
        '''
        # Empty target_ids
        if target_ids.size == 0:
            return np.array([])

        estimate_preferences = np.vectorize(self.estimate_preference)

        preferences = estimate_preferences(source_id, target_ids)

        preference_values = preferences[~np.isnan(preferences)]
        target_ids = target_ids[~np.isnan(preferences)]

        sorted_preferences = np.lexsort((preference_values,))[::-1]

        sorted_preferences = sorted_preferences[0:how_many] \
            if how_many and sorted_preferences.size > how_many \
            else sorted_preferences

        if self.with_preference:
            top_n_recs = [(target_ids[ind], \
                           preferences[ind]) for ind in sorted_preferences]
        else:
            top_n_recs = [target_ids[ind]
                          for ind in sorted_preferences]

        return top_n_recs

    def most_similar_items(self, item_id, how_many=None):
        '''
        Return the most similar items to the given item, ordered
        from most similar to least.

        Parameters
        -----------
        item_id:  int or string
            ID of item for which to find most similar other items

        how_many: int
            Desired number of most similar items to find (default=None ALL)
        '''
        old_how_many = self.similarity.num_best
        # +1 since it returns the identity.
        self.similarity.num_best = how_many + 1 \
            if how_many is not None else None
        similarities = self.similarity[item_id]
        self.similarity.num_best = old_how_many

        return np.array([item for item, pref in similarities \
                         if item != item_id and not np.isnan(pref)])

    def recommended_because(self, user_id, item_id, how_many=None, **params):
        '''
        Returns the items that were most influential in recommending a
        given item to a given user. In most implementations, this
        method will return items that the user prefers and that
        are similar to the given item.

        Parameters
        -----------
        user_id : int or string
            ID of the user who was recommended the item

        item_id: int or string
            ID of item that was recommended

        how_many: int
            Maximum number of items to return (default=None ALL)

        Returns
        ----------
        The list of items ordered from most influential in
        recommended the given item to least
        '''
        preferences = self.model.preferences_from_user(user_id)

        if self.model.has_preference_values():
            similarities = \
                np.array([self.similarity.get_similarity(item_id, to_item_id) \
                          for to_item_id, pref in preferences
                          if to_item_id != item_id]).flatten()
            prefs = np.array([pref for it, pref in preferences])
            item_ids = np.array([it for it, pref in preferences])
        else:
            similarities = \
                np.array([self.similarity.get_similarity(item_id, to_item_id) \
                          for to_item_id in preferences
                          if to_item_id != item_id]).flatten()
            prefs = np.array([1.0 for it in preferences])
            item_ids = np.array(preferences)

        scores = prefs[~np.isnan(similarities)] * \
                 (1.0 + similarities[~np.isnan(similarities)])

        sorted_preferences = np.lexsort((scores,))[::-1]

        sorted_preferences = sorted_preferences[0:how_many] \
            if how_many and sorted_preferences.size > how_many \
            else sorted_preferences

        if self.with_preference:
            top_n_recs = [(item_ids[ind], \
                           prefs[ind]) for ind in sorted_preferences]
        else:
            top_n_recs = [item_ids[ind]
                          for ind in sorted_preferences]

        return top_n_recs


# =====================
# User Based Recommender

class UserBasedRecommender(UserRecommender):
    """
    User Based Collaborative Filtering Recommender.


    Parameters
    -----------
    data_model: The data model instance that will be data source
         for the recommender.

    similarity: The User Similarity instance that will be used to
        score the users that are the most similar to the user.

    neighborhood_strategy: The user neighborhood strategy that you
         can choose for selecting the most similar users to find
         the items to recommend.
         default = NearestNeighborsStrategy

    capper: bool (default=True)
        Cap the preferences with maximum and minimum preferences
        in the model.
    with_preference: bool (default=False)
        Return the recommendations with the estimated preferences if True.

    Attributes
    -----------
    `model`: The data model instance that will be data source
         for the recommender.

    `similarity`: The User Similarity instance that will be used to
        score the users that are the most similar to the user.

    `neighborhood_strategy`: The user neighborhood strategy that you
         can choose for selecting the most similar users to find
         the items to recommend.
         default = NearestNeighborsStrategy

    `capper`: bool (default=True)
        Cap the preferences with maximum and minimum preferences
        in the model.
    `with_preference`: bool (default=False)
        Return the recommendations with the estimated preferences if True.

    Examples
    -----------

    Notes
    -----------
    This UserBasedRecommender does not yet provide
    suppot for rescorer functions.

    References
    -----------
    User-based collaborative filtering recommendation algorithms by

    """

    def __init__(self, model, similarity, neighborhood_strategy=None,
                 capper=True, with_preference=False):
        UserRecommender.__init__(self, model, with_preference)
        self.similarity = similarity
        self.capper = capper
        if neighborhood_strategy is None:
            self.neighborhood_strategy = NearestNeighborsStrategy()
        else:
            self.neighborhood_strategy = neighborhood_strategy

    def all_other_items(self, user_id, **params):
        '''
        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed. (default= 'user_similarity')

        Optional Parameters
        --------------------
        n_similarity: string
            The similarity used in the neighborhood strategy

        distance: the metrics.pairwise function to set.
                The pairwise function to compute the similarity (default = euclidean_distances)

        nhood_size:  int
            The neighborhood size (default=None  ALL)

        minimal_similarity: float
            minimal similarity required for neighbors (default = 0.0)

        sampling_rate: int
            percentage of users to consider when building neighborhood
                (default = 1)

        Returns
        ---------
        Return items in the `model` for which the user has not expressed
        the preference and could possibly be recommended to the user.

        '''
        n_similarity = params.pop('n_similarity', 'user_similarity')
        distance = params.pop('distance', self.similarity.distance)
        nhood_size = params.pop('nhood_size', None)

        nearest_neighbors = self.neighborhood_strategy.user_neighborhood(user_id,
                                                                         self.model, n_similarity, distance, nhood_size,
                                                                         **params)

        items_from_user_id = self.model.items_from_user(user_id)
        possible_items = []
        for to_user_id in nearest_neighbors:
            possible_items.extend(self.model.items_from_user(to_user_id))

        possible_items = np.unique(np.array(possible_items).flatten())

        return np.setdiff1d(possible_items, items_from_user_id)

    def estimate_preference(self, user_id, item_id, **params):
        '''
        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.

        item_id:  int or string
            ID of item for which wants to find the estimated preference.

        Returns
        -------
        Return an estimated preference if the user has not expressed a
        preference for the item, or else the user's actual preference for the
        item. If a preference cannot be estimated, returns None.
        '''

        preference = self.model.preference_value(user_id, item_id)
        if not np.isnan(preference):
            return preference

        n_similarity = params.pop('n_similarity', 'user_similarity')
        distance = params.pop('distance', self.similarity.distance)
        nhood_size = params.pop('nhood_size', None)

        nearest_neighbors = self.neighborhood_strategy.user_neighborhood(user_id,
                                                                         self.model, n_similarity, distance, nhood_size,
                                                                         **params)

        preference = 0.0
        total_similarity = 0.0

        similarities = np.array([self.similarity.get_similarity(user_id, to_user_id)
                                 for to_user_id in nearest_neighbors]).flatten()

        prefs = np.array([self.model.preference_value(to_user_id, item_id)
                          for to_user_id in nearest_neighbors])

        prefs = prefs[~np.isnan(prefs)]
        similarities = similarities[~np.isnan(prefs)]

        prefs_sim = np.sum(prefs[~np.isnan(similarities)] *
                           similarities[~np.isnan(similarities)])
        total_similarity = np.sum(similarities)

        # Throw out the estimate if it was based on no data points,
        # of course, but also if based on just one. This is a bit
        # of a band-aid on the 'stock' item-based algorithm for
        # the moment. The reason is that in this case the estimate
        # is, simply, the user's rating for one item that happened
        # to have a defined similarity. The similarity score doesn't
        # matter, and that seems like a bad situation.
        if total_similarity == 0.0 or \
                not similarities[~np.isnan(similarities)].size:
            return np.nan

        estimated = prefs_sim / total_similarity

        if self.capper:
            max_p = self.model.maximum_preference_value()
            min_p = self.model.minimum_preference_value()
            estimated = max_p if estimated > max_p else min_p \
                if estimated < min_p else estimated

        return estimated

    def most_similar_users(self, user_id, how_many=None):
        '''
        Return the most similar users to the given user, ordered
        from most similar to least.

        Parameters
        -----------
        user_id:  int or string
            ID of user for which to find most similar other users

        how_many: int
            Desired number of most similar users to find (default=None ALL)
        '''
        old_how_many = self.similarity.num_best
        # +1 since it returns the identity.
        self.similarity.num_best = how_many + 1 \
            if how_many is not None else None
        similarities = self.similarity[user_id]
        self.similarity.num_best = old_how_many
        return np.array([to_user_id for to_user_id, pref in similarities \
                         if user_id != to_user_id and not np.isnan(pref)])

    def recommend(self, user_id, how_many=None, **params):
        '''
        Return a list of recommended items, ordered from most strongly
        recommend to least.

        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.
        how_many: int
                 Desired number of recommendations (default=None ALL)

        '''

        self._set_params(**params)

        candidate_items = self.all_other_items(user_id, **params)

        recommendable_items = self._top_matches(user_id, \
                                                candidate_items, how_many)

        return recommendable_items

    def _top_matches(self, source_id, target_ids, how_many=None, **params):
        '''
        Parameters
        ----------
        target_ids: array of shape [n_target_ids]

        source_id: int or string
                item id to compare against.

        how_many: int
            Desired number of most top items to recommend (default=None ALL)

        Returns
        --------
        Return the top N matches
        It can be user_ids or item_ids.
        '''
        # Empty target_ids
        if target_ids.size == 0:
            return np.array([])

        estimate_preferences = np.vectorize(self.estimate_preference)

        preferences = estimate_preferences(source_id, target_ids)

        preference_values = preferences[~np.isnan(preferences)]
        target_ids = target_ids[~np.isnan(preferences)]

        sorted_preferences = np.lexsort((preference_values,))[::-1]

        sorted_preferences = sorted_preferences[0:how_many] \
            if how_many and sorted_preferences.size > how_many \
            else sorted_preferences

        if self.with_preference:
            top_n_recs = [(target_ids[ind], \
                           preferences[ind]) for ind in sorted_preferences]
        else:
            top_n_recs = [target_ids[ind]
                          for ind in sorted_preferences]

        return top_n_recs

    def recommended_because(self, user_id, item_id, how_many=None, **params):
        '''
        Returns the users that were most influential in recommending a
        given item to a given user. In most implementations, this
        method will return users that prefers the recommended item and that
        are similar to the given user.

        Parameters
        -----------
        user_id : int or string
            ID of the user who was recommended the item

        item_id: int or string
            ID of item that was recommended

        how_many: int
            Maximum number of items to return (default=None ALL)

        Returns
        ----------
        The list of items ordered from most influential in
        recommended the given item to least
        '''
        preferences = self.model.preferences_for_item(item_id)

        if self.model.has_preference_values():
            similarities = \
                np.array([self.similarity.get_similarity(user_id, to_user_id) \
                          for to_user_id, pref in preferences
                          if to_user_id != user_id]).flatten()
            prefs = np.array([pref for it, pref in preferences])
            user_ids = np.array([usr for usr, pref in preferences])
        else:
            similarities = \
                np.array([self.similarity.get_similarity(user_id, to_user_id) \
                          for to_user_id in preferences
                          if to_user_id != user_id]).flatten()
            prefs = np.array([1.0 for it in preferences])
            user_ids = np.array(preferences)

        scores = prefs[~np.isnan(similarities)] * \
                 (1.0 + similarities[~np.isnan(similarities)])

        sorted_preferences = np.lexsort((scores,))[::-1]

        sorted_preferences = sorted_preferences[0:how_many] \
            if how_many and sorted_preferences.size > how_many \
            else sorted_preferences

        if self.with_preference:
            top_n_recs = [(user_ids[ind], \
                           prefs[ind]) for ind in sorted_preferences]
        else:
            top_n_recs = [user_ids[ind]
                          for ind in sorted_preferences]

        return top_n_recs
