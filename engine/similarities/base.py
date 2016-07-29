#!python2.7
# -*- coding: utf-8 -*-
"""
Created by kun on 2016/7/29.
"""

import numpy as np

__author__ = 'kun'


class BaseSimilarity(object):
    def __init__(self, model, distance, num_best=None):
        self.model = model
        self.distance = distance
        self._set_num_best(num_best)

    def _set_num_best(self, num_best):
        self.num_best = num_best

    def get_similarity(self, source_id, target_id):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def get_similarities(self, source_id):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def __getitem__(self, source_id):
        """
        Get similarities of the `source_id` to all sources in the model.
        """
        all_sims = self.get_similarities(source_id)

        # return either all similarities as a list,
        # or only self.num_best most similar,
        # depending on settings from the constructor

        tops = sorted(all_sims, key=lambda x: -x[1])

        if all_sims:
            item_ids, preferences = zip(*all_sims)
            preferences = np.array(preferences).flatten()
            item_ids = np.array(item_ids).flatten()
            sorted_prefs = np.argsort(-preferences)
            tops = zip(item_ids[sorted_prefs], preferences[sorted_prefs])

        # return at most numBest top 2-tuples (label, sim)
        return tops[:self.num_best] if self.num_best is not None else tops
