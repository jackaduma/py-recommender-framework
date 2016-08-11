#!python2.7
# -*- coding: utf-8 -*-
"""
Created by kun on 2016/8/11.

This module contains main implementations that encapsulate
    retrieval-related statistics about the quality of the recommender's
    recommendations.

"""

import operator
import numpy as np
from base import RecommenderEvaluator
from metrics import root_mean_square_error
from metrics import mean_absolute_error
from metrics import normalized_mean_absolute_error
from metrics import evaluation_error
from cross_validation import KFold
from metrics import precision_score
from metrics import recall_score
from metrics import f1_score
from sampling import SplitSampling
from scikits.learn.base import clone
from ..models.utils import ItemNotFoundError, UserNotFoundError

__author__ = 'kun'

# Collaborative Filtering Evaluator
# ==================================

evaluation_metrics = {
    'rmse': root_mean_square_error,
    'mae': mean_absolute_error,
    'nmae': normalized_mean_absolute_error,
    'precision': precision_score,
    'recall': recall_score,
    'f1score': f1_score
}


def check_sampling(sampling, n):
    if sampling is None:
        sampling = 1.0
    if operator.isNumberType(sampling):
        sampling = SplitSampling(n, evaluation_fraction=sampling)

    return sampling


def check_cv(cv, n):
    if cv is None:
        cv = 3
    if operator.isNumberType(cv):
        cv = KFold(n, cv, indices=True)

    return cv


class CfEvaluator(RecommenderEvaluator):
    def _build_recommender(self, dataset, recommender):
        recommender_training = clone(recommender)
        # if the recommender's model has the build_model implemented.

        if not recommender.model.has_preference_values():
            recommender_training.model.dataset = recommender_training.model._load_dataset(dataset.copy())
        else:
            recommender_training.model.dataset = dataset

        if hasattr(recommender_training.model, 'build_model'):
            recommender_training.model.build_model()

        return recommender_training

    def evaluate(self, recommender, metric=None, **kwargs):
        sampling_users = kwargs.pop('sampling_users', None)
        sampling_ratings = kwargs.pop('sampling_ratings', 0.7)
        permutation = kwargs.pop('permutation', True)
        at = kwargs.pop('at', 3)

        if metric not in evaluation_metrics and metric is not None:
            raise ValueError('metric %s is not recognized. valid keywords \
                      are %s' % (metric, evaluation_metrics.keys()))

        n_users = recommender.model.users_count()
        sampling_users = check_sampling(sampling_users, n_users)
        users_set, _ = sampling_users.split(permutation=permutation)

        training_set = {}
        testing_set = {}

        # Select the users to be evaluated.
        user_ids = recommender.model.user_ids()
        for user_id in user_ids[users_set]:
            # Select the ratings to be evaluated.
            preferences = recommender.model.preferences_from_user(user_id)

            sampling_eval = check_sampling(sampling_ratings, \
                                           len(preferences))
            train_set, test_set = sampling_eval.split(indices=True,
                                                      permutation=permutation)

            preferences = list(preferences)
            if recommender.model.has_preference_values():
                training_set[user_id] = dict((preferences[idx]
                                              for idx in train_set)) if preferences else {}
                testing_set[user_id] = [preferences[idx]
                                        for idx in test_set] if preferences else []
            else:
                training_set[user_id] = dict(((preferences[idx], 1.0)
                                              for idx in train_set)) if preferences else {}
                testing_set[user_id] = [(preferences[idx], 1.0)
                                        for idx in test_set] if preferences else []

        # Evaluate the recommender.
        recommender_training = self._build_recommender(training_set, \
                                                       recommender)

        real_preferences = []
        estimated_preferences = []

        for user_id, preferences in testing_set.iteritems():
            for item_id, preference in preferences:
                # Estimate the preferences
                try:
                    estimated = recommender_training.estimate_preference(
                        user_id, item_id)
                    real_preferences.append(preference)
                except ItemNotFoundError:
                    # It is possible that an item exists in the test data but
                    # not training data in which case an exception will be
                    # throw. Just ignore it and move on
                    continue
                estimated_preferences.append(estimated)

        # Return the error results.
        if metric in ['rmse', 'mae', 'nmae']:
            eval_function = evaluation_metrics[metric]
            if metric == 'nmae':
                return {metric: eval_function(real_preferences,
                                              estimated_preferences,
                                              recommender.model.maximum_preference_value(),
                                              recommender.model.minimum_preference_value())}
            return {metric: eval_function(real_preferences,
                                          estimated_preferences)}

        # IR_Statistics
        relevant_arrays = []
        real_arrays = []

        # Select the users to be evaluated.
        user_ids = recommender.model.user_ids()
        for user_id in user_ids[users_set]:
            preferences = recommender.model.preferences_from_user(user_id)
            preferences = list(preferences)
            if len(preferences) < 2 * at:
                # Really not enough prefs to meaningfully evaluate the user
                continue

            # List some most-preferred items that would count as most
            if not recommender.model.has_preference_values():
                preferences = [(preference, 1.0) for preference in preferences]

            preferences = sorted(preferences, key=lambda x: x[1], reverse=True)
            relevant_item_ids = [item_id for item_id, preference
                                 in preferences[:at]]

            if len(relevant_item_ids) == 0:
                continue

            training_set = {}
            for other_user_id in recommender.model.user_ids():
                preferences_other_user = \
                    recommender.model.preferences_from_user(other_user_id)

                if not recommender.model.has_preference_values():
                    preferences_other_user = [(preference, 1.0)
                                              for preference in preferences_other_user]
                if other_user_id == user_id:
                    preferences_other_user = \
                        [pref for pref in preferences_other_user \
                         if pref[0] not in relevant_item_ids]

                    if preferences_other_user:
                        training_set[other_user_id] = \
                            dict(preferences_other_user)
                else:
                    training_set[other_user_id] = dict(preferences_other_user)

            # Evaluate the recommender
            recommender_training = self._build_recommender(training_set, \
                                                           recommender)

            try:
                preferences = \
                    recommender_training.model.preferences_from_user(user_id)
                preferences = list(preferences)
                if not preferences:
                    continue
            except:
                # Excluded all prefs for the user. move on.
                continue

            recommended_items = recommender_training.recommend(user_id, at)
            relevant_arrays.append(list(relevant_item_ids))
            real_arrays.append(list(recommended_items))

        relevant_arrays = np.array(relevant_arrays)
        real_arrays = np.array(real_arrays)

        # Return the IR results.
        if metric in ['precision', 'recall', 'f1score']:
            eval_function = evaluation_metrics[metric]
            return {metric: eval_function(real_arrays, relevant_arrays)}

        if metric is None:
            # Return all
            mae, nmae, rmse = evaluation_error(real_preferences,
                                               estimated_preferences,
                                               recommender.model.maximum_preference_value(),
                                               recommender.model.minimum_preference_value())
            f = f1_score(real_arrays, relevant_arrays)
            r = recall_score(real_arrays, relevant_arrays)
            p = precision_score(real_arrays, relevant_arrays)

            return {'mae': mae, 'nmae': nmae, 'rmse': rmse,
                    'precision': p, 'recall': r, 'f1score': f}

    def evaluate_on_split(self, recommender, metric=None, cv=None, **kwargs):
        sampling_users = kwargs.pop('sampling_users', 0.7)
        permutation = kwargs.pop('permutation', True)
        at = kwargs.pop('at', 3)

        if metric not in evaluation_metrics and metric is not None:
            raise ValueError('metric %s is not recognized. valid keywords \
                      are %s' % (metric, evaluation_metrics.keys()))

        permutation_scores_error = []
        permutation_scores_ir = []
        final_score_error = {'avg': {}, 'stdev': {}}
        final_score_ir = {'avg': {}, 'stdev': {}}

        n_users = recommender.model.users_count()
        sampling_users = check_sampling(sampling_users, n_users)
        users_set, _ = sampling_users.split(permutation=permutation)

        total_ratings = []
        # Select the users to be evaluated.
        user_ids = recommender.model.user_ids()
        for user_id in user_ids[users_set]:
            # Select the ratings to be evaluated.
            preferences = recommender.model.preferences_from_user(user_id)
            preferences = list(preferences)
            total_ratings.extend([(user_id, preference)
                                  for preference in preferences])

        n_ratings = len(total_ratings)
        cross_val = check_cv(cv, n_ratings)
        # Defining the splits and run on the splits.
        for train_set, test_set in cross_val:

            training_set = {}
            testing_set = {}

            for idx in train_set:
                user_id, pref = total_ratings[idx]
                if recommender.model.has_preference_values():
                    training_set.setdefault(user_id, {})
                    training_set[user_id][pref[0]] = pref[1]
                else:
                    training_set.setdefault(user_id, {})
                    training_set[user_id][pref] = 1.0

            for idx in test_set:
                user_id, pref = total_ratings[idx]
                if recommender.model.has_preference_values():
                    testing_set.setdefault(user_id, [])
                    testing_set[user_id].append(pref)
                else:
                    testing_set.setdefault(user_id, [])
                    testing_set[user_id].append((pref, 1.0))

            # Evaluate the recommender.
            recommender_training = self._build_recommender(training_set, \
                                                           recommender)

            real_preferences = []
            estimated_preferences = []

            for user_id, preferences in testing_set.iteritems():
                for item_id, preference in preferences:
                    # Estimate the preferences
                    try:
                        estimated = recommender_training.estimate_preference(
                            user_id, item_id)
                        real_preferences.append(preference)
                    except:
                        # It is possible that an item exists
                        # in the test data but
                        # not training data in which case
                        # an exception will be
                        # throw. Just ignore it and move on
                        continue
                    estimated_preferences.append(estimated)

            # Return the error results.
            if metric in ['rmse', 'mae', 'nmae']:
                eval_function = evaluation_metrics[metric]
                if metric == 'nmae':
                    permutation_scores_error.append({
                        metric: eval_function(real_preferences,
                                              estimated_preferences,
                                              recommender.model.maximum_preference_value(),
                                              recommender.model.minimum_preference_value())})
                else:
                    permutation_scores_error.append(
                        {metric: eval_function(real_preferences,
                                               estimated_preferences)})
            elif metric is None:
                # Return all
                mae, nmae, rmse = evaluation_error(real_preferences,
                                                   estimated_preferences,
                                                   recommender.model.maximum_preference_value(),
                                                   recommender.model.minimum_preference_value())
                permutation_scores_error.append({'mae': mae, 'nmae': nmae,
                                                 'rmse': rmse})

        # IR_Statistics (Precision, Recall and F1-Score)
        n_users = recommender.model.users_count()
        cross_val = check_cv(cv, n_users)

        for train_idx, test_idx in cross_val:
            relevant_arrays = []
            real_arrays = []
            for user_id in user_ids[train_idx]:
                preferences = recommender.model.preferences_from_user(user_id)
                preferences = list(preferences)
                if len(preferences) < 2 * at:
                    # Really not enough prefs to meaningfully evaluate the user
                    continue

                # List some most-preferred items that would count as most
                if not recommender.model.has_preference_values():
                    preferences = [(preference, 1.0) for preference in preferences]

                preferences = sorted(preferences, key=lambda x: x[1], reverse=True)
                relevant_item_ids = [item_id for item_id, preference
                                     in preferences[:at]]

                if len(relevant_item_ids) == 0:
                    continue

                # Build the training set.
                training_set = {}
                for other_user_id in recommender.model.user_ids():
                    preferences_other_user = recommender.model.preferences_from_user(other_user_id)

                    if not recommender.model.has_preference_values():
                        preferences_other_user = [(preference, 1.0) for preference in preferences_other_user]
                    if other_user_id == user_id:
                        preferences_other_user = [pref for pref in preferences_other_user if
                                                  pref[0] not in relevant_item_ids]

                        if preferences_other_user:
                            training_set[other_user_id] = dict(preferences_other_user)
                    else:
                        training_set[other_user_id] = dict(preferences_other_user)

                # Evaluate the recommender
                recommender_training = self._build_recommender(training_set, recommender)

                try:
                    preferences = recommender_training.model.preferences_from_user(user_id)
                    preferences = list(preferences)
                    if not preferences:
                        continue
                except:
                    # Excluded all prefs for the user. move on.
                    continue

                recommended_items = recommender_training.recommend(user_id, at)
                relevant_arrays.append(list(relevant_item_ids))
                real_arrays.append(list(recommended_items))

            relevant_arrays = np.array(relevant_arrays)
            real_arrays = np.array(real_arrays)

            # Return the IR results.
            if metric in ['precision', 'recall', 'f1score']:
                eval_function = evaluation_metrics[metric]
                permutation_scores_ir.append({metric: eval_function(real_arrays, relevant_arrays)})
            elif metric is None:
                f = f1_score(real_arrays, relevant_arrays)
                r = recall_score(real_arrays, relevant_arrays)
                p = precision_score(real_arrays, relevant_arrays)
                permutation_scores_ir.append({'precision': p, 'recall': r, 'f1score': f})

        # Compute the final score for Error Statistics
        for result in permutation_scores_error:
            for key in result:
                final_score_error['avg'].setdefault(key, [])
                final_score_error['avg'][key].append(result[key])
        for key in final_score_error['avg']:
            final_score_error['stdev'][key] = np.std(final_score_error['avg'][key])
            final_score_error['avg'][key] = np.average(final_score_error['avg'][key])

        # Compute the final score for IR statistics
        for result in permutation_scores_ir:
            for key in result:
                final_score_ir['avg'].setdefault(key, [])
                final_score_ir['avg'][key].append(result[key])
        for key in final_score_ir['avg']:
            final_score_ir['stdev'][key] = np.std(final_score_ir['avg'][key])
            final_score_ir['avg'][key] = np.average(final_score_ir['avg'][key])

        permutation_scores = {}
        scores = {}
        if permutation_scores_error:
            permutation_scores['error'] = permutation_scores_error
            scores['final_error'] = final_score_error
        if permutation_scores_ir:
            permutation_scores['ir'] = permutation_scores_ir
            scores.setdefault('final_error', {})
            scores['final_error'].setdefault('avg', {})
            scores['final_error'].setdefault('stdev', {})
            scores['final_error']['avg'].update(final_score_ir['avg'])
            scores['final_error']['stdev'].update(final_score_ir['stdev'])

        return permutation_scores, scores
