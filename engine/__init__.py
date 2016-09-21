#!python2.7
# -*- coding: utf-8 -*-
"""
Created by kun on 2016/7/29.
"""

import logging

__author__ = 'kun'

try:
    from numpy.testing import nosetester


    class NoseTester(nosetester.NoseTester):
        u"""
        Subclass numpy's NoseTester to add doctests by default
        """

        def test(self, label='fast', verbose=1, extra_argv=["--exe"],
                 doctests=True, coverage=False, raise_warnings=None):
            return super(NoseTester, self).test(label=label, verbose=verbose, extra_argv=extra_argv, doctests=doctests,
                                                coverage=coverage, raise_warnings=raise_warnings)


    test = NoseTester().test
    del nosetester
except:
    pass

__all__ = ['datasets', 'metrics', 'similarities', 'models', 'recommenders']

__version__ = '0.1.git'


class NullHandler(logging.Handler):
    u"""
    For python versions <= 2.6; same as `logging.NullHandler` in 2.7.
    """

    def emit(self, record):
        pass


logger = logging.getLogger('py-recommender-framework')
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())
