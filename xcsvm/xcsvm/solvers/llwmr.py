import datetime
import json
import numpy as np
import os
import scipy
import sys
import time
import traceback

from ..utils import base as ubase

from .base import *
from . import cython as solvers_cython

__all__ = [
    "LLW_MR_Sparse_Solver",
]


###############################################################################


class WeightMapReduceMixin(object):

    def __init__(self, openmp_map_reduce=True, **kwargs):
        self.openmp_map_reduce = openmp_map_reduce
        super(WeightMapReduceMixin, self).__init__(**kwargs)

    def _setup(self, X, y):
        try:
            super(WeightMapReduceMixin, self)._setup(X, y)
        except AttributeError as e:
            pass

        self._do_openmp_map_reduce = (self.openmp_map_reduce and
                                      self._openmp_threads > 1)

        if self._do_openmp_map_reduce is True:
            self._mapreduce_tmp = self._zeros_with_cache_aligned_rows(
                "mapreduce_tmp",
                (self._openmp_threads, X.shape[1]), dtype=self.dtype)
        pass

    def _teardown(self):
        try:
            super(WeightMapReduceMixin, self)._teardown()
        except AttributeError as e:
            pass

        if self._do_openmp_map_reduce is True:
            del self._mapreduce_tmp
        del self._do_openmp_map_reduce
        pass

    def _mapreduce_weights(self):
        if self._do_openmp_map_reduce is not True:
            w_sum = self._W.sum(axis=0)
            w_sum = self._mpi.sum(w_sum)
            self._W -= w_sum/self._all_classes.shape[0]
        else:
            self._mapreduce_tmp[:, :] = 0
            w_sum = self._base_c.dist_sum_axis_0(
                self._W, self._mapreduce_tmp, self._openmp_threads)
            w_sum = self._mpi.sum(w_sum)
            self._base_c.dist_subtraction_axis_0(
                self._W, w_sum/self._all_classes.shape[0],
                self._openmp_threads)
        pass


class LLW_MR_Sparse_Solver(DefaultSetupMixin,
                           WeightMapReduceMixin,
                           FitWithUpdateMixin,
                           BaseXMCSolver):

    def __init__(self,
                 folds=1,
                 variant=0,
                 shrinking=0,
                 decrement_epsilon=False,
                 shrinked_start_state=-1,
                 shrink_state=1,
                 stop_shrinking=0,
                 **kwargs):
        self.folds = folds
        self.variant = variant
        self.shrinking = shrinking
        self.decrement_epsilon = decrement_epsilon
        self.shrinked_start_state = shrinked_start_state
        self.shrink_state = shrink_state
        self.stop_shrinking = stop_shrinking

        kwargs["remove_zero_K_samples"] = True
        super(LLW_MR_Sparse_Solver, self).__init__(**kwargs)
        pass

    def _init_sys(self):
        try:
            super(LLW_MR_Sparse_Solver, self)._init_sys()
        except AttributeError as e:
            pass

        self._llwmr_c = solvers_cython.get_module("llwmr",
                                                  self.dtype, self.idtype)
        pass

    def _primal_dual_gap(self):

        if self.primal_dual_gap_samples > 0:
            idx = np.random.choice(self._X.shape[0],
                                   size=self.primal_dual_gap_samples)
        else:
            idx = np.arange(self._X.shape[0])

        X = self._X[idx]
        alpha = self._alpha[:, idx]
        yi = self._yi_local[idx]
        r = np.arange(idx.size)
        local_idx = yi != self._all_classes.size

        margin = 1+X.dot(self._W.T)
        margin[local_idx, yi[local_idx]] = 0
        primal = (0.5 * (np.linalg.norm(self._W, axis=1)**2).sum() +
                  self.C * margin[margin > 0].sum())

        w_mean = self._W.mean(axis=1)
        tmp = w_mean-X.T.dot(alpha.T)
        dual = (-0.5 * (np.linalg.norm(tmp, axis=0)**2).sum() +
                alpha.sum() - alpha[yi[local_idx], local_idx].sum())

        return np.abs(primal-dual)

    def _fit(self):
        """
        shrinking:

        variant 0:
        - for each class we have a list of active samples
        - list of actives classes, i.e. has active samples

        variant 1:
        - for each samples list of active classes
        - list of active samples, i.e. has active classes
        """
        # this is a view on a cache aligned array
        class_optimal = self._zeros_with_cache_aligned_rows(
            None,
            (self._classes.size, 1), dtype=np.uint8)[:, 0]

        epsilon = [self.epsilon]
        last_violation = [-1000.0]
        data_shrink_state = np.zeros((1, 1), dtype=np.int8)
        shrinked_start_state = np.int8(0)
        shrink_state = np.int8(1)
        fresh_start_count = [0]
        if self.shrinking == 1:
            fresh_start = [True]
            if self.decrement_epsilon is True:
                epsilon = [min(1.0, 10 * self.epsilon)]

            data_shrink_state = np.zeros((self._classes.size,
                                          self._idx.shape[0]),
                                         dtype=np.int8)
            assert self.shrinked_start_state < 0
            if self.shrinked_start_state != -1:
                raise NotImplementedError("Not used at the moment.")
            shrinked_start_state = np.int8(self.shrinked_start_state)
            shrink_state = np.int8(self.shrink_state)

        def update_function(i, class_max_violation):
            class_optimal[:] = 1

            if self.shrinking == 1 and fresh_start[0] is True:
                fresh_start_count[0] += 1
                if self.shrinking == 1:
                    data_shrink_state[:, :] = 0

            shuffled = False
            if(self.shuffle is True or
               (self.shuffle > 0 and i % self.shuffle == 0)):
                np.random.shuffle(self._idx)
                shuffled = True

            X, yi, K = self._X, self._yi_local, self._K
            for s, e in ubase.folds(self.folds, self._X.shape[0]):
                idx = self._idx[s:e]
                args = [
                    X,
                    yi,
                    idx,
                    self._alpha,
                    K,
                    self._W,
                    self._classes,
                    class_optimal,
                    class_max_violation,
                    self.epsilon,
                    self.C,

                    np.uint32(self.shrinking),
                    data_shrink_state,
                    shrink_state,
                    shrinked_start_state,
                    last_violation[0],
                ]
                c = self._llwmr_c
                if self.variant == 0:
                    c.llw_mr_sparse_solver_updates__variant_0(*args)
                elif self.variant == 1:
                    c.llw_mr_sparse_solver_updates__variant_1(*args)
                else:
                    raise NotImplementedError()
                self._mapreduce_weights()

                # assert np.all(self._alpha >= 0)
                # assert np.all(self._alpha <= self.C)

            violation = class_max_violation.max()

            extra = ""
            if self.shrinking == 1:
                optimal = False

                active_count = (float((data_shrink_state >= 0).sum()) /
                                self._classes.size)
                extra += ", active %6g samples" % active_count

                if(np.isclose(violation, last_violation[0]) or
                   violation <= epsilon[0]):
                    if(violation <= self.epsilon and
                       epsilon[0] == self.epsilon):
                        if fresh_start[0] is True:
                            optimal = True
                        else:
                            extra += ", fresh_start"
                            fresh_start[0] = True
                    else:
                        extra += ", fresh_start"
                        fresh_start[0] = True
                        epsilon[0] = max(epsilon[0]*0.5, self.epsilon)
                else:
                    fresh_start[0] = False

                if(self.stop_shrinking != 0 and
                   self.stop_shrinking * self.epsilon > violation):
                    self._log.info("Stop shrinking at iterations %i." % i)
                    self.shrinking = 0
                    fresh_start[0] = True
            else:
                optimal = self._mpi.land(class_optimal.min())

            last_violation[0] = violation
            return optimal, extra

        ret = super(LLW_MR_Sparse_Solver, self)._fit(update_function)

        self._log.debug("Fresh start count: %i" % fresh_start_count[0])

        return ret
