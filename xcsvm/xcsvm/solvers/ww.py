import datetime
import json
import numpy as np
import os
import scipy
import sys
import time
import traceback

from .base import *
from . import cython as solvers_cython

__all__ = [
    "GroupSplitSetupMixin",

    "WW_Sparse_Solver",
]


###############################################################################


def get_match(c, r, C):
    even = C % 2 == 0
    if not even:
        C += 1

    ret = c
    if c == C-1:
        ret = r
    elif c == r:
        ret = C-1
    else:
        ret = (2*(C-1)+2*r-c) % (C-1)

    if not even and ret == C-1:
        ret = c

    return ret


###############################################################################


class GroupSplitSetupMixin(DefaultSetupMixin):
    """
    This extension of the DefaultSetupMixin groups the
    local classes into different groups. The group sizes
    and how they are composed can be changed with the parameters.

    In addition the data gets preprocessed in order to make it
    more accessible when working with groups.
    """

    def __init__(self, group_count=-1, grouping_criteria=None, folds=1,
                 grouping_shuffle_sizes=False,
                 force_global_grouping=False,
                 reduce_mem_allocation=False,
                 **kwargs):
        self.group_count = int(group_count)
        self.grouping_criteria = {"samples": 0,
                                  "classes": 1,
                                  "samples_x_classes": 2,
                                  None: 1}[grouping_criteria]
        self.grouping_shuffle_sizes = grouping_shuffle_sizes
        self.folds = folds
        self.force_global_grouping = force_global_grouping
        self.reduce_mem_allocation = reduce_mem_allocation

        if ("cache_alignment" in kwargs and
            kwargs["cache_alignment"] is True):
            raise Exception("Cache alignment is compatible "
                            "with GroupSplitSetupMixin")
        kwargs["cache_alignment"] = False
        super(GroupSplitSetupMixin, self).__init__(**kwargs)

    def _W_sparsity(self):
        if not(len(self._W.shape) > 2 and self._global_grouping is True):
            return super(GroupSplitSetupMixin, self)._W_sparsity()
        raise NotImplementedError("Needs to be updated.")
        active_W_range = (self.idtype(self._local_block_count-1) *
                          self._W_block_size)
        n_nz = (self._W[:active_W_range] != 0).sum()
        n_nz = self._mpi.sum(n_nz)
        n_entries = self._dimensions * len(self._all_classes)
        return float(n_nz)/n_entries

    def _setup_class_distribution(self, X, y):
        # all done in _setup_data_distribution
        pass

    def _define_groups(self):
        # each group consists of several classes
        self.group_count = min(self.group_count, self._all_classes.size)
        if self.group_count < 0:
            self.group_count = int(np.ceil(self._all_classes.size /
                                           (-1.0 * self.group_count)))

        if not (self.group_count > 0 and
                self.group_count <= self._all_classes.size):
            raise Exception("The group count needs to lie between "
                            " %i < x <= %i. %i does not." %
                            (0, self.group_count, self._all_classes.size))

        ret = [[[], 0, 0]  # group members, sample count, class count
               for i in xrange(self.group_count)]

        idx = np.arange(self._all_classes.size)
        if self.grouping_shuffle_sizes is True:
            np.random.shuffle(idx)
        for ci, c_size in zip(idx, self._all_class_sizes[idx]):
            min_i, min_val = -1, None
            for i, group in enumerate(ret):
                if self.grouping_criteria == 0:
                    val = group[1]+c_size
                if self.grouping_criteria == 1:
                    val = group[2]+1
                if self.grouping_criteria == 2:
                    val = (group[1]+c_size) * (group[2]+1)

                if min_val is None or min_val > val:
                    min_i, min_val = i, val

            ret[min_i][0].append(ci)
            ret[min_i][1] += c_size
            ret[min_i][2] += 1

        stats = np.asarray([(g[1], g[2], g[1]*g[2])
                            for g in ret])
        self._log.info("Use %i groups." % self.group_count)
        self._log.debug("Data group stats: sample count max %i "
                        "- mean %.1f - min %.1f"
                        % (np.max(stats[:, 0]),
                           np.mean(stats[:, 0]),
                           np.min(stats[:, 0])))
        self._log.debug("Data group stats: class count max %i "
                        "- mean %.1f - min %.1f"
                        % (np.max(stats[:, 1]),
                           np.mean(stats[:, 1]),
                           np.min(stats[:, 1])))
        self._log.debug("Data group stats: sample x class max %i "
                        "- mean %.1f - min %.1f"
                        % (np.max(stats[:, 2]),
                           np.mean(stats[:, 2]),
                           np.min(stats[:, 2])))

        return [g[0] for g in ret]

    def _setup_data_distribution(self, X, y):
        super(GroupSplitSetupMixin, self)._setup_data_distribution(X, y)

        self._all_class_sizes = np.asarray([(y == c).sum()
                                            for c in self._all_classes])
        idx = np.argsort(self._all_class_sizes)[::-1]
        self._all_classes = self._all_classes[idx]
        self._all_class_sizes = self._all_class_sizes[idx]

        groups = self._define_groups()

        permutation = np.arange(self._X.shape[0], dtype=self.idtype)
        self._group_members = []
        self._group_ranges = []
        self._group_data_ranges = []
        self._folded_group_data_ranges = [[] for i in xrange(self.folds)]

        i = 0
        data_i = 0
        mask = np.zeros((self._X.shape[0],), dtype=np.uint8)
        for g in groups:
            mask[:] = 0
            for ci in g:
                self._group_members.append(ci)
                c = self._all_classes[ci]
                mask = np.logical_or(mask, self._y == c)

            next_i = i+len(g)
            self._group_ranges.append((i, next_i))
            i = next_i

            idx = self._idx[mask]
            x = (data_i, data_i+idx.shape[0])
            permutation[x[0]:x[1]] = idx
            self._group_data_ranges.append(x)
            data_i = x[1]

            size = x[1]-x[0]
            fold_size = int(size / self.folds)
            missing = size % self.folds
            for fold_i in xrange(self.folds):
                self._folded_group_data_ranges[fold_i].append(
                    (x[0]+fold_i * fold_size + min(missing, fold_i),
                     x[0]+(fold_i+1) * fold_size + min(missing, fold_i+1))
                )

        # faster access with ndarray
        self._group_members = np.asarray(self._group_members,
                                         dtype=self.idtype)
        self._group_ranges = np.asarray(self._group_ranges, dtype=self.idtype)
        self._group_data_ranges = np.asarray(self._group_data_ranges,
                                             dtype=self.idtype)
        self._folded_group_data_ranges = np.asarray(
            self._folded_group_data_ranges, dtype=self.idtype)

        # reorder classes for faster access of W
        self._all_classes = self._all_classes[self._group_members]
        self._all_class_sizes = self._all_class_sizes[self._group_members]

        self._X = self._X[permutation]
        self._y = self._y[permutation]
        self._K = self._K[permutation]
        self._idx = np.arange(X.shape[0], dtype=self.idtype)

        self._group_count = self._group_ranges.shape[0]
        self._group_sizes = self._group_ranges[:, 1]-self._group_ranges[:, 0]
        self._max_group_size = self._group_sizes.max()

        self._yi_to_gx = np.zeros((self._classes.size, 2), dtype=self.idtype)
        self._gx_to_yi = np.zeros((self._group_count,
                                   self._max_group_size),
                                  dtype=self.idtype)
        for i, (a, b) in enumerate(self._group_ranges):
            for j in range(a, b):
                i, j = self.idtype(i), self.idtype(j)
                self._yi_to_gx[j] = [i, j-a]
                self._gx_to_yi[i, j-a] = ci

        self._global_grouping = self._mpi_size > 1 or self.force_global_grouping is True
        if self._global_grouping is True:
            self._global_group_count = 2 * self._mpi_size
            ggc = self._global_group_count
            self._global_group_sizes = np.ones((ggc,), dtype=self.idtype)
            self._global_group_sizes *= self.idtype(np.floor(self._group_count /
                                                             float(ggc)))
            missing = self.idtype(self._group_count-self._global_group_sizes.sum())
            self._global_group_sizes[:missing] += 1
            self._max_global_group_size = self._global_group_sizes.max()
            assert self._global_group_sizes.sum() == self._group_count
            self._log.debug("Groups per global group: min %.1f - %.1f - %.1f max"
                            % (self._global_group_sizes.min(),
                               self._global_group_sizes.mean(),
                               self._global_group_sizes.max()))
            if self._global_group_sizes.min() == 0:
                raise Exception("Some MPI nodes have no groups to handle. "
                                "This is expected. Please adjust your "
                                "group count.")

            self._W_block_size = self._max_global_group_size

            self._global_groups = np.zeros((ggc,
                                            self._max_global_group_size),
                                           dtype=self.idtype)
            tmp_range = np.arange(self._group_count, dtype=self.idtype)
            for i in xrange(ggc):
                s = self._global_group_sizes[i]
                self._global_groups[i, :s] = tmp_range[i::ggc]

            self._local_global_groups = [2*self._mpi_rank,
                                         2*self._mpi_rank+1]

            self._local_block_count = self.idtype(len(self._local_global_groups)+1)
            lbc = self._local_block_count

            self._group_mapping = np.zeros(self._group_count,
                                           dtype=self.idtype)
            for i, (s, gg)  in enumerate(zip(self._global_group_sizes,
                                           self._global_groups)):
                for j in xrange(s):
                    if i in self._local_global_groups:
                        # map to first/second block
                        self._group_mapping[gg[j]] = (i%2)*self._W_block_size+j
                    else:
                        # all remotes are mapped to last block
                        self._group_mapping[gg[j]] = (lbc-1)*self._W_block_size+j

            self._W_train_shape = (self.idtype(lbc) * self._W_block_size,
                                   self._max_group_size,
                                   self._dimensions)

            class_count = self.idtype(0)
            for ggi in self._local_global_groups:
                for gi in self._global_groups[ggi][:self._global_group_sizes[ggi]]:
                    class_count += self._group_ranges[gi, 1]-self._group_ranges[gi, 0]
            self._classes = np.empty((class_count,), dtype=self.dtype)
            self._class_sizes = np.empty((class_count,), dtype=self.idtype)

            i = self.idtype(0)
            for ggi in self._local_global_groups:
                for g in self._global_groups[ggi][:self._global_group_sizes[ggi]]:
                    s = self._group_ranges[g, 1]-self._group_ranges[g, 0]
                    self._classes[i:i+s] = self._all_classes[self._group_ranges[g, 0]:
                                                             self._group_ranges[g, 1]]
                    self._class_sizes[i:i+s] = self._all_class_sizes[self._group_ranges[g, 0]:
                                                                     self._group_ranges[g, 1]]
                    i += s
            assert i == self._classes.size

            n = self.idtype(self._n_samples)
            ggr_masks = np.zeros((self._global_group_count, n), dtype=np.uint8)
            for i in xrange(self._global_group_count):
                for g in self._global_groups[i][:self._global_group_sizes[i]]:
                    for c in range(self._group_ranges[g, 0], self._group_ranges[g, 1]):
                        ggr_masks[i][:] += self._y == self._all_classes[c]
            assert ggr_masks.sum() == self._n_samples
            self._ggr_masks = ggr_masks

            ggr_active_masks = np.zeros(self._ggr_masks.shape, dtype=np.uint8)

            mpi_rounds = self._mpi_size-1
            if self._mpi_size % 2 == 1:
                mpi_rounds += 1
            mpir = self._mpi_rank

            ggr_la, ggr_lb = 2*mpir, 2*mpir+1

            ggr_active_masks[ggr_la] += ggr_masks[ggr_la]
            ggr_active_masks[ggr_lb] += ggr_masks[ggr_lb]

            ggr_active_masks[ggr_la] += ggr_masks[ggr_lb]
            ggr_active_masks[ggr_lb] += ggr_masks[ggr_la]

            for mpi_round in xrange(mpi_rounds):
                mpim = get_match(mpir, mpi_round, self._mpi_size)
                if mpim == mpir:
                    continue

                ggr_ra, ggr_rb = 2*mpim, 2*mpim+1

                if mpir < mpim:
                    ggr_active_masks[ggr_la] += ggr_masks[ggr_ra]
                    ggr_active_masks[ggr_ra] += ggr_masks[ggr_la]

                    ggr_active_masks[ggr_la] += ggr_masks[ggr_rb]
                    ggr_active_masks[ggr_rb] += ggr_masks[ggr_la]
                else:
                    ggr_active_masks[ggr_la] += ggr_masks[ggr_rb]
                    ggr_active_masks[ggr_rb] += ggr_masks[ggr_la]

                    ggr_active_masks[ggr_lb] += ggr_masks[ggr_rb]
                    ggr_active_masks[ggr_rb] += ggr_masks[ggr_lb]

            for i in range(self._global_group_count):
                assert self._mpi.sum(ggr_active_masks[i]).sum() == self._n_samples
            self._ggr_active_masks = ggr_active_masks

            # todo: use maxint
            if self.reduce_mem_allocation is True:
                self._samples_per_ggr = ggr_active_masks.sum(axis=1,
                                                             dtype=self.idtype)
                self._ggr_to_samples = np.zeros(self._ggr_masks.shape,
                                                self.idtype) + 10**7

                for i in xrange(self._global_group_count):
                    tmp = np.arange(self._samples_per_ggr[i], dtype=self.idtype)
                    self._ggr_to_samples[i][ggr_active_masks[i].astype(np.bool)] = tmp

                self._gr_to_alpha = np.zeros(self._group_count,
                                             dtype=self.idtype)
                tmp = self.idtype(0)
                tmpi = 0
                for i in xrange(self._global_group_count):
                    for g in self._global_groups[i][:self._global_group_sizes[i]]:
                        self._gr_to_alpha[tmpi] = tmp
                        tmp += self._samples_per_ggr[i]*self._max_group_size
                        tmpi += 1

                self._alpha_shape = (tmp,)
                assert self._mpi.sum(tmp) == (self._group_count*self._max_group_size*n)
            else:
                tmp = np.arange(n, dtype=self.idtype)
                self._ggr_to_samples = np.broadcast_to(tmp,
                                                       (self._global_group_count, n))
                self._samples_per_gr = np.zeros(self._group_count,
                                                dtype=self.idtype) + n
                self._gr_to_alpha = np.arange(self._group_count,
                                              dtype=self.idtype)
                self._gr_to_alpha *= n * self._max_group_size
                self._alpha_shape = (self.idtype(self._group_count *
                                                 self._max_group_size *
                                                 self._n_samples),)
        else:
            self._global_group_count = 1
            self._global_group_sizes = np.ones((1,), dtype=self.idtype)
            self._global_group_sizes[:] = self._group_count
            self._W_block_size = 0
            self._global_groups = np.arange(self._group_count,
                                            dtype=self.idtype).reshape((1, -1))
            self._local_global_groups = [0]

            self._group_mapping = np.arange(self._group_count,
                                            dtype=self.idtype)

            self._W_train_shape = (self._group_count,
                                   self._max_group_size,
                                   self._dimensions)

            self._classes = self._all_classes.copy()
            self._class_sizes = self._all_class_sizes.copy()

            n = self.idtype(self._n_samples)
            tmp = np.arange(n, dtype=self.idtype)
            self._ggr_to_samples = tmp.reshape((1, -1))
            self._samples_per_ggr = np.zeros(self._global_group_count,
                                             dtype=self.idtype) + n
            self._gr_to_alpha = np.arange(self._group_count,
                                          dtype=self.idtype)
            self._gr_to_alpha *= n * self._max_group_size
            self._alpha_shape = (self.idtype(self._group_count *
                                             self._max_group_size *
                                             self._n_samples),)

        # we reordered the classes
        for i, c in enumerate(self._all_classes):
            self._yi[self._y == c] = i
        self._yi_local[:] = self._all_classes.size
        for i, c in enumerate(self._classes):
            self._yi_local[self._y == c] = i
        pass

    def _teardown_data_distribution(self):
        super(GroupSplitSetupMixin, self)._teardown_data_distribution()

        del self._group_members
        del self._group_ranges
        del self._group_sizes
        del self._group_count
        del self._max_group_size
        del self._group_mapping

        del self._yi_to_gx
        del self._gx_to_yi

        #todo delete all variables

        del self._group_data_ranges
        del self._folded_group_data_ranges
        pass

    def _setup_train_model(self):
        super(GroupSplitSetupMixin, self)._setup_train_model()
        pass

    def _teardown_train_model(self):
        super(GroupSplitSetupMixin, self)._teardown_train_model()
        W = np.empty((self._classes.size, self._W.shape[2]), dtype=self.dtype)
        W_old = self._W

        i = self.idtype(0)
        for Wbi, ggi in enumerate(self._local_global_groups):
            for gi, g in enumerate(self._global_groups[ggi][:self._global_group_sizes[ggi]]):
                g_size = self._group_ranges[g, 1]-self._group_ranges[g, 0]
                idx = self.idtype(Wbi*self._W_block_size+gi)
                W[i:i+g_size] = self._W[idx, 0:g_size]
                i += g_size

        self._W = W
        del W_old
        pass


###############################################################################


class WW_Sparse_Solver(GroupSplitSetupMixin,
                       FitWithUpdateMixin,
                       BaseXMCSolver):

    def __init__(self,
                 variant=1,
                 shuffle_rounds=True,
                 shrinking=0,
                 decrement_epsilon=False,
                 shrinked_start_state=-1,
                 shrink_state=1,
                 stop_shrinking=0,
                 shrinking_max_iter=10,
                 mpi_data_distributed=False,
                 mpi_send_sparse=True,
                 mpi_send_changes_back=True,
                 mpi_cache_sparsity=True,
                 mpi_cache_sparsity_check=False,
                 mpi_local_folds=True,
                 mpi_com_switch=True,
                 mpi_fast_sparsify=True,
                 mpi_join_communication=False,
                 mpi_join_communication_start_size=10,
                 mpi_join_communication_single_link=False,
                 inner_repeat=1,
                 **kwargs):
        self.variant = variant
        self.shuffle_rounds = shuffle_rounds
        self.shrinking = shrinking
        self.decrement_epsilon = decrement_epsilon
        self.shrinked_start_state = shrinked_start_state
        self.shrink_state = shrink_state
        self.stop_shrinking = stop_shrinking
        self.shrinking_max_iter = shrinking_max_iter

        self.mpi_data_distributed = mpi_data_distributed
        self.mpi_send_sparse = mpi_send_sparse
        self.mpi_send_changes_back = mpi_send_changes_back
        self.mpi_cache_sparsity = mpi_cache_sparsity
        self.mpi_cache_sparsity_check = mpi_cache_sparsity_check
        self.mpi_local_folds = mpi_local_folds
        self.mpi_com_switch = mpi_com_switch
        self.mpi_fast_sparsify = mpi_fast_sparsify
        self.mpi_join_communication = mpi_join_communication
        self.mpi_join_communication_start_size = mpi_join_communication_start_size
        self.mpi_join_communication_single_link = mpi_join_communication_single_link

        if(self.mpi_cache_sparsity is True and
           self.mpi_send_changes_back is False):
            raise Exception("Cache sparsity relies on send changes back."
                            " Please enable it too.")

        if self.mpi_com_switch is True and self.mpi_data_distributed is True:
            raise NotImplementedError()

        self.inner_repeat = inner_repeat

        kwargs["remove_zero_K_samples"] = True
        super(WW_Sparse_Solver, self).__init__(**kwargs)
        pass

    def _init_sys(self):
        super(WW_Sparse_Solver, self)._init_sys()

        self._ww_c = solvers_cython.get_module("ww",
                                               self.dtype, self.idtype)
        pass

    def _primal_dual_gap(self):
        if self._mpi_size > 1:
            raise NotImplementedError("Primal dual gap is not "
                                      "supported with mpi.")

        if self.group_padding is True:
            raise NotImplementedError("Primal dual gap is not "
                                      "supported with group padding.")

        if self.primal_dual_gap_samples > 0:
            idx = np.random.choice(self._X.shape[0],
                                   size=self.primal_dual_gap_samples)
        else:
            idx = np.arange(self._X.shape[0])

        X = self._X[idx]
        alpha = self._alpha[idx]
        yi = self._yi_local[idx]
        r = np.arange(idx.size)

        margin = X.dot(self._W)
        margin = 1 - margin + margin[r, yi][:, None]
        margin[r, yi] = 0
        primal = (0.5 * (np.linalg.norm(self._W, axis=0)**2).sum() +
                  self.C * margin[margin > 0].sum())

        tmp = -X.T.dot(alpha)
        dual = (-0.5 * (np.linalg.norm(tmp, axis=1)**2).sum() +
                alpha.sum() - alpha[r, yi].sum())

        return np.abs(primal-dual)

    def _setup_train_model(self):
        super(WW_Sparse_Solver, self)._setup_train_model()

        def transpose_on_existing_data(x):
            shape = list(x.shape)
            shape[-2], shape[-1] = shape[-1], shape[-2]
            return np.frombuffer(x, x.dtype).reshape(shape)

        self._W = transpose_on_existing_data(self._W)
        pass

    def _teardown_train_model(self):
        self._W = np.swapaxes(self._W, -2, -1)

        super(WW_Sparse_Solver, self)._teardown_train_model()
        pass

    _mpi__comm_history__ = None
    _mpisendrecv__cache = None
    def _mpisendrecv(
            self,
            mpi_rank,
            send_nda, recv_nda,
            in_place=False,
            send_only_changes=False,
            recv_only_changes=False,
            send_changed_mask=None,
            remote_idx=None,
            keep_remote_idx=None,
            send_cache=None,
            recv_cache=None,
    ):
        if self._mpisendrecv__cache is None:
            self._mpisendrecv__cache = {}
        if self._mpi__comm_history__ is None:
            self._mpi__comm_history__ = [0, send_nda.size+1, np.ones(25)*1024]

        def issorted(s, a):
            if s == 0:
                return True
            x = a[:s]
            ret = np.all((x[1:]-x[:-1]) > 0) and (np.unique(x).size == s)
            if not ret:
                print "issorted", s, x
            return ret

        if self.mpi_send_sparse is True:
            key, size = send_nda.size, send_nda.size+1
            if key not in self._mpisendrecv__cache:
                if self.mpi_fast_sparsify is False:
                    send_sparse_mask = np.empty((size,), dtype=np.bool)
                    idx_range = np.arange(size, dtype=self.idtype)
                else:
                    send_sparse_mask, idx_range = None, None
                send_sparse_idx = np.empty((size,), dtype=self.idtype)
                send_sparse_data = np.empty((size,), dtype=self.dtype)
                send_sparse_idx2 = np.empty((size,), dtype=self.idtype)
                send_sparse_data2 = np.empty((size,), dtype=self.dtype)
                recv_sparse_idx = np.empty((size,), dtype=self.idtype)
                recv_sparse_data = np.empty((size,), dtype=self.dtype)
                cache = (
                    send_sparse_idx, send_sparse_data,
                    send_sparse_idx2, send_sparse_data2,
                    recv_sparse_idx, recv_sparse_data,
                    send_sparse_mask, idx_range,
                )
                self._mpisendrecv__cache[key] = cache
            else:
                cache = self._mpisendrecv__cache[key]
                send_sparse_idx, send_sparse_data = cache[0:2]
                send_sparse_idx2, send_sparse_data2 = cache[2:4]
                recv_sparse_idx, recv_sparse_data = cache[4:6]


            if self.mpi_cache_sparsity_check is True:
                tmp = [send_sparse_idx, send_sparse_data]
                def cache_sparsity_check(s, nda, nnz, ssi, ssd):
                    s = "[rank %i] %s" % (self._mpi_rank, s)
                    tmp_ssi, tmp_ssd = tmp
                    tmp_nnz = self._base_c.sparsify(nda,
                                                    tmp_ssi,
                                                    tmp_ssd)

                    u_size = np.unique(ssi[:nnz]).size
                    assert nnz == u_size, (s, nnz, u_size)
                    assert (ssd[:nnz] == 0).sum() == 0

                    if tmp_nnz != nnz:
                        tmp_ssi, tmp_ssd = tmp_ssi[:tmp_nnz], tmp_ssd[:tmp_nnz]
                        ssi, ssd = ssi[:nnz], ssd[:nnz]
                        #print tmp_ssi[:100], ssi[:100]
                        diff = np.in1d(ssi, tmp_ssi, invert=True)
                        print s, diff.size, tmp_nnz, nnz, (nda != 0).sum()
                        print s, (nda[tmp_ssi] == 0).sum(), (nda[ssi] == 0).sum()
                        print s, ssi[(nda[ssi] == 0)], nda[ssi[(nda[ssi] == 0)]]
                    assert tmp_nnz == nnz, (s, tmp_nnz, nnz)
                    tmp_ssi, tmp_ssd = tmp_ssi[:tmp_nnz], tmp_ssd[:tmp_nnz]
                    ssi, ssd = ssi[:nnz], ssd[:nnz]

                    condition = np.allclose(tmp_ssi, ssi)
                    if not condition:
                        print nnz, tmp_ssi[:100], ssi[:100]
                    assert condition, "%s_sparse_idx" % s
                    condition = np.allclose(tmp_ssd, ssd)
                    if not condition:
                        print nnz, tmp_ssd[:100], ssd[:100], nda
                    assert condition, "%s_sparse_data" % s

            send_nda = send_nda.ravel()
            if send_only_changes is False and send_cache is None:
                start_time = time.time()
                if self.mpi_fast_sparsify is False:
                    send_sparse_mask, idx_range = cache[6:8]
                    np.not_equal(send_nda, 0, send_sparse_mask)
                    nnz = send_sparse_mask.sum()
                    send_sparse_idx[:nnz] = idx_range[send_sparse_mask]
                    send_sparse_data[:nnz] = send_nda[send_sparse_idx[:nnz]]
                else:
                    nnz = self._base_c.sparsify(send_nda,
                                                send_sparse_idx,
                                                send_sparse_data)
                self._proc_time1 += time.time() - start_time
            else:
                start_time = time.time()
                if self.mpi_fast_sparsify is False:
                    scm_shape = send_changed_mask.shape
                    send_changed_mask = send_changed_mask.reshape(list(scm_shape)+[1,])
                    send_changed_mask = np.broadcast_to(
                        send_changed_mask,
                        list(scm_shape)+[self._max_group_size,])
                    send_sparse_mask, idx_range = cache[6:8]
                    idx_range = idx_range[send_changed_mask.ravel()]
                    send_sparse_mask = send_sparse_mask[:idx_range.size]
                    np.not_equal(send_nda[idx_range], 0, send_sparse_mask)
                    nnz = send_sparse_mask.sum()
                    send_sparse_idx[:nnz] = idx_range[send_sparse_mask]
                    send_sparse_data[:nnz] = send_nda[send_sparse_idx[:nnz]]
                else:
                    nnz = self._base_c.sparsify_with_changes(send_nda,
                                                             send_changed_mask.view(np.int8),
                                                             send_sparse_idx,
                                                             send_sparse_data,
                                                             self._max_group_size)
                self._proc_time1 += time.time() - start_time
                start_time = time.time()

                if send_cache is not None:
                    send_cache = send_cache()

                got_zero = None
                ref_idx = None
                if remote_idx is not None:
                    ref_idx = remote_idx
                if send_cache is not None:
                    ref_idx = [send_cache[0][0], send_cache[1]]
                if ref_idx is not None:
                    if self.mpi_fast_sparsify is False:
                        maybe_zero = np.in1d(ref_idx[1][:ref_idx[0]],
                                             send_sparse_idx[:nnz],
                                             assume_unique=True, invert=True)
                        maybe_zero = ref_idx[1][:ref_idx[0]][maybe_zero]
                        got_zero = maybe_zero[send_nda[maybe_zero] == 0]
                    else:
                        got_zero_n = self._base_c.got_zero(send_nda,
                                                           nnz, send_sparse_idx,
                                                           ref_idx[0], ref_idx[1],
                                                           recv_sparse_idx)
                        got_zero = recv_sparse_idx[:got_zero_n]

                self._proc_time2 += time.time() - start_time
                start_time = time.time()
                if send_cache is not None:

                    #ssi = send_sparse_idx
                    #u_size = np.unique(ssi[:nnz]).size
                    #assert nnz == u_size, ("foo", nnz, u_size)
                    #ssi = send_cache[1]
                    #u_size = np.unique(ssi[:send_cache[0][0]]).size
                    #assert send_cache[0][0] == u_size, ("foo", send_cache[0][0], u_size)
                    if nnz > 0:
                        nnz = self._base_c.merge_sparse(send_cache[0][0],
                                                        send_cache[1], send_cache[2],
                                                        nnz,
                                                        send_sparse_idx, send_sparse_data,
                                                        send_cache[3], send_cache[4],)
                        send_cache[5]()
                        send_sparse_idx = send_cache[3]
                        send_sparse_data = send_cache[4]
                        send_cache[0][0] = nnz
                    else:
                        nnz = send_cache[0][0]
                        send_sparse_idx = send_cache[1]
                        send_sparse_data = send_cache[2]

                    #ssi = send_sparse_idx
                    #u_size = np.unique(ssi[:nnz]).size
                    #assert nnz == u_size, ("foo2", nnz, u_size)

                    # remove zeros from sparse representation
                    if got_zero is not None and got_zero.size > 0:
                        if self.mpi_fast_sparsify is False:
                            new_nnz = nnz - got_zero.size
                            idx = np.in1d(send_sparse_idx[:nnz], got_zero,
                                          assume_unique=True, invert=True)

                            send_sparse_idx[:new_nnz] = send_sparse_idx[:nnz][idx]
                            send_sparse_data[:new_nnz] = send_sparse_data[:nnz][idx]
                            nnz = new_nnz
                        else:
                            nnz = self._base_c.filter_zeros_sparse2(nnz,
                                                                    send_sparse_idx,
                                                                    send_sparse_data,
                                                                    got_zero)
                        send_cache[0][0] = nnz

                    #ssi = send_sparse_idx
                    #u_size = np.unique(ssi[:nnz]).size
                    #assert nnz == u_size, ("foo2", nnz, u_size)

                    if self.mpi_cache_sparsity_check is True:
                        if nnz > 0:
                            cache_sparsity_check("send", send_nda, nnz,
                                                 send_sparse_idx, send_sparse_data)

                self._proc_time3 += time.time() - start_time

                if got_zero is not None and got_zero.size > 0 and send_cache is not None:
                    if self.mpi_fast_sparsify is False:
                        # update zeros at remote peer
                        new_nnz = nnz+got_zero.size
                        send_sparse_idx[nnz:new_nnz] = got_zero
                        send_sparse_data[nnz:new_nnz] = 0

                        tmp_ssi, tmp_ssd = send_sparse_idx, send_sparse_data

                        sort = send_sparse_idx[:new_nnz].argsort()
                        # need to copy, use old cache arrays
                        send_sparse_idx, send_sparse_data = cache[2:4]
                        send_sparse_idx[:new_nnz] = tmp_ssi[:new_nnz][sort]
                        send_sparse_data[:new_nnz] = tmp_ssd[:new_nnz][sort]
                        nnz = new_nnz
                    else:
                        old_nnz = nnz
                        nnz = self._base_c.merge_sparse2(nnz,
                                                         send_sparse_idx,
                                                         send_sparse_data,
                                                         got_zero,
                                                         cache[2], cache[3])
                        send_sparse_idx = cache[2]
                        send_sparse_data = cache[3]

                    #print np.unique(send_sparse_idx[:nnz]).size, nnz
                    #print got_zero, send_sparse_idx[:nnz][send_sparse_data[:nnz] == 0], send_sparse_data[:nnz] == 0
                    #assert np.unique(send_sparse_idx[:nnz]).size == nnz

            if self.mpi_join_communication is False:
                send_sparse_size = np.asarray(nnz, dtype=self.idtype)
                recv_sparse_size = np.asarray(self.idtype(0), dtype=self.idtype)

                start_time = time.time()
                self._mpi.Sendrecv(send_sparse_size, mpi_rank,
                                   recvbuf=recv_sparse_size, source=mpi_rank)
                self._network_time1 += time.time()-start_time

                start_time = time.time()

                if keep_remote_idx is not None:
                    keep_remote_idx[0] = recv_sparse_size
                    recv_sparse_idx = keep_remote_idx[1]

                recv_sparse_idx = recv_sparse_idx[:recv_sparse_size]
                recv_sparse_data = recv_sparse_data[:recv_sparse_size]

                send_sparse_idx_buf = np.getbuffer(send_sparse_idx, 0, nnz*self.idtype_size)
                send_sparse_data_buf = np.getbuffer(send_sparse_data, 0, nnz*self.dtype_size)
                recv_sparse_idx_buf = np.getbuffer(recv_sparse_idx, 0, self.idtype(recv_sparse_size*self.idtype_size))
                recv_sparse_data_buf = np.getbuffer(recv_sparse_data, 0, self.idtype(recv_sparse_size*self.dtype_size))

                self._proc_time4 += time.time() - start_time
                if send_sparse_size > 0 or recv_sparse_size > 0:
                    start_time = time.time()
                    self._mpi.Sendrecv(send_sparse_idx_buf, mpi_rank,
                                       recvbuf=recv_sparse_idx_buf, source=mpi_rank)
                    self._network_time2 += time.time()-start_time
                    start_time = time.time()
                    self._mpi.Sendrecv(send_sparse_data_buf, mpi_rank,
                                       recvbuf=recv_sparse_data_buf, source=mpi_rank)
                    self._network_time3 += time.time()-start_time

                self._network_traffic += (send_sparse_size + recv_sparse_size)
            else:
                raise NotImplementedError("This implementation does not work.")
                history = self._mpi__comm_history__
                #print history
                if False:
                    start_size = max(self.idtype(history[2].max()*1.1), history[1])
                else:
                    start_size = 1024*self.mpi_join_communication_start_size
                start_size = min(send_sparse_idx.size-1, start_size-1)

                tmp_x = send_sparse_idx[start_size]
                send_sparse_idx[start_size] = nnz
                send_sparse_size = nnz

                # send first part
                ssi_l, ssd_l = (start_size+1)*self.idtype_size, (start_size+1)*self.dtype_size
                if self.mpi_join_communication_single_link is False:
                    send_sparse_idx_buf = np.getbuffer(send_sparse_idx, 0, ssi_l)
                    send_sparse_data_buf = np.getbuffer(send_sparse_data, 0, ssd_l)
                    recv_sparse_idx_buf = np.getbuffer(recv_sparse_idx, 0, ssi_l)
                    recv_sparse_data_buf = np.getbuffer(recv_sparse_data, 0, ssd_l)

                    start_time = time.time()
                    self._mpi.Sendrecv(send_sparse_idx_buf, mpi_rank,
                                       recvbuf=recv_sparse_idx_buf, source=mpi_rank)
                    self._mpi.Sendrecv(send_sparse_data_buf, mpi_rank,
                                       recvbuf=recv_sparse_data_buf, source=mpi_rank)
                    self._network_time2 += time.time()-start_time

                    recv_sparse_size = recv_sparse_idx[start_size]
                else:
                    send_sparse_idx_buf = np.getbuffer(send_sparse_idx, 0, ssi_l+ssd_l)
                    send_sparse_data_buf = np.getbuffer(send_sparse_data, 0, ssd_l)
                    recv_sparse_idx_buf = np.getbuffer(recv_sparse_idx, 0, ssi_l+ssd_l)
                    recv_sparse_data_buf = np.getbuffer(recv_sparse_data, 0, ssd_l)

                    if nnz > 0:
                        s = min(start_size, nnz)*self.dtype_size
                        # if slices to not match use two links!
                        send_sparse_idx_buf[ssi_l:ssi_l+s] = send_sparse_data_buf[:s]

                    start_time = time.time()
                    self._mpi.Sendrecv(send_sparse_idx_buf, mpi_rank,
                                       recvbuf=recv_sparse_idx_buf, source=mpi_rank)
                    self._network_time2 += time.time()-start_time

                    recv_sparse_size = recv_sparse_idx[start_size]
                    if recv_sparse_size > 0:
                        s = int(min(start_size, recv_sparse_size))*self.dtype_size
                        recv_sparse_data_buf[:s] = recv_sparse_idx_buf[ssi_l:ssi_l+s]

                send_sparse_idx[start_size] = tmp_x

                history[2][history[0]] = max(nnz, recv_sparse_size)
                history[0] = (history[0]+1)%history[2].size

                if nnz > start_size or recv_sparse_size > start_size:
                    # send second part
                    lsend = max(0, (nnz-start_size))
                    lrecv = max(0, (recv_sparse_size-start_size))
                    ssi_o, ssd_o = start_size*self.idtype_size, start_size*self.dtype_size
                    ssi_sl, ssd_sl = lsend*self.idtype_size, lsend*self.dtype_size
                    ssi_rl, ssd_rl = lrecv*self.idtype_size, lrecv*self.dtype_size
                    if True or self.mpi_join_communication_single_link is False:
                        send_sparse_idx_buf = np.getbuffer(send_sparse_idx, ssi_o, ssi_sl)
                        send_sparse_data_buf = np.getbuffer(send_sparse_data, ssd_o, ssd_sl)
                        recv_sparse_idx_buf = np.getbuffer(recv_sparse_idx, ssi_o, ssi_rl)
                        recv_sparse_data_buf = np.getbuffer(recv_sparse_data, ssd_o, ssd_rl)

                        start_time = time.time()
                        self._mpi.Sendrecv(send_sparse_idx_buf, mpi_rank,
                                           recvbuf=recv_sparse_idx_buf, source=mpi_rank)
                        self._mpi.Sendrecv(send_sparse_data_buf, mpi_rank,
                                           recvbuf=recv_sparse_data_buf, source=mpi_rank)
                        self._network_time3 += time.time()-start_time
                    else:
                        raise NotImplementedError()

                    self._network_traffic += (send_sparse_size + recv_sparse_size)
                else:
                    self._network_traffic += 2*start_size

                recv_sparse_idx = recv_sparse_idx[:recv_sparse_size]
                recv_sparse_data = recv_sparse_data[:recv_sparse_size]

            start_time = time.time()
            if recv_only_changes is False:
                if self.mpi_fast_sparsify is False:
                    recv_nda.ravel()[:] = 0
                else:
                    self._base_c.dist_set_to_zero(recv_nda.ravel(),
                                                  self._openmp_threads)
            self._proc_time5 += time.time() - start_time
            start_time = time.time()
            if self.mpi_fast_sparsify is False:
                recv_nda.ravel()[recv_sparse_idx] = recv_sparse_data
            else:
                self._base_c.sparse_update(recv_nda.ravel(),
                                           recv_sparse_idx,
                                           recv_sparse_data,
                                           self._openmp_threads)
            self._proc_time6 += time.time() - start_time
            start_time = time.time()
            if recv_cache is not None and recv_sparse_size > 0:
                recv_cache = recv_cache()

                if False:
                    size = self._base_c.merge_sparse(recv_cache[0][0],
                                                     recv_cache[1], recv_cache[2],
                                                     recv_sparse_size,
                                                     recv_sparse_idx, recv_sparse_data,
                                                     recv_cache[3], recv_cache[4])

                    # remove values that got zero
                    if self.mpi_fast_sparsify is False:
                        mask = recv_cache[4][:size] != 0
                        new_size = mask.sum()
                        recv_cache[3][:new_size] = recv_cache[3][:size][mask]
                        recv_cache[4][:new_size] = recv_cache[4][:size][mask]
                        size = new_size
                    else:
                        size = self._base_c.filter_zeros_sparse(size,
                                                                recv_cache[3], recv_cache[4])
                else:
                    size = self._base_c.merge_sparse_skip_zeros(recv_cache[0][0],
                                                                recv_cache[1], recv_cache[2],
                                                                recv_sparse_size,
                                                                recv_sparse_idx, recv_sparse_data,
                                                                recv_cache[3], recv_cache[4])

                recv_cache[0][0] = size
                recv_cache[5]()

                if self.mpi_cache_sparsity_check is True:
                    cache_sparsity_check("recv", recv_nda.ravel(), size,
                                         recv_cache[3], recv_cache[4])
            self._proc_time7 += time.time() - start_time

            if False:
                # check correctness
                # the check might be broken
                send_tmp = send_nda.ravel()
                recv_tmp = recv_nda.ravel().copy()
                s = send_tmp.size
                i = 0
                step = (2**31-1)/self.dtype(0).nbytes
                while i < s:
                    b, e = i, min(s, i+step)
                    #print b, e
                    self._mpi.Sendrecv(send_tmp[b:e], mpi_rank, recvbuf=recv_tmp[b:e], source=mpi_rank)
                    i = e
                assert np.allclose(recv_nda.ravel(), recv_tmp)
        else:
            # todo: inplace might make problems here.
            send_tmp = send_nda.ravel()
            recv_tmp = recv_nda.ravel()
            s = send_tmp.size
            i = 0
            step = (2**31-1)/self.dtype(0).nbytes
            while i < s:
                b, e = i, min(s, i+step)
                #print b, e
                start_time = time.time()
                self._mpi.Sendrecv(send_tmp[b:e], mpi_rank, recvbuf=recv_tmp[b:e], source=mpi_rank)
                self._network_time1 += time.time()-start_time
                i = e

            self._network_traffic += 2*s
        pass

    __remote_sparse_idx__ = None
    __sparsity_cache__ = None
    def _transfer_W_blocks(self, mpi_rank, send_block_idx, recv_block_idx):
        start_time = time.time()
        #self._log.all_info("Communicate with %i: send %i <-> recv %i" %
        #                   (mpi_rank, send_block_idx, recv_block_idx))
        Ws = self._W_block_size
        send_idx = np.asarray((send_block_idx*Ws, (send_block_idx+1)*Ws),
                              dtype=self.idtype)
        recv_idx = np.asarray((recv_block_idx*Ws, (recv_block_idx+1)*Ws),
                              dtype=self.idtype)

        send_o_changes, recv_o_changes = False, False
        send_changed_mask = None
        remote_idx = None
        keep_remote_idx = None
        send_cache, recv_cache = None, None
        if self.mpi_send_changes_back is True:
            # block to is a remote block, we send only diff back
            send_o_changes = send_block_idx == 2
            # local block as was at remote node
            # we will receive only changes
            recv_o_changes = recv_block_idx in [0, 1]
            send_changed_mask = self._changed_mask[send_idx[0]:send_idx[1]]

            if self.__remote_sparse_idx__  is None:
                size = self._W[send_idx[0]:send_idx[1]].size+1
                self.__remote_sparse_idx__ = [0, np.empty((size, ),
                                                          dtype=self.idtype)]

            if send_block_idx == 2:
                remote_idx = self.__remote_sparse_idx__
            if recv_block_idx == 2:
                keep_remote_idx = self.__remote_sparse_idx__

            if self.mpi_cache_sparsity is True:
                if self.__sparsity_cache__ is None:
                    size = self._W[send_idx[0]:send_idx[1]].size+1
                    sparse_cache_size = [[0], [0]]
                    sparse_cache_map = [0, 1, 2]
                    sparse_cache_idx = np.empty((len(self._local_global_groups)+1, size),
                                                dtype=self.idtype)
                    sparse_cache_data = np.empty((len(self._local_global_groups)+1, size),
                                                 dtype=self.dtype)

                    self.__sparsity_cache__ = [
                        sparse_cache_size,
                        sparse_cache_map,
                        sparse_cache_idx,
                        sparse_cache_data
                    ]
                else:
                    sparse_cache_size = self.__sparsity_cache__[0]
                    sparse_cache_map = self.__sparsity_cache__[1]
                    sparse_cache_idx = self.__sparsity_cache__[2]
                    sparse_cache_data = self.__sparsity_cache__[3]

                if send_block_idx in [0, 1]:
                    def getcache():
                        sidx = sparse_cache_map[send_block_idx]
                        didx = sparse_cache_map[2]
                        def update():
                            sparse_cache_map[send_block_idx], sparse_cache_map[2] = sparse_cache_map[2], sparse_cache_map[send_block_idx]
                        return (sparse_cache_size[send_block_idx],
                                sparse_cache_idx[sidx],
                                sparse_cache_data[sidx],
                                sparse_cache_idx[didx],
                                sparse_cache_data[didx],
                                update)
                    send_cache = getcache

                if recv_block_idx in [0, 1]:
                    def getcache():
                        sidx = sparse_cache_map[recv_block_idx]
                        didx = sparse_cache_map[2]
                        def update():
                            sparse_cache_map[recv_block_idx], sparse_cache_map[2] = sparse_cache_map[2], sparse_cache_map[recv_block_idx]
                        return (sparse_cache_size[recv_block_idx],
                                sparse_cache_idx[sidx],
                                sparse_cache_data[sidx],
                                sparse_cache_idx[didx],
                                sparse_cache_data[didx],
                                update)
                    recv_cache = getcache

        self._mpisendrecv(
            mpi_rank,
            self._W[send_idx[0]:send_idx[1]],
            self._W[recv_idx[0]:recv_idx[1]],
            send_block_idx == recv_block_idx,
            send_o_changes, recv_o_changes,
            send_changed_mask,
            remote_idx, keep_remote_idx,
            send_cache, recv_cache,
        )

        start_time2 = time.time()
        if self.mpi_send_changes_back is True:
            send_changed_mask[:, :] = 0
        self._proc_time8 += time.time() - start_time2

        self._communication_time += time.time()-start_time
        pass

    def _fit(self):
        """
        * matrix with mappings created for all solvers
        * then each solver has own structures to store active size
          information

        - each group has list of actives samples, i.e. has active classes
        - for each sample and each group is a list of active classes
        - if samples is actives can be determind by using group ranges!

        - group_ranges are the actives samples.
        - need new structure to save number of classes in group.
        - need matrix to store the class mapping. use the above.
        - structure for active group samples.
        """
        # todo: implement in a smarter way
        # this is a view on a cache aligned array
        class_optimal = self._zeros_with_cache_aligned_rows(
            None,
            (self._all_classes.size, 1), dtype=np.uint8)[:, 0]

        if self._global_grouping is False:
            block_size = self._group_count
        else:
            block_size = self.idtype(self._local_block_count) * self._W_block_size

        class_tmp = np.zeros((block_size, self._max_group_size),
                             dtype=self.dtype)
        idx_tmp = np.zeros((block_size, self._max_group_size),
                           dtype=self.idtype)
        pairs_tmp = np.empty((self._group_count, 2),
                             dtype=self.idtype)
        pairs_shuffle_tmp = np.empty((self._group_count,),
                                     dtype=np.uint8)

        epsilon = [self.epsilon]
        last_violation = [-1000.0]
        iter_since_fs = [0]
        data_shrink_state = np.zeros((1,), dtype=np.int8)
        gr_to_shr = np.zeros((1,), dtype=self.idtype)
        shrinked_start_state = np.int8(0)
        shrink_state = np.int8(1)
        fresh_start_count = [0]
        if self.shrinking == 1:
            fresh_start = [True]
            if self.decrement_epsilon is True:
                epsilon = [min(1.0, 10 * self.epsilon)]

			# can be more efficient!
            data_shrink_state = np.zeros((self._group_count *
                                          self._idx.shape[0]),
                                         dtype=np.int8)
            assert self.shrinked_start_state < 0
            if self.shrinked_start_state != -1:
                raise NotImplementedError("Not used at the moment.")
            shrinked_start_state = np.int8(self.shrinked_start_state)
            shrink_state = np.int8(self.shrink_state)
            if self._global_grouping is False or self.reduce_mem_allocation is False:
                gr_to_shr = (np.arange(self._group_count,
                                       dtype=self.idtype) *
                             (self._idx.shape[0]))
            else:
                gr_to_shr = np.zeros(self._group_count,
                                     dtype=self.idtype)
                tmp = self.idtype(0)
                tmpi = 0
                for i in xrange(self._global_group_count):
                    for g in self._global_groups[i][:self._global_group_sizes[i]]:
                        gr_to_shr[tmpi] = tmp
                        tmp += self._samples_per_ggr[i]
                        tmpi += 1

        self._changed_mask = np.zeros((self._W.shape[0],
                                       self._dimensions),
                                      np.bool)

        if self._global_grouping is False:
            # easier track:
            rounds = self._group_count-1
            if self._group_count % 2 == 1:
                rounds += 1
            rounds = np.arange(rounds+1, dtype=self.idtype)

            def update_function(i, class_max_violation):
                class_optimal[:] = 1

                if self.shrinking == 1 and fresh_start[0] is True:
                    fresh_start_count[0] += 1
                    if self.shrinking == 1:
                        data_shrink_state[:] = 0
                    iter_since_fs[0] = 0
                else:
                    iter_since_fs[0] += 1

                if self.shuffle is True:
                    for a, b in self._group_data_ranges:
                        np.random.shuffle(self._idx[a:b])

                X, yi, K = self._X, self._yi, self._K
                alpha, idx = self._alpha, self._idx

                for fold in xrange(self.folds):

                    if self.shuffle_rounds is True:
                        np.random.shuffle(rounds)
                        pairs_shuffle_tmp[:] = np.random.randint(0, 2, pairs_shuffle_tmp.shape, pairs_shuffle_tmp.dtype)

                    args = [
                        X,
                        yi,
                        K,
                        idx,

                        self.idtype(0),
                        self.idtype(0),
                        self._global_group_sizes,
                        self._global_groups,
                        self._ggr_to_samples,

                        self._group_sizes,
                        self._group_mapping,
                        self._yi_to_gx,
                        self._folded_group_data_ranges[fold],
                        pairs_tmp,
                        pairs_shuffle_tmp,
                        rounds,

                        class_optimal,
                        class_max_violation,
                        self._W,
                        self._alpha,
                        self._gr_to_alpha,
                        self._classes,
                        class_tmp,
                        idx_tmp,

                        self.epsilon,
                        self.C,
                        self.idtype(0),

                        np.uint32(self.shrinking),
                        data_shrink_state,
                        gr_to_shr,
                        shrink_state,
                        shrinked_start_state,
                        last_violation[0],

                        np.uint8(self.mpi_send_changes_back),
                        self._changed_mask.view(np.int8),
                    ]
                    c = self._ww_c
                    if self.variant == 0:
                        c.ww_sparse_solver_updates_local__variant_0(*args)
                    elif self.variant == 1:
                        c.ww_sparse_solver_updates_local__variant_1(*args)
                    else:
                        raise NotImplementedError()
                violation = self._mpi.max(class_max_violation.max())

                extra = ""
                if self.shrinking == 1:
                    optimal = False

                    active_count = (float((data_shrink_state >= 0).sum()) /
                                    self._group_count)
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

                    if self.shrinking_max_iter > 0 and iter_since_fs[0] > self.shrinking_max_iter:
                        extra += ", fresh_start"
                        fresh_start[0] = True

                    if(self.stop_shrinking != 0 and
                       self.stop_shrinking * self.epsilon > violation):
                        self._log.info("Stop shrinking at iterations %i." % i)
                        self.shrinking = 0
                        fresh_start[0] = True
                else:
                    optimal = self._mpi.land(class_optimal.min())

                last_violation[0] = violation
                return optimal, extra
        else:
            self._communication_time = 0
            self._proc_time1 = 0
            self._proc_time2 = 0
            self._proc_time3 = 0
            self._proc_time4 = 0
            self._proc_time5 = 0
            self._proc_time6 = 0
            self._proc_time7 = 0
            self._proc_time8 = 0
            self._network_time1 = 0
            self._network_time2 = 0
            self._network_time3 = 0
            self._network_traffic = 0

            # use same random state for cross mpi
            # randomness
            maxint = np.iinfo(np.int32).max
            next_seed = [np.random.randint(maxint)]
            random_state = np.random.RandomState()

            mpi_rounds = self._mpi_size-1
            if self._mpi_size % 2 == 1:
                mpi_rounds += 1
            mpi_rounds = np.arange(mpi_rounds+1, dtype=self.idtype)

            data_distributed = self.mpi_data_distributed
            mpir = self._mpi_rank

            def update_function(i, class_max_violation):
                class_optimal[:] = 1

                if self.shrinking == 1 and fresh_start[0] is True:
                    fresh_start_count[0] += 1
                    if self.shrinking == 1:
                        data_shrink_state[:] = 0
                    iter_since_fs[0] = 0
                else:
                    iter_since_fs[0] += 1

                random_state.seed(next_seed[0])
                next_seed[0] = np.random.randint(maxint)

                if self.shuffle is True:
                    for a, b in self._group_data_ranges:
                        random_state.shuffle(self._idx[a:b])

                X, yi, K = self._X, self._yi, self._K
                alpha, idx = self._alpha, self._idx

                global_folds = 1
                if self.mpi_local_folds is False:
                    global_folds = self.folds
                for global_fold in xrange(global_folds):

                    if self.shuffle_rounds is True:
                        random_state.shuffle(mpi_rounds)

                    for mpi_round in mpi_rounds:
                        self._mpi.barrier()

                        #random_state.seed(next_seed[0])
                        #next_seed[0] = np.random.randint(maxint)

                        pairs = []
                        if mpi_round == 0:
                            """
                            (local_tag, ggia, ggib)

                            => ggia optimizes with ggib
                            """
                            pairs.append((1,
                                          {"l1": self._local_global_groups[0],
                                           "l2": self._local_global_groups[0]}))
                            if len(self._local_global_groups) > 1:
                                pairs.append((1,
                                              {"l1": self._local_global_groups[1],
                                               "l2": self._local_global_groups[1]}))
                                pairs.append((1,
                                              {"l1": self._local_global_groups[0],
                                               "l2": self._local_global_groups[1]}))
                        else:
                            mpim = int(get_match(mpir, mpi_round-1, self._mpi_size))

                            if mpim == mpir:
                                continue

                            """
                            (remote_tag, pairings)

                            l1, l2: local
                            r1, r2: remote
                            """
                            n1, n2 = sorted((mpir, mpim))
                            ggr_la, ggr_lb = 2*n1, 2*n1+1
                            ggr_ra, ggr_rb = 2*n2, 2*n2+1
                            pairs.append((2,
                                          {"l1": ggr_la, "l2": ggr_ra,
                                           "r1": ggr_lb, "r2": ggr_rb}))
                            pairs.append((2,
                                          {"l1": ggr_la, "l2": ggr_rb,
                                           "r1": ggr_lb, "r2": ggr_ra}))
                            if data_distributed is True:
                                pairs.append((2,
                                              {"l1": ggr_lb, "l2": ggr_ra,
                                               "r1": ggr_la, "r2": ggr_rb}))
                                pairs.append((2,
                                              {"l1": ggr_lb, "l2": ggr_rb,
                                               "r1": ggr_la, "r2": ggr_ra}))

                            if mpir > mpim:
                                # "opposite site", flip pairs
                                for p in pairs:
                                    d = p[1]
                                    d["l1"], d["r2"] = d["r2"], d["l1"]
                                    d["l2"], d["r1"] = d["r1"], d["l2"]

                        if self.shuffle_rounds is True:
                            random_state.shuffle(pairs)

                        #self._log.all_info("%i - %i: %s" % (i, mpi_round, pairs,))
                        first_pair = True
                        for pair in pairs:
                            if pair[1]["l1"] == pair[1]["l2"]:
                                s = self._global_group_sizes[pair[1]["l1"]]
                                rounds = s-1
                                if s % 2 == 1:
                                    rounds += 1
                                rounds += 1 # each group with itself
                            else:
                                rounds = max(self._global_group_sizes[pair[1]["l1"]],
                                             self._global_group_sizes[pair[1]["l2"]])
                            ggia, ggib = pair[1]["l1"], pair[1]["l2"]
                            rounds = np.arange(rounds, dtype=self.idtype)

                            if pair[0] == 2:
                                if self.mpi_com_switch is False or first_pair is True:
                                    self._transfer_W_blocks(mpim, pair[1]["r1"] % 2, 2)
                                else:
                                    if mpir > mpim:
                                        self._transfer_W_blocks(mpim, pair[1]["r1"] % 2, pair[1]["l1"] % 2)
                                    else:
                                        self._transfer_W_blocks(mpim, 2, 2)

                            local_folds = 1
                            if self.mpi_local_folds is True:
                                local_folds = self.folds

                            lcmv = None
                            for inner_repeat in xrange(self.inner_repeat):

                                if self.shuffle is True:
                                    for a, b in self._group_data_ranges:
                                        random_state.shuffle(self._idx[a:b])

                                class_max_violation2 = np.zeros_like(class_max_violation)

                                for local_fold in xrange(local_folds):
                                    if self.mpi_local_folds is True:
                                        fold = local_fold
                                    else:
                                        fold = global_fold

                                    if self.shuffle_rounds is True and rounds.size > 1:
                                        random_state.shuffle(rounds)
                                        pairs_shuffle_tmp[:] = random_state.randint(0, 2, pairs_shuffle_tmp.shape, pairs_shuffle_tmp.dtype)

                                    #self._log.all_info("MPI-round %i: Optimize ggr %i with ggr %i." %
                                    #                   (mpi_round, ggia, ggib))
                                    args = [
                                        X,
                                        yi,
                                        K,
                                        idx,

                                        self.idtype(ggia),
                                        self.idtype(ggib),
                                        self._global_group_sizes,
                                        self._global_groups,
                                        self._ggr_to_samples,

                                        self._group_sizes,
                                        self._group_mapping,
                                        self._yi_to_gx,
                                        self._folded_group_data_ranges[fold],
                                        pairs_tmp,
                                        pairs_shuffle_tmp,
                                        rounds,

                                        class_optimal,
                                        class_max_violation2,
                                        self._W,
                                        self._alpha,
                                        self._gr_to_alpha,
                                        self._classes,
                                        class_tmp,
                                        idx_tmp,

                                        self.epsilon,
                                        self.C,
                                        # just update where we have local alpha
                                        self.idtype(data_distributed is True and
                                                    pair[0] == 2),

                                        np.uint32(self.shrinking),
                                        data_shrink_state,
                                        gr_to_shr,
                                        shrink_state,
                                        shrinked_start_state,
                                        last_violation[0],

                                        np.uint8(self.mpi_send_changes_back),
                                        self._changed_mask.view(np.int8),
                                    ]
                                    c = self._ww_c
                                    if self.variant == 0:
                                        c.ww_sparse_solver_updates_local__variant_0(*args)
                                    elif self.variant == 1:
                                        c.ww_sparse_solver_updates_local__variant_1(*args)
                                    else:
                                        raise NotImplementedError()

                                if lcmv is None:
                                    lcmv = class_max_violation2.max()
                                else:
                                    #self._log.info("Improved by: %f" % (class_max_violation2.max()-lcmv))
                                    lcmv = class_max_violation2.max()
                                class_max_violation[0] = np.maximum(class_max_violation[0], lcmv)
                                if lcmv == 0:
                                    break
                                pass

                            if pair[0] == 2:
                                if self.mpi_com_switch is False or first_pair is False:
                                    self._transfer_W_blocks(mpim, 2, pair[1]["r1"] % 2)
                            first_pair = False

                violation = self._mpi.max(class_max_violation.max())

                extra = ""
                if self.shrinking == 1:
                    optimal = False

                    # todo: needs to be fixed for mpi
                    # todo: same for max_violation
                    active_count = (float((data_shrink_state >= 0).sum()) /
                                    self._group_count)
                    extra += ", active %6g samples" % active_count

                    changed_mean = self._changed_mask.sum(dtype=self.dtype)/self._changed_mask.shape[0]
                    extra += ", changed %6g dim" % changed_mean

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

                    if self.shrinking_max_iter > 0 and iter_since_fs[0] > self.shrinking_max_iter:
                        extra += ", fresh_start"
                        fresh_start[0] = True

                    if(self.stop_shrinking != 0 and
                       self.stop_shrinking * self.epsilon > violation):
                        self._log.info("Stop shrinking at iterations %i." % i)
                        self.shrinking = 0
                        fresh_start[0] = True
                else:
                    optimal = self._mpi.land(class_optimal.min())

                last_violation[0] = violation
                return optimal, extra

        ret = super(WW_Sparse_Solver, self)._fit(update_function)

        self._log.debug("Fresh start count: %i" % fresh_start_count[0])

        if self._global_grouping is True:
            net_s = self._network_time1+self._network_time2+self._network_time3
            proc_s = (self._proc_time1+self._proc_time2+self._proc_time3+self._proc_time4+
                     self._proc_time5+self._proc_time6+self._proc_time7+self._proc_time8)

            msg = ("Communication time:    %g\n" % self._communication_time)+\
                  ("Network time - 1:      %g\n" % self._network_time1)+\
                  ("Network time - 2:      %g\n" % self._network_time2)+\
                  ("Network time - 3:      %g\n" % self._network_time3)+\
                  ("Network time - s:      %g\n" % net_s)+\
                  ("Network traffic:       %g MB\n" % (self._network_traffic/(1024.0**2)))+\
                  ("Com - net time:        %g\n" % (self._communication_time - net_s))+\
                  ("Proc time - 1:         %g\n" % self._proc_time1)+\
                  ("Proc time - 2:         %g\n" % self._proc_time2)+\
                  ("Proc time - 3:         %g\n" % self._proc_time3)+\
                  ("Proc time - 4:         %g\n" % self._proc_time4)+\
                  ("Proc time - 5:         %g\n" % self._proc_time5)+\
                  ("Proc time - 6:         %g\n" % self._proc_time6)+\
                  ("Proc time - 7:         %g\n" % self._proc_time7)+\
                  ("Proc time - 8:         %g\n" % self._proc_time8)+\
                  ("Proc time - s:         %g\n" % proc_s)+\
                  ("Com - net s -proc s:   %g\n" % (self._communication_time - net_s -proc_s))

            self._log.all_info(msg)
        return ret
