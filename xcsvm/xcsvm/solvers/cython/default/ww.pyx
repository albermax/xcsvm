#// cython: profile=True
#// cython: linetrace=True
#// cython: binding=True
#// distutils: define_macros=CYTHON_TRACE_NOGIL=1

from cmath import *
import cython
from cython cimport parallel
import numpy as np
cimport numpy as np
cimport openmp
import scipy

from cython.operator cimport dereference as DR
from cython.operator cimport postincrement as PI

from .dtypes cimport *

__all__ = [
    "ww_sparse_solver_updates_local__variant_0",
    "ww_sparse_solver_updates_local__variant_1",
]


###############################################################################


cdef IDTYPE_t get_match(IDTYPE_t c, IDTYPE_t r, IDTYPE_t C):
    cdef np.uint8_t even = C % 2 == 0
    if not even:
        C += 1

    cdef IDTYPE_t ret = c
    if c == C-1:
        ret = r
    elif c == r:
        ret = C-1
    else:
        ret = (2*(C-1)+2*r-c) % (C-1)

    if not even and ret == C-1:
        ret = c

    return ret;


###############################################################################


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
def ww_sparse_solver_updates_local__variant_0(
        object X,	
        np.ndarray[IDTYPE_t] y,
        np.ndarray[DTYPE_t] K,
        np.ndarray[IDTYPE_t] idx,

        IDTYPE_t ggia,
        IDTYPE_t ggib,
        np.ndarray[IDTYPE_t, ndim=1] global_group_sizes,
        np.ndarray[IDTYPE_t, ndim=2] global_groups,
        np.ndarray[IDTYPE_t, ndim=2] ggr_to_samples,
        
        np.ndarray[IDTYPE_t, ndim=1] group_sizes,
        np.ndarray[IDTYPE_t, ndim=1] group_mapping,
        np.ndarray[IDTYPE_t, ndim=2] yi_to_gx,
        np.ndarray[IDTYPE_t, ndim=2] group_data_ranges,
        np.ndarray[IDTYPE_t, ndim=2] pairs,
        np.ndarray[np.uint8_t, ndim=1] pairs_shuffle,
        np.ndarray[IDTYPE_t, ndim=1] rounds_lookup,

        np.ndarray[np.uint8_t] class_optimal,
        np.ndarray[DTYPE_t] class_max_violation,
        np.ndarray[DTYPE_t, ndim=3] global_W,
        np.ndarray[DTYPE_t, ndim=1] global_alpha,
        np.ndarray[IDTYPE_t, ndim=1] gr_to_alpha,
        np.ndarray[DTYPE_t] classes,
        np.ndarray[DTYPE_t, ndim=2] global_class_tmp,
        np.ndarray[IDTYPE_t, ndim=2] global_idx_tmp,

        DTYPE_t epsilon,
        DTYPE_t C,
        IDTYPE_t update_only_first,

        IDTYPE_t shrinking,
        np.ndarray[np.int8_t, ndim=1] global_data_shrink_state,
        np.ndarray[IDTYPE_t, ndim=1] gr_to_shr,
        np.int8_t shrink_state,
        np.int8_t shrinked_start_state,
        DTYPE_t old_max_violation,

        np.uint8_t log_changes,
        np.ndarray[np.int8_t, ndim=2] global_changed_mask,
):

    cdef IDTYPE_t round, round_idx, c, match, len_pairs, pi

    cdef DTYPE_t[:] data = X.data
    cdef np.int32_t[:] indices = X.indices
    cdef np.int32_t[:] indptr = X.indptr

    cdef IDTYPE_t i, ii, ii2, j, gia, gib, xi, gi, ci, idx_tmp_i, idx_tmp_ii
    cdef IDTYPE_t yi, yi_rel
    cdef DTYPE_t tmp, a

    cdef DTYPE_t *W_a, *W_b
    cdef IDTYPE_t Wxdim = global_W.shape[2]
    cdef DTYPE_t *alpha
    cdef IDTYPE_t alphaxdim = Wxdim
    cdef DTYPE_t * class_tmp
    cdef IDTYPE_t * idx_tmp, * idx2
    cdef DTYPE_t max_buffer, cur_buffer
    cdef np.int8_t * data_shrink_state = NULL
    cdef np.int8_t * changed_mask_a = NULL
    cdef np.int8_t * changed_mask_b = NULL

    for round_idx in xrange(rounds_lookup.shape[0]):
        len_pairs = 0

        round = rounds_lookup[round_idx]

        if ggia == ggib:
            if round == 0:
                for gi in xrange(global_group_sizes[ggia]):
                    pairs[len_pairs, 0] = global_groups[ggia, gi]
                    pairs[len_pairs, 1] = global_groups[ggia, gi]
                    len_pairs += 1
            else:
                if global_group_sizes[ggia] == 1:
                    continue

                for gi in xrange(global_group_sizes[ggia]):
                    match = get_match(gi, round-1, global_group_sizes[ggia])
                    if gi < match:
                        pairs[len_pairs, 0] = global_groups[ggia, gi]
                        pairs[len_pairs, 1] = global_groups[ggia, match]
                        len_pairs += 1
        else:
            if global_group_sizes[ggia] > global_group_sizes[ggib]:
                assert rounds_lookup.shape[0] == global_group_sizes[ggia]
                for gi in xrange(global_group_sizes[ggib]):
                    pairs[len_pairs, 0] = global_groups[ggia,
                                                        (gi+round) %
                                                        global_group_sizes[ggia]]
                    pairs[len_pairs, 1] = global_groups[ggib, gi]
                    len_pairs += 1
            else:
                assert rounds_lookup.shape[0] == global_group_sizes[ggib]
                for gi in xrange(global_group_sizes[ggia]):
                    pairs[len_pairs, 0] = global_groups[ggia, gi]
                    pairs[len_pairs, 1] = global_groups[ggib,
                                                        (gi+round) %
                                                        global_group_sizes[ggib]]
                    len_pairs += 1

        for pi in parallel.prange(len_pairs,
                                  nogil=True,
                                  schedule="dynamic"):
        #for pi in xrange(len_pairs):
            for xi in xrange(2):
                if (pairs_shuffle[round]+xi)%2 == 0:
                    gia, gib = pairs[pi, 0], pairs[pi, 1]
                    idx2 = &ggr_to_samples[ggib, 0]
                else:
                    gia, gib = pairs[pi, 1], pairs[pi, 0]
                    idx2 = &ggr_to_samples[ggia, 0]
                    if gia == gib:
                        continue

                W_a = &global_W[group_mapping[gia], 0, 0]
                W_b = &global_W[group_mapping[gib], 0, 0]
                alpha = &global_alpha[gr_to_alpha[gib]]
                class_tmp = &global_class_tmp[group_mapping[gia], 0]
                idx_tmp = &global_idx_tmp[group_mapping[gia], 0]
                if shrinking == 1:
                    data_shrink_state = &global_data_shrink_state[gr_to_shr[gib]]
                if log_changes == 1:
                    changed_mask_a = &global_changed_mask[group_mapping[gia], 0]
                    changed_mask_b = &global_changed_mask[group_mapping[gib], 0]

                for i in xrange(group_data_ranges[gia, 0], group_data_ranges[gia, 1]):
                    ii = idx[i]
                    ii2 = idx2[ii]
                    #assert ii2 != 10000000

                    if shrinking == 1 and data_shrink_state[ii2] < 0:
                        #data_shrink_state[ii2] += 1
                        continue

                    yi = y[ii]
                    yi_rel = yi_to_gx[yi, 1]

                    #  it follows g = -1 - X[ii].dot(W[gia]-W[gib])
                    for ci in xrange(group_sizes[gib]):
                        class_tmp[ci] = 0

                    tmp = 0
                    if gia == gib:
                        for j in xrange(indptr[ii], indptr[ii+1]):
                            for ci in xrange(group_sizes[gib]):
                                class_tmp[ci] = class_tmp[ci] + data[j] * W_b[indices[j]*Wxdim+ci]
                        tmp = class_tmp[yi_rel]
                    else:
                        for j in xrange(indptr[ii], indptr[ii+1]):
                            tmp = tmp + data[j] * W_a[indices[j]*Wxdim+yi_rel]
                            for ci in xrange(group_sizes[gib]):
                                class_tmp[ci] = class_tmp[ci] + data[j] * W_b[indices[j]*Wxdim+ci]

                    for ci in xrange(group_sizes[gib]):
                        class_tmp[ci] = tmp - 1.0 - class_tmp[ci]

                    max_buffer = 1E10;
                    tmp = 0
                    idx_tmp_i = 0

                    for ci in xrange(group_sizes[gib]):
                        if gia == gib and ci == yi_rel:
                            continue

                        a = alpha[ii2*alphaxdim+ci]
                        class_tmp[ci] = class_tmp[ci] + tmp*K[ii]
                        if not ((class_tmp[ci] < - epsilon and a < C) or
                                class_tmp[ci] > epsilon and a > 0):
                            class_tmp[ci] = 0
                            # todo: shirnk in case class_tmp[ci] and a == 0

                            # if shrinking == 1:
                            #     if a == C:
                            #         cur_buffer = -class_tmp[ci]
                            #     elif a == 0:
                            #         cur_buffer = class_tmp[ci]
                            #     else:
                            #         cur_buffer = 0
                            #     max_buffer = min(max_buffer, cur_buffer)
                            continue
                        class_optimal[gia] = 0
                        if class_tmp[ci] < 0:
                            class_max_violation[gia] = max(class_max_violation[gia], -class_tmp[ci])
                        else:
                            class_max_violation[gia] = max(class_max_violation[gia], class_tmp[ci])

                        class_tmp[idx_tmp_i] = -0.5 * class_tmp[ci] / K[ii]

                        if a+class_tmp[idx_tmp_i] < 0:
                            class_tmp[idx_tmp_i] = -a
                            alpha[ii2*alphaxdim+ci] = 0
                        elif a+class_tmp[idx_tmp_i] > C:
                            class_tmp[idx_tmp_i] = C-a
                            alpha[ii2*alphaxdim+ci] = C
                        else:
                            alpha[ii2*alphaxdim+ci] += class_tmp[idx_tmp_i]
                        tmp = tmp + class_tmp[idx_tmp_i]
                        idx_tmp[idx_tmp_i] = ci
                        idx_tmp_i = idx_tmp_i + 1

                    if idx_tmp_i > 0:
                        if shrinking == 1:
                            data_shrink_state[ii2] = 0
                        
                        # it follows W[gia] += delta * X[ii]
                        # it follows W[gib] -= delta * X[ii]
                        for j in xrange(indptr[ii], indptr[ii+1]):
                            if log_changes == 1:
                                changed_mask_a[indices[j]] = 1
                                changed_mask_b[indices[j]] = 1

                            W_a[indices[j]*Wxdim+yi_rel] += tmp * data[j]
                            for idx_tmp_ii in xrange(0, idx_tmp_i):
                                W_b[indices[j]*Wxdim+idx_tmp[idx_tmp_ii]] -= class_tmp[idx_tmp_ii] * data[j]
                    elif shrinking == 1 and idx_tmp_i < group_sizes[gib]:
                        data_shrink_state[ii2] += 1
                        if data_shrink_state[ii2] >= shrink_state:
                            data_shrink_state[ii2] = shrinked_start_state

    pass


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
def ww_sparse_solver_updates_local__variant_1(
        object X,	
        np.ndarray[IDTYPE_t] y,
        np.ndarray[DTYPE_t] K,
        np.ndarray[IDTYPE_t] idx,

        IDTYPE_t ggia,
        IDTYPE_t ggib,
        np.ndarray[IDTYPE_t, ndim=1] global_group_sizes,
        np.ndarray[IDTYPE_t, ndim=2] global_groups,
        np.ndarray[IDTYPE_t, ndim=2] ggr_to_samples,

        np.ndarray[IDTYPE_t, ndim=1] group_sizes,
        np.ndarray[IDTYPE_t, ndim=1] group_mapping,
        np.ndarray[IDTYPE_t, ndim=2] yi_to_gx,
        np.ndarray[IDTYPE_t, ndim=2] group_data_ranges,
        np.ndarray[IDTYPE_t, ndim=2] pairs,
        np.ndarray[np.uint8_t, ndim=1] pairs_shuffle,
        np.ndarray[IDTYPE_t, ndim=1] rounds_lookup,

        np.ndarray[np.uint8_t] class_optimal,
        np.ndarray[DTYPE_t] class_max_violation,
        np.ndarray[DTYPE_t, ndim=3] global_W,
        np.ndarray[DTYPE_t, ndim=1] global_alpha,
        np.ndarray[IDTYPE_t, ndim=1] gr_to_alpha,
        np.ndarray[DTYPE_t] classes,
        np.ndarray[DTYPE_t, ndim=2] global_class_tmp,
        np.ndarray[IDTYPE_t, ndim=2] global_idx_tmp,

        DTYPE_t epsilon,
        DTYPE_t C,
        IDTYPE_t update_only_first,

        IDTYPE_t shrinking,
        np.ndarray[np.int8_t, ndim=1] global_data_shrink_state,
        np.ndarray[IDTYPE_t, ndim=1] gr_to_shr,
        np.int8_t shrink_state,
        np.int8_t shrinked_start_state,
        DTYPE_t old_max_violation,

        np.uint8_t log_changes,
        np.ndarray[np.int8_t, ndim=2] global_changed_mask,
):

    cdef IDTYPE_t round, round_idx, c, match, len_pairs, pi

    cdef DTYPE_t[:] data = X.data
    cdef np.int32_t[:] indices = X.indices
    cdef np.int32_t[:] indptr = X.indptr
    cdef DTYPE_t *wp, *wp2, *dp
    cdef IDTYPE_t * idxp, *idxp2, *idx2

    cdef DTYPE_t *W_a, *W_b
    cdef IDTYPE_t Wxdim = global_W.shape[2]
    cdef DTYPE_t *alpha
    cdef IDTYPE_t alphaxdim = Wxdim
    cdef DTYPE_t * class_tmp
    cdef IDTYPE_t * idx_tmp

    cdef IDTYPE_t i, ii, ii2, j, gia, gib, xi, gi, ci, idx_tmp_i
    cdef IDTYPE_t yi, yi_rel
    cdef DTYPE_t tmp, a, tmp_y, max_violation
    cdef np.uint8_t optimal

    cdef IDTYPE_t gra0, gra1, grb0, grb1
    cdef np.int32_t dindicesp
    cdef np.int32_t *indicesp, *indicesp1, *indicesp2
    cdef DTYPE_t *datap, *datap1
    cdef DTYPE_t max_buffer, cur_buffer
    cdef np.int8_t * data_shrink_state = NULL
    cdef np.int8_t * changed_mask_a = NULL
    cdef np.int8_t * changed_mask_b = NULL

    for round_idx in xrange(rounds_lookup.size):
        len_pairs = 0

        round = rounds_lookup[round_idx]

        if ggia == ggib:
            if round == 0:
                for gi in xrange(global_group_sizes[ggia]):
                    pairs[len_pairs, 0] = global_groups[ggia, gi]
                    pairs[len_pairs, 1] = global_groups[ggia, gi]
                    len_pairs += 1
            else:
                if global_group_sizes[ggia] == 1:
                    continue

                for gi in xrange(global_group_sizes[ggia]):
                    match = get_match(gi, round-1, global_group_sizes[ggia])
                    if gi < match:
                        pairs[len_pairs, 0] = global_groups[ggia, gi]
                        pairs[len_pairs, 1] = global_groups[ggia, match]
                        len_pairs += 1
        else:
            if global_group_sizes[ggia] > global_group_sizes[ggib]:
                assert rounds_lookup.shape[0] == global_group_sizes[ggia]
                for gi in xrange(global_group_sizes[ggib]):
                    pairs[len_pairs, 0] = global_groups[ggia,
                                                        (gi+round) %
                                                        global_group_sizes[ggia]]
                    pairs[len_pairs, 1] = global_groups[ggib, gi]
                    len_pairs += 1
            else:
                assert rounds_lookup.shape[0] == global_group_sizes[ggib]
                for gi in xrange(global_group_sizes[ggia]):
                    pairs[len_pairs, 0] = global_groups[ggia, gi]
                    pairs[len_pairs, 1] = global_groups[ggib,
                                                        (gi+round) %
                                                        global_group_sizes[ggib]]
                    len_pairs += 1
                    
        for pi in parallel.prange(len_pairs,
                                  nogil=True,
                                  schedule="dynamic"):
        #for pi in xrange(len_pairs):
            for xi in xrange(2):
                if (pairs_shuffle[round]+xi)%2 == 0:
                    gia, gib = pairs[pi, 0], pairs[pi, 1]
                    idx2 = &ggr_to_samples[ggib, 0]
                else:
                    gia, gib = pairs[pi, 1], pairs[pi, 0]
                    idx2 = &ggr_to_samples[ggia, 0]
                    if gia == gib:
                        continue

                gra0, gra1 = 0, group_sizes[gia]
                grb0, grb1 = 0, group_sizes[gib]

                W_a = &global_W[group_mapping[gia], 0, 0]
                W_b = &global_W[group_mapping[gib], 0, 0]
                alpha = &global_alpha[gr_to_alpha[gib]]
                class_tmp = &global_class_tmp[group_mapping[gia], 0]
                idx_tmp = &global_idx_tmp[group_mapping[gia], 0]
                if shrinking == 1:
                    data_shrink_state = &global_data_shrink_state[gr_to_shr[gib]]
                if log_changes == 1:
                    changed_mask_a = &global_changed_mask[group_mapping[gia], 0]
                    changed_mask_b = &global_changed_mask[group_mapping[gib], 0]

                optimal = class_optimal[gia]
                max_violation = class_max_violation[gia]

                for i in xrange(group_data_ranges[gia, 0], group_data_ranges[gia, 1]):
                    ii = idx[i]
                    ii2 = idx2[ii]

                    if shrinking == 1 and data_shrink_state[ii2] < 0:
                        #data_shrink_state[ii2] += 1
                        continue
                    
                    yi = y[ii]
                    yi_rel = yi_to_gx[yi, 1]

                    indicesp1, indicesp2 = &indices[indptr[ii]], &indices[indptr[ii+1]]
                    datap1 = &data[indptr[ii]]

                    #  it follows g = -1 - X[ii].dot(W[gia]-W[gib])
                    for ci in xrange(grb0, grb1):
                        class_tmp[ci] = 0

                    tmp_y = 0
                    if gia == gib:
                        dp = &class_tmp[grb0]
                        indicesp, datap = indicesp1, datap1
                        while indicesp != indicesp2:
                            ci, a, dindicesp = 0, DR(datap), DR(indicesp)
                            wp = W_b+dindicesp*Wxdim+grb0
                            wp2 = W_b+dindicesp*Wxdim+grb1

                            while wp != wp2:
                                dp[PI(ci)] += a * DR(PI(wp))

                            PI(indicesp); PI(datap)

                        tmp_y = class_tmp[yi_rel]
                    else:
                        dp = &class_tmp[grb0]
                        indicesp, datap = indicesp1, datap1
                        while indicesp != indicesp2:
                            ci, a, dindicesp = 0, DR(datap), DR(indicesp)
                            wp = W_b+dindicesp*Wxdim+grb0
                            wp2 = W_b+dindicesp*Wxdim+grb1

                            while wp != wp2:
                                dp[PI(ci)] += a * DR(PI(wp))

                            PI(indicesp); PI(datap)

                        indicesp, datap = indicesp1, datap1
                        while indicesp != indicesp2:
                            a, dindicesp = DR(datap), DR(indicesp)
                            tmp_y = tmp_y + a * W_a[dindicesp*Wxdim+yi_rel]
                            PI(indicesp); PI(datap)

                    for ci in xrange(grb0, grb1):
                        class_tmp[ci] = tmp_y - 1.0 - class_tmp[ci]

                    max_buffer = 1E10;
                    tmp_y = 0
                    idx_tmp_i = grb0
                    for ci in xrange(grb0, grb1):
                        if gia == gib and ci == yi_rel:
                            continue

                        a = alpha[ii2*alphaxdim+ci]
                        tmp = class_tmp[ci]
                        tmp = tmp + tmp_y*K[ii]
                        if not ((tmp < - epsilon and a < C) or
                                tmp > epsilon and a > 0):
                            tmp = 0

                            # if shrinking == 1:
                            #     if a == C:
                            #         cur_buffer = -tmp
                            #     elif a == 0:
                            #         cur_buffer = tmp
                            #     else:
                            #         cur_buffer = 0
                            #     max_buffer = min(max_buffer, cur_buffer)
                            continue
                        optimal = 0
                        if tmp < 0:
                            max_violation = max(max_violation, -tmp)
                        else:
                            max_violation = max(max_violation, tmp)

                        tmp = -0.5 * tmp / K[ii]

                        if a+tmp < 0:
                            tmp = -a
                            alpha[ii2*alphaxdim+ci] = 0
                        elif a+tmp > C:
                            tmp = C-a
                            alpha[ii2*alphaxdim+ci] = C
                        else:
                            alpha[ii2*alphaxdim+ci] += tmp
                        tmp_y = tmp_y + tmp
                        class_tmp[idx_tmp_i] = tmp
                        idx_tmp[idx_tmp_i] = ci
                        idx_tmp_i = idx_tmp_i + 1

                    if idx_tmp_i > grb0:
                        # it follows W[gia] += delta * X[ii]
                        # it follows W[gib] -= delta * X[ii]
                        if shrinking == 1:
                            data_shrink_state[ii2] = 0

                        if gia == gib:
                            if log_changes == 1:
                                indicesp = indicesp1
                                while indicesp != indicesp2:
                                    changed_mask_a[DR(indicesp)] = 1
                                    PI(indicesp)

                            idxp2 = &idx_tmp[idx_tmp_i]
                            indicesp, datap = indicesp1, datap1
                            while indicesp != indicesp2:
                                ci, a = 0, DR(datap)
                                idxp = &idx_tmp[grb0]
                                wp = W_a+DR(indicesp)*Wxdim
                                wp2 = &class_tmp[grb0]
                                wp[yi_rel] += tmp_y * a
                                while idxp != idxp2:
                                    wp[DR(PI(idxp))] -= DR(PI(wp2))*a
                                PI(indicesp); PI(datap)
                        else:
                            if log_changes == 1:
                                indicesp = indicesp1
                                while indicesp != indicesp2:
                                    changed_mask_a[DR(indicesp)] = 1
                                    changed_mask_b[DR(indicesp)] = 1
                                    PI(indicesp)

                            indicesp, datap = indicesp1, datap1
                            while indicesp != indicesp2:
                                a = DR(datap)
                                wp = W_a+DR(indicesp)*Wxdim
                                wp[yi_rel] += tmp_y * a
                                PI(indicesp); PI(datap)
                        
                            idxp2 = &idx_tmp[idx_tmp_i]
                            indicesp, datap = indicesp1, datap1
                            while indicesp != indicesp2:
                                ci, a = 0, DR(datap)
                                idxp = &idx_tmp[grb0]
                                wp = W_b+DR(indicesp)*Wxdim
                                wp2 = &class_tmp[grb0]
                                while idxp != idxp2:
                                    wp[DR(PI(idxp))] -= DR(PI(wp2))*a
                                PI(indicesp); PI(datap)
                    elif shrinking == 1 and idx_tmp_i < grb1:
                        data_shrink_state[ii2] += 1
                        if data_shrink_state[ii2] >= shrink_state:
                            data_shrink_state[ii2] = shrinked_start_state

                class_optimal[gia] = optimal
                class_max_violation[gia] = max_violation
    pass
