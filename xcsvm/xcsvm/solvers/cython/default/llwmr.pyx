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
    "llw_mr_sparse_solver_updates__variant_0",
    "llw_mr_sparse_solver_updates__variant_1",
]


###############################################################################


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
cdef DTYPE_t llw_mr_subproblem(
    DTYPE_t[:, :] alpha,
    DTYPE_t K,
    np.uint8_t[:] class_optimal,
    DTYPE_t[:] class_max_violation,
    size_t ci,
    size_t ii,
    DTYPE_t a,
    DTYPE_t g,
    DTYPE_t epsilon,
    DTYPE_t C,
) nogil:
    cdef DTYPE_t delta

    if a == 0:
        if g > -epsilon:
            return 0
        delta = -g / K

        if a+delta > C:
            delta = C-a
            alpha[ci, ii] = C
        else:
            alpha[ci, ii] += delta
    elif a == C:
        if g < epsilon:
            return 0
        delta = -g / K

        if a+delta < 0:
            delta = -a
            alpha[ci, ii] = 0
        else:
            alpha[ci, ii] += delta
    else:
        if g < epsilon and g > -epsilon:
            return 0
        delta = -g / K

        if a+delta < 0:
            delta = -a
            alpha[ci, ii] = 0
        elif a+delta > C:
            delta = C-a
            alpha[ci, ii] = C
        else:
            alpha[ci, ii] += delta
            
    class_optimal[ci] = 0
    if g < 0:
        g = -g
    class_max_violation[ci] = max(class_max_violation[ci], g)

    return delta


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
cdef DTYPE_t llw_mr_subproblem_short(
    DTYPE_t[:, :] alpha,
    DTYPE_t K,
    np.uint8_t[:] class_optimal,
    DTYPE_t[:] class_max_violation,
    size_t ci,
    size_t ii,
    DTYPE_t a,
    DTYPE_t g,
    DTYPE_t epsilon,
    DTYPE_t C,
) nogil:
    cdef DTYPE_t delta

    if a == 0:
        if g > -epsilon:
            # todo in this case shrink
            return 0
    elif a == C:
        if g < epsilon:
            return 0
    else:
        if g < epsilon and g > -epsilon:
            return 0

    delta = -g / K

    if a+delta < 0:
        delta = -a
        alpha[ci, ii] = 0
    elif a+delta > C:
        delta = C-a
        alpha[ci, ii] = C
    else:
        alpha[ci, ii] += delta
            
    class_optimal[ci] = 0
    if g < 0:
        g = -g
    class_max_violation[ci] = max(class_max_violation[ci], g)

    return delta


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
def llw_mr_sparse_solver_updates__variant_0(
    object X,
    np.ndarray[IDTYPE_t] yi,
    np.ndarray[IDTYPE_t] idx,
    np.ndarray[DTYPE_t, ndim=2] alpha_,
    np.ndarray[DTYPE_t] K_,
    np.ndarray[DTYPE_t, ndim=2] W,
    np.ndarray[DTYPE_t] classes,
    np.ndarray[np.uint8_t] class_optimal_,
    np.ndarray[DTYPE_t] class_max_violation_,
    DTYPE_t epsilon,
    DTYPE_t C,

    IDTYPE_t shrinking,
    np.ndarray[np.int8_t, ndim=2] data_shrink_state,
    np.int8_t shrink_state,
    np.int8_t shrinked_start_state,
    DTYPE_t old_max_violation,
):

    cdef IDTYPE_t len_classes, len_X, i, ii, ci, j
    cdef DTYPE_t a, g, class_, delta

    cdef np.ndarray[DTYPE_t] data = X.data
    cdef np.ndarray[np.int32_t] indices = X.indices
    cdef np.ndarray[np.int32_t] indptr = X.indptr
    cdef DTYPE_t[:, :] alpha = alpha_
    cdef DTYPE_t[:] K = K_
    cdef np.uint8_t[:] class_optimal = class_optimal_
    cdef DTYPE_t[:] class_max_violation = class_max_violation_

    len_classes = len(classes)
    len_X = idx.shape[0]

    for ci in parallel.prange(len_classes,
                              nogil=True,
                              schedule="dynamic"):
    #for ci in xrange(len_classes):

        for i in xrange(len_X):
            ii = idx[i]

            if yi[ii] == ci:
                continue

            if shrinking == 1 and data_shrink_state[ci, ii] < 0:
                #data_shrink_state[ci, ii] += 1
                continue

            a = alpha[ci, ii]
            #  it follows g = -1 - X[ii].dot(W[ci])
            g = -1
            for j in xrange(indptr[ii], indptr[ii+1]):
                g = g - data[j]*W[ci, indices[j]]

            delta = llw_mr_subproblem(
                alpha,
                K[ii],
                class_optimal,
                class_max_violation,
                ci,
                ii,
                a,
                g,
                epsilon,
                C)
            if delta == 0:
                if shrinking == 1:
                    data_shrink_state[ci, ii] += 1
                    if data_shrink_state[ci, ii] >= shrink_state:
                        data_shrink_state[ci, ii] = shrinked_start_state
                continue
            if shrinking == 1:
                data_shrink_state[ci, ii] = 0

            # it follows W[ci] -= delta * X[ii]
            for j in xrange(indptr[ii], indptr[ii+1]):
                W[ci, indices[j]] -= delta * data[j]

    pass


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
def llw_mr_sparse_solver_updates__variant_1(
    object X,
    np.ndarray[IDTYPE_t] yi,
    np.ndarray[IDTYPE_t] idx,
    np.ndarray[DTYPE_t, ndim=2] alpha_,
    np.ndarray[DTYPE_t] K_,
    np.ndarray[DTYPE_t, ndim=2] W,
    np.ndarray[DTYPE_t] classes,
    np.ndarray[np.uint8_t] class_optimal_,
    np.ndarray[DTYPE_t] class_max_violation_,
    DTYPE_t epsilon,
    DTYPE_t C,


    IDTYPE_t shrinking,
    np.ndarray[np.int8_t, ndim=2] data_shrink_state,
    np.int8_t shrink_state,
    np.int8_t shrinked_start_state,
    DTYPE_t old_max_violation,
):

    cdef IDTYPE_t len_classes, len_X, i, ii, ci, j
    cdef DTYPE_t a, g, class_, delta

    cdef np.ndarray[DTYPE_t] data = X.data
    cdef np.ndarray[np.int32_t] indices = X.indices
    cdef np.ndarray[np.int32_t] indptr = X.indptr
    cdef DTYPE_t[:, :] alpha = alpha_
    cdef DTYPE_t[:] K = K_
    cdef np.uint8_t[:] class_optimal = class_optimal_
    cdef DTYPE_t[:] class_max_violation = class_max_violation_
    
    len_classes = len(classes)
    len_X = X.shape[0]


    for i in xrange(len_X):
        ii = idx[i]

        for ci in parallel.prange(len_classes,
                                  nogil=True,
                                  schedule="dynamic"):

            if yi[ii] == ci:
                continue

            if shrinking == 1 and data_shrink_state[ci, ii] < 0:
                #data_shrink_state[ci, ii] += 1
                continue

            a = alpha[ci, ii]
            #  it follows g = -1 - X[ii].dot(W[ci])
            g = -1
            for j in xrange(indptr[ii], indptr[ii+1]):
                g = g - data[j]*W[ci, indices[j]]

            delta = llw_mr_subproblem(
                alpha,
                K[ii],
                class_optimal,
                class_max_violation,
                ci,
                ii,
                a,
                g,
                epsilon,
                C)
            if delta == 0:
                if shrinking == 1:
                    data_shrink_state[ci, ii] += 1
                    if data_shrink_state[ci, ii] >= shrink_state:
                        data_shrink_state[ci, ii] = shrinked_start_state
                continue
            if shrinking == 1:
                data_shrink_state[ci, ii] = 0

            # it follows W[ci] -= delta * X[ii]
            for j in xrange(indptr[ii], indptr[ii+1]):
                W[ci, indices[j]] -= delta * data[j]

    pass
