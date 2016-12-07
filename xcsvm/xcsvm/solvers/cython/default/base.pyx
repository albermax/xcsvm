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
    "openmp_get_num_threads",
    "openmp_set_num_threads",

    "dist_sum_axis_0",
    "dist_subtraction_axis_0",

    "dist_set_to_zero",

    "sparsify",
    "sparsify_with_changes",
    "merge_sparse",
]


###############################################################################


def openmp_get_num_threads():
    return openmp.omp_get_max_threads()


def openmp_set_num_threads(nr_threads):
    return openmp.omp_set_num_threads(nr_threads)


###############################################################################


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
def dist_sum_axis_0(
        np.ndarray[DTYPE_t, ndim=2] x,
        np.ndarray[DTYPE_t, ndim=2] tmp,
        int nr_threads,
):
    cdef size_t chunk_size, plus_one, nr_threads_c
    cdef IDTYPE_t a, b, i, j, ci

    chunk_size = int(x.shape[0]/nr_threads)
    plus_one = x.shape[0] % nr_threads
    nr_threads_c = nr_threads

    for ci in parallel.prange(nr_threads_c,
                             nogil=True,
                             schedule="static"):
        a = ci*chunk_size + min(ci, plus_one)
        b = (ci+1)*chunk_size + min(ci+1, plus_one)

        for i in xrange(a, b):
            for j in xrange(x.shape[1]):
                tmp[ci, j] += x[i, j]

    return tmp.sum(axis=0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def dist_subtraction_axis_0(
        np.ndarray[DTYPE_t, ndim=2] x,
        np.ndarray[DTYPE_t] s,
        int nr_threads,
):
    cdef size_t chunk_size, plus_one, nr_threads_c
    cdef IDTYPE_t a, b, i, j, ci

    chunk_size = int(x.shape[0]/nr_threads)
    plus_one = x.shape[0] % nr_threads
    nr_threads_c = nr_threads

    for ci in parallel.prange(nr_threads_c,
                             nogil=True,
                             schedule="static"):
        a = ci*chunk_size + min(ci, plus_one)
        b = (ci+1)*chunk_size + min(ci+1, plus_one)

        for i in xrange(a, b):
            for j in xrange(x.shape[1]):
                x[i, j] -= s[j]
    pass


###############################################################################


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def dist_set_to_zero(
        np.ndarray[DTYPE_t] x,
        int nr_threads,
):
    cdef size_t chunk_size, plus_one, nr_threads_c
    cdef IDTYPE_t a, b, i, j, ci

    chunk_size = int(x.shape[0]/nr_threads)
    plus_one = x.shape[0] % nr_threads
    nr_threads_c = nr_threads

    for ci in parallel.prange(nr_threads_c,
                             nogil=True,
                             schedule="static"):
        a = ci*chunk_size + min(ci, plus_one)
        b = (ci+1)*chunk_size + min(ci+1, plus_one)

        for i in xrange(a, b):
            x[i] = 0
    pass


###############################################################################


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def sparsify(
        np.ndarray[DTYPE_t] x,
        np.ndarray[IDTYPE_t] idx,
        np.ndarray[DTYPE_t] data,
):
    cdef IDTYPE_t ret, i, l
    cdef DTYPE_t *xp, *xp_start, *xp_end

    l = x.size
    ret = 0
    xp_start = &x[0]
    xp = xp_start
    xp_end = xp+l

    with nogil:
        while xp != xp_end:
            if DR(xp) != 0:
                idx[ret] = xp-xp_start
                data[ret] = DR(xp)
                ret = ret+1
            xp = xp+1
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def sparsify_with_changes(
        np.ndarray[DTYPE_t] x,
        np.ndarray[np.int8_t, ndim=2] mask,
        np.ndarray[IDTYPE_t] idx,
        np.ndarray[DTYPE_t] data,
        IDTYPE_t max_group_size,
):
    cdef IDTYPE_t ret, m_size
    cdef DTYPE_t *xp, *xp_start, *xp_end
    cdef np.int8_t *m, *m_start, *m_end

    m_size = mask.size
    m_start = &mask[0, 0]
    m = m_start
    m_end = m_start+m_size
    ret = 0

    xp_start = &x[0]

    with nogil:
        while m != m_end:
            if DR(m) != 0:
                xp = xp_start+(m-m_start)*max_group_size
                xp_end = xp+max_group_size

                while xp != xp_end:
                    if DR(xp) != 0:
                        idx[ret] = xp-xp_start
                        data[ret] = DR(xp)
                        ret = ret+1
                    xp = xp+1
            m = m+1
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def merge_sparse(
        IDTYPE_t s1,
        np.ndarray[IDTYPE_t] idx1,
        np.ndarray[DTYPE_t] data1,
        IDTYPE_t s2,
        np.ndarray[IDTYPE_t] idx2,
        np.ndarray[DTYPE_t] data2,
        np.ndarray[IDTYPE_t] idxd,
        np.ndarray[DTYPE_t] datad,
):
    cdef IDTYPE_t ret, i1, i2
    cdef IDTYPE_t *idxp1, *idx1_start, *idx1_end
    cdef IDTYPE_t *idxp2, *idx2_start, *idx2_end
    cdef IDTYPE_t *idxpd
    cdef DTYPE_t *datap1, *datap2, *datapd

    idx1_start = &idx1[0]
    idxp1 = idx1_start
    idx1_end = idx1_start+s1
    datap1 = &data1[0]

    idx2_start = &idx2[0]
    idxp2 = idx2_start
    idx2_end = idx2_start+s2
    datap2 = &data2[0]

    idxpd = &idxd[0]
    datapd = &datad[0]

    ret = 0

    with nogil:
        while idxp1 != idx1_end and idxp2 != idx2_end:
            i1, i2 = DR(idxp1), DR(idxp2)
            if i1 < i2:
                idxd[ret] = DR(idxp1)
                datapd[ret] = DR(datap1)
                idxp1 += 1
                datap1 += 1
            else:
                idxd[ret] = DR(idxp2)
                datapd[ret] = DR(datap2)
                idxp2 += 1
                datap2 += 1

                if i1 == i2:
                    idxp1 += 1
                    datap1 += 1
            ret += 1

        while idxp1 != idx1_end:
            idxd[ret] = DR(idxp1)
            datapd[ret] = DR(datap1)
            idxp1 += 1
            datap1 += 1
            ret += 1

        while idxp2 != idx2_end:
            idxd[ret] = DR(idxp2)
            datapd[ret] = DR(datap2)
            idxp2 += 1
            datap2 += 1
            ret += 1

    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def merge_sparse(
        IDTYPE_t s1,
        np.ndarray[IDTYPE_t] idx1,
        np.ndarray[DTYPE_t] data1,
        IDTYPE_t s2,
        np.ndarray[IDTYPE_t] idx2,
        np.ndarray[DTYPE_t] data2,
        np.ndarray[IDTYPE_t] idxd,
        np.ndarray[DTYPE_t] datad,
):
    cdef IDTYPE_t ret, i1, i2
    cdef IDTYPE_t *idxp1, *idx1_start, *idx1_end
    cdef IDTYPE_t *idxp2, *idx2_start, *idx2_end
    cdef IDTYPE_t *idxpd
    cdef DTYPE_t *datap1, *datap2, *datapd

    idx1_start = &idx1[0]
    idxp1 = idx1_start
    idx1_end = idx1_start+s1
    datap1 = &data1[0]

    idx2_start = &idx2[0]
    idxp2 = idx2_start
    idx2_end = idx2_start+s2
    datap2 = &data2[0]

    idxpd = &idxd[0]
    datapd = &datad[0]

    ret = 0

    with nogil:
        while idxp1 != idx1_end and idxp2 != idx2_end:
            i1, i2 = DR(idxp1), DR(idxp2)
            if i1 < i2:
                idxd[ret] = DR(idxp1)
                datapd[ret] = DR(datap1)
                idxp1 += 1
                datap1 += 1
            else:
                idxd[ret] = DR(idxp2)
                datapd[ret] = DR(datap2)
                idxp2 += 1
                datap2 += 1

                if i1 == i2:
                    idxp1 += 1
                    datap1 += 1
            ret += 1

        while idxp1 != idx1_end:
            idxd[ret] = DR(idxp1)
            datapd[ret] = DR(datap1)
            idxp1 += 1
            datap1 += 1
            ret += 1

        while idxp2 != idx2_end:
            idxd[ret] = DR(idxp2)
            datapd[ret] = DR(datap2)
            idxp2 += 1
            datap2 += 1
            ret += 1

    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def merge_sparse_skip_zeros(
        IDTYPE_t s1,
        np.ndarray[IDTYPE_t] idx1,
        np.ndarray[DTYPE_t] data1,
        IDTYPE_t s2,
        np.ndarray[IDTYPE_t] idx2,
        np.ndarray[DTYPE_t] data2,
        np.ndarray[IDTYPE_t] idxd,
        np.ndarray[DTYPE_t] datad,
):
    cdef IDTYPE_t ret, i1, i2
    cdef DTYPE_t d
    cdef IDTYPE_t *idxp1, *idx1_start, *idx1_end
    cdef IDTYPE_t *idxp2, *idx2_start, *idx2_end
    cdef IDTYPE_t *idxpd
    cdef DTYPE_t *datap1, *datap2, *datapd

    idx1_start = &idx1[0]
    idxp1 = idx1_start
    idx1_end = idx1_start+s1
    datap1 = &data1[0]

    idx2_start = &idx2[0]
    idxp2 = idx2_start
    idx2_end = idx2_start+s2
    datap2 = &data2[0]

    idxpd = &idxd[0]
    datapd = &datad[0]

    ret = 0

    with nogil:
        while idxp1 != idx1_end and idxp2 != idx2_end:
            i1, i2 = DR(idxp1), DR(idxp2)
            if i1 < i2:
                idxd[ret] = DR(idxp1)
                datapd[ret] = DR(datap1)
                idxp1 += 1
                datap1 += 1
                ret += 1
            else:
                d = DR(datap2)
                if d != 0:
                    idxd[ret] = DR(idxp2)
                    datapd[ret] = DR(datap2)
                    ret += 1
                idxp2 += 1
                datap2 += 1

                if i1 == i2:
                    idxp1 += 1
                    datap1 += 1

        while idxp1 != idx1_end:
            idxd[ret] = DR(idxp1)
            datapd[ret] = DR(datap1)
            idxp1 += 1
            datap1 += 1
            ret += 1

        while idxp2 != idx2_end:
            d = DR(datap2)
            if d != 0:
                idxd[ret] = DR(idxp2)
                datapd[ret] = d
                ret += 1
            idxp2 += 1
            datap2 += 1

    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def sparse_update(
        np.ndarray[DTYPE_t] x,
        np.ndarray[IDTYPE_t] idx,
        np.ndarray[DTYPE_t] data,
        IDTYPE_t nr_threads,
):
    cdef IDTYPE_t ret, i, l
    cdef IDTYPE_t *idxp, *idxp_start, *idxp_end
    cdef DTYPE_t *datap, *datap_start, *datap_end

    l = idx.size

    idxp_start = &idx[0]
    idxp = idxp_start
    idxp_end = idxp+l

    datap_start = &data[0]
    datap = datap_start
    datap_end = datap+l

    with nogil:
        while idxp != idxp_end:
            x[DR(idxp)] = DR(datap)
            idxp += 1
            datap += 1
    pass


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def got_zero(
        np.ndarray[DTYPE_t] x,
        IDTYPE_t idx_len,
        np.ndarray[IDTYPE_t] idx,
        IDTYPE_t ridx_len,
        np.ndarray[IDTYPE_t] ridx,
        np.ndarray[IDTYPE_t] oidx,
):
    cdef IDTYPE_t ret, i, l
    cdef IDTYPE_t *idxp, *idxp_start, *idxp_end
    cdef IDTYPE_t *ridxp, *ridxp_start, *ridxp_end
    cdef IDTYPE_t *oidxp

    idxp_start = &idx[0]
    idxp = idxp_start
    idxp_end = idxp+idx_len

    ridxp_start = &ridx[0]
    ridxp = ridxp_start
    ridxp_end = ridxp+ridx_len

    oidxp = &oidx[0]

    ret = 0
    with nogil:
        while ridxp != ridxp_end:
            i = DR(ridxp)
            while idxp != idxp_end and i < DR(idxp):
                idxp += 1

            if idxp == idxp_end or i != DR(idxp):
                if x[i] == 0:
                    oidxp[ret] = i
                    ret += 1
            ridxp += 1
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def filter_zeros_sparse(
        IDTYPE_t s,
        np.ndarray[IDTYPE_t] idx,
        np.ndarray[DTYPE_t] data,
):
    cdef IDTYPE_t ret, l
    cdef DTYPE_t *datap, *datap_start, *datap_end
    cdef IDTYPE_t *idxp

    datap_start = &data[0]
    datap = datap_start
    datap_end = datap+s

    idxp = &idx[0]

    ret = 0
    with nogil:
        while datap != datap_end:
            if DR(datap) != 0:
                data[ret] = DR(datap)
                idx[ret] = DR(idxp)
                ret += 1
            datap += 1
            idxp += 1
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def filter_zeros_sparse2(
        IDTYPE_t s,
        np.ndarray[IDTYPE_t] idx,
        np.ndarray[DTYPE_t] data,
        np.ndarray[IDTYPE_t] gotzero,
):
    cdef IDTYPE_t ret, l, i
    cdef IDTYPE_t *idxp, *idxp_start, *idxp_end
    cdef IDTYPE_t *gzp, *gzp_start, *gzp_end
    cdef DTYPE_t *datap

    idxp_start = &idx[0]
    idxp = idxp_start
    idxp_end = idxp+s

    datap = &data[0]

    l = gotzero.size
    gzp_start = &gotzero[0]
    gzp = gzp_start
    gzp_end = gzp+l

    ret = 0
    with nogil:
        while idxp != idxp_end:
            if gzp != gzp_end and DR(idxp) == DR(gzp):
                gzp += 1
            else:
                data[ret] = DR(datap)
                idx[ret] = DR(idxp)
                ret += 1
            idxp += 1
            datap += 1

    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def merge_sparse2(
        IDTYPE_t s,
        np.ndarray[IDTYPE_t] idx,
        np.ndarray[DTYPE_t] data,
        np.ndarray[IDTYPE_t] gotzero,
        np.ndarray[IDTYPE_t] oi,
        np.ndarray[DTYPE_t] od,
):
    cdef IDTYPE_t ret, l, i1, i2
    cdef IDTYPE_t *idxp, *idxp_start, *idxp_end
    cdef IDTYPE_t *gzp, *gzp_start, *gzp_end
    cdef DTYPE_t *datap

    idxp_start = &idx[0]
    idxp = idxp_start
    idxp_end = idxp+s

    datap = &data[0]

    l = gotzero.size
    gzp_start = &gotzero[0]
    gzp = gzp_start
    gzp_end = gzp+l

    ret = 0
    with nogil:
        while idxp != idxp_end and gzp != gzp_end:
            i1, i2 = DR(idxp), DR(gzp)
            if i1 < i2:
                oi[ret] = i1
                od[ret] = DR(datap)
                idxp += 1
                datap += 1
            else:
                oi[ret] = i2
                od[ret] = 0
                gzp += 1
            ret += 1

        while idxp != idxp_end:
            oi[ret] = DR(idxp)
            od[ret] = DR(datap)
            idxp += 1
            datap += 1
            ret += 1

        while gzp != gzp_end:
            oi[ret] = DR(gzp)
            od[ret] = 0
            gzp += 1
            ret += 1
    return ret
