import datetime
import json
import numpy as np
import os
import scipy
import sys
import time
import traceback

from ..utils import base as ubase
from ..utils import log as ulog
from ..utils import mpi as umpi
from . import cython as solvers_cython

__all__ = [
    "BaseXMCSolver",

    "FitWithUpdateMixin",
    "DefaultSetupMixin",
]


###############################################################################


CACHE_LINE_SIZE = 64
STATS_FILE_NAME = "model_stats.json"
WEIGHT_FILE_NAME = "weight_vector_%i.npy"

DTYPES = {
    None: np.float64,
    "float32": np.float32,
    "float64": np.float64,
}

IDTYPES = {
    None: np.uint64,
    "uint32": np.uint32,
    "uint64": np.uint64,
}


###############################################################################


class BaseXMCSolver(object):
    """
    Base class for the xmc solvers. It provides basic routines for
    fitting, predicting, modle un- and serializing.

    A subclass should overwrite the _fit method and can overwrite
    the _init_sys, _setup, _teardown methods.
    """

    def __init__(
            self,
            max_iter=1000,
            C=1,
            epsilon=0.1,
            shuffle=True,
            seed=1,

            verbosity=1,
            dtype=None,
            idtype=None,
            nr_threads=1,
            cache_alignment=True,
            print_intervall=60,
            mode="testing",
            build_dir=None,
            mpi_comm=None,
    ):
        # todo: remove these at teardown?
        self._mpi = umpi.MPIHelper(mpi_comm)
        self._log = ulog.MPILogger(verbosity, self._mpi.rank())

        self.dtype = DTYPES[dtype]
        self.idtype = IDTYPES[idtype]
        self.dtype_size = self.dtype(0).nbytes
        self.idtype_size = self.idtype(0).nbytes

        self.max_iter = max_iter
        self.C = C
        self.epsilon = self.dtype(epsilon)
        self.shuffle = shuffle
        self.seed = seed

        self.nr_threads = nr_threads
        self.cache_alignment = False
        if cache_alignment is not None:
            self.cache_alignment = bool(cache_alignment)
        self.print_intervall = print_intervall
        if mode not in ("testing",
                        "profiling",
                        "line_profiling",
                        "production"):
            raise ValueError("Mode %s is not valid." % mode)
        self.mode = mode

        self._log.info("Solver uses %s mode." % mode)
        ubase.setup_cython(self.dtype, self.idtype,
                           self._mpi,
                           mode=mode, build_dir=build_dir)

        self._mem_allocation_info = {}
        pass

    def _register_mem_allocation(self, name, shape, dtype):
        if name is not None:
            size = np.prod(shape) * dtype(1).nbytes
            if name not in self._mem_allocation_info:
                self._mem_allocation_info[name] = 0
            self._mem_allocation_info[name] += size    
        pass

    def _log_mem_allocation(self):
        self._log.info("Mem allocations:")
        total_size = 0
        for k in sorted(self._mem_allocation_info.keys()):
            size = self._mem_allocation_info[k]
            total_size += size
            self._log.debug("  * %s: %.3f MB" % (k, size/(1024.0**2)))
        self._log.info("  Total: %.3f MB" % (total_size/(1024.0**2)))
        pass

    def _zeros_with_cache_aligned_rows(self, name, shape, dtype):
        if self.cache_alignment is True:
            assert len(shape) == 2
            shape = list(shape)
            nbytes = dtype(1).nbytes
            missing_size = (shape[1]*nbytes) % CACHE_LINE_SIZE
            shape[1] += int(np.ceil(missing_size / nbytes))
            # todo check if cache lines are actually aligned!
        self._register_mem_allocation(name, shape, dtype)
        return np.zeros(shape, dtype=dtype)

    def _init_sys(self):
        self._mpi_rank = self._mpi.rank()
        self._mpi_size = self._mpi.size()

        self._base_c = solvers_cython.get_module("base",
                                                 self.dtype, self.idtype)

        if("OMP_NUM_THREADS" in os.environ and
           int(os.environ["OMP_NUM_THREADS"]) != self.nr_threads):
            self._log.warn("Environment variable OMP_NUM_THREADS ignored."
                           " Use parameter nr_threads!")

        self._openmp_threads = self.nr_threads
        self._base_c.openmp_set_num_threads(self.nr_threads)

        self._log.info("Using %i MPI processes and %i MP threads."
                       % (self._mpi_size, self._openmp_threads))
        try:
            super(BaseXMCSolver, self)._init_sys()
        except AttributeError as e:
            pass
        pass

    def _init_fit(self, X, y):
        self._log.info("Use C=%g and epsilon=%g" %
                       (self.C, self.epsilon))

        np.random.seed(self.seed)
        self._init_sys()
        self._all_classes = np.unique(y)
        np.random.shuffle(self._all_classes)
        self._n_samples = X.shape[0]
        self._dimensions = X.shape[1]
        nnz = self._dimensions
        if hasattr(X, "nnz"):
            nnz = X.nnz/float(self._n_samples)
            self._log.debug("Samples: %i, Dimensions: %i "
                            "(NNZ-Dim.: %.1f), Classes: %i"
                            % (self._n_samples, self._dimensions,
                               nnz, self._all_classes.size))
        pass

    def _setup(self, X, y):
        try:
            super(BaseXMCSolver, self)._setup()
        except AttributeError as e:
            pass

        self._classes = self._all_classes.copy()
        pass

    def _teardown(self):
        try:
            super(BaseXMCSolver, self)._teardown()
        except AttributeError as e:
            pass
        pass

    def _fit(self):
        pass

    def fit(self, X, y):
        # default we assume that each model gets all the data.
        # special implementations might assume different patterns.

        assert X.dtype == self.dtype
        assert y.dtype == self.dtype

        self._init_fit(X, y)

        try:
            self._mpi.barrier()
            self._log.debug("Setup training.")
            self._setup(X, y)
            self._log_mem_allocation()

            self._mpi.barrier()
            self._log.debug("Start training.")
            self._fit()
        except Exception as e:
            exc_info = sys.exc_info()
            try:
                self._log.debug("Teardown training.")
                self._teardown()
            except:
                traceback.print_exc()
            traceback.print_exc()
            raise exc_info[0], exc_info[1], exc_info[2]
        else:
            try:
                self._log.debug("Teardown training.")
                self._teardown()
            except:
                traceback.print_exc()

        return self

    def predict(self, X):
        # default we assume all nodes have the same data, though
        # the weight vectors are distributed

        assert X.dtype == self.dtype

        self._init_sys()

        W = self._W
        if X.shape[1] > W.shape[1]:
            X = X[:, :W.shape[1]]
        if X.shape[1] < W.shape[1]:
            W = W[:, :X.shape[1]]
        y_pred = X.dot(W.T)

        if self._mpi_size > 1:
            msg = np.zeros((X.shape[0], 2), dtype=self.dtype)
            tmp = y_pred.argmax(axis=1)
            r = np.arange(X.shape[0])
            msg[:, 0] = self._classes[tmp]

            msg[:, 1] = y_pred[r, tmp]

            if self._mpi_rank == 0:
                msg_remote = np.empty_like(msg)
                tmp0, tmp1 = np.empty_like(msg), np.empty_like(msg)
                tmp0[:, 0] = msg[:, 0]
                tmp1[:, 0] = msg[:, 1]
                for i in xrange(1, self._mpi_size):
                    self._mpi.Recv(msg_remote, source=i)
                    tmp0[:, 1] = msg_remote[:, 0]
                    tmp1[:, 1] = msg_remote[:, 1]
                    idx = tmp1.argmax(axis=1)
                    tmp0[:, 0] = tmp0[r, idx]
                    tmp1[:, 0] = tmp1[r, idx]
                ret = tmp0[:, 0]
            else:
                self._mpi.Send(msg, dest=0)
                ret = None
        else:
            ret = self._classes[y_pred.argmax(axis=1)]
        return ret

    def accuracy(self, X, y):
        # default we assume all nodes have the same data, though
        # the weight vectors are distributed
        # in case the data is distributed too,
        # this interface would a allow a distributed testing.

        assert X.dtype == self.dtype
        assert y.dtype == self.dtype

        y_hat_global = self.predict(X)

        if self._mpi_rank == 0:
            return np.asarray(((y_hat_global == y).sum(), y.size),
                              dtype=np.int32)
        else:
            return np.asarray((0, 0), dtype=np.int32)

    def serialize(self, dir_name):
        if self._mpi_rank == 0:
            if os.path.exists(dir_name):
                raise Exception("Model directory already exists.")
            os.makedirs(dir_name)

            stats = {
                # todo: add other options
                "n_classes": self._all_classes.size,
                "n_samples": self._n_samples,
                "dimensions": self._dimensions,
                "dtype": {np.float32: "float32",
                          np.float64: "float64"}[self.dtype],
                "idtype": {np.uint32: "uint32",
                           np.uint64: "uint64"}[self.idtype],

                "zzz_classes": self._all_classes.tolist(),  # as last in file
            }
            with open(os.path.join(dir_name, STATS_FILE_NAME), "w") as f:
                json.dump(stats, f, indent=2, sort_keys=True)

        self._mpi.barrier()
        for i, c in enumerate(self._classes):
            f = os.path.join(dir_name, WEIGHT_FILE_NAME % c)
            # todo: save only needed dimensions.
            # todo: maybe save only sparys representation!
            np.save(f, self._W[i])
        pass

    def _unserialize(self, dir_name, stats):
        self._init_sys()

        self._all_classes = np.asarray(stats["zzz_classes"])
        self._n_samples = stats["n_samples"]
        self._dimensions = stats["dimensions"]

        # default each rank takes its share
        start = int(self._all_classes.size*self._mpi_rank/self._mpi_size)
        end = int(self._all_classes.size*(self._mpi_rank+1)/self._mpi_size)
        self._classes = self._all_classes[start:end]
        self._log.all_debug("Take classes %s: %s"
                            % (self._classes.size, self._classes))

        dim = self._dimensions
        if self.cache_alignment is True:
            dim += int(np.ceil(CACHE_LINE_SIZE / self.dtype(1).nbytes))

        self._W = self._zeros_with_cache_aligned_rows(
            "weights",
            (self._classes.size, dim),
            dtype=self.dtype)
        self._log.all_debug("Weight matrix size: %.3f MB"
                            % (self._W.nbytes/1024.0**2))

        for i, c in enumerate(self._classes):
            f = os.path.join(dir_name,
                             WEIGHT_FILE_NAME % c)
            self._W[i] = np.load(f)
        pass

    @staticmethod
    def unserialize(dir_name, mpi_comm=None):
        with open(os.path.join(dir_name, STATS_FILE_NAME)) as f:
            stats = json.load(f)

        ret = BaseXMCSolver(dtype=stats["dtype"],
                            idtype=stats["idtype"],
                            mpi_comm=mpi_comm)
        ret._unserialize(dir_name, stats)
        return ret


###############################################################################


class FitWithUpdateMixin(object):
    """
    This mixin provides useful code for optimization iterations. When using
    this mixin one should overwrite the _fit method and call it with a
    custom update function.

    Additionally one can provide the function _primal_dual_gap in order
    to enable printing of the gap.
    """

    def __init__(self,
                 print_primal_dual_gap=False, primal_dual_gap_samples=-1,
                 print_sparsity=False,
                 **kwargs):
        self.print_primal_dual_gap = print_primal_dual_gap
        self.primal_dual_gap_samples = primal_dual_gap_samples
        self.print_sparsity = print_sparsity
        super(FitWithUpdateMixin, self).__init__(**kwargs)

    def _primal_dual_gap(self):
        return 0

    def _sparsity(self):
        n_nz = (self._W != 0).sum()
        n_nz = self._mpi.sum(n_nz)
        n_entries = self._dimensions * len(self._all_classes)
        W_sparsity = float(n_nz)/n_entries

        alpha_sparsity = 0
        alpha_sum = 0
        if hasattr(self, "_alpha"):
            n_nz = (self._alpha != 0).sum()
            n_nz = self._mpi.sum(n_nz)
            n_entries = self._n_samples * len(self._all_classes)
            alpha_sparsity = float(n_nz)/n_entries

            alpha_sum = self._alpha.sum()
            alpha_sum = self._mpi.sum(alpha_sum)
        return W_sparsity, alpha_sparsity, alpha_sum

    def _fit(self, update_function):
        # this is a view on a cache aligned array
        # todo: all classes?
        class_max_violation = self._zeros_with_cache_aligned_rows(
            None,
            (self._all_classes.size, 1), dtype=self.dtype)[:, 0]

        start_time = time.time()
        last_print = start_time
        iter_mod = 1

        for i in xrange(1, self.max_iter+1):
            class_max_violation[:] = 0

            optimal, extra = update_function(i, class_max_violation)

            extra = "" if extra is None else extra

            # max_violation is aggregated only local as is just for debugging
            max_violation = class_max_violation.max()
            if(optimal or
               (i % iter_mod == 0) or
               ((time.time()-last_print) > self.print_intervall)):
                last_print = time.time()

                if self.print_primal_dual_gap is True:
                    gap = (self._mpi.sum(self._primal_dual_gap()) /
                           self._all_classes.size)
                    extra += ", Primal-Dual-G.: %10.6f" % gap
                if self.print_sparsity is True:
                    extra += (", W-Sp.: %8.6f a-Sp.: %8.6f a-Su.: %8.4f"
                              % self._sparsity())
                self._log.info("Iter: %10d, Max. Viol.: %10.6f, Time: %s%s"
                               % (i,
                                  max_violation,
                                  datetime.timedelta(
                                      seconds=last_print-start_time),
                                  extra))
                if i/iter_mod == 10:
                    iter_mod *= 10

            if optimal:
                break

        self._log.info("Optimization stopped after %i iterations." % i)
        sparsity = self._sparsity()
        self._log.info("Sparsity of W is: %g" % sparsity[0])
        self._log.info("Sparsity and sum of alpha are: %g - %g" % sparsity[1:])
        if i == self.max_iter:
            self._log.warn("Reached maximum number of iterations: %i" % i)
        pass


class DefaultSetupMixin(object):
    """
    This mixin provides useful routines for setting up the training.
    I.e. classes are distributed over the different mpi ranks,
    important data properties are extracted.
    """

    def __init__(self, remove_zero_K_samples=False,
                 removed_size_array_typo_error=True, **kwargs):
        self.remove_zero_K_samples = remove_zero_K_samples
        self.removed_size_array_typo_error = removed_size_array_typo_error
        super(DefaultSetupMixin, self).__init__(**kwargs)

    def _setup_class_distribution(self, X, y):
        if not isinstance(X, scipy.sparse.csr_matrix):
            raise ValueError("Expect X as scipy.sparse.csr_matrix.")

        self._all_class_sizes = np.asarray([(y == c).sum()
                                            for c in self._all_classes])
        idx = np.argsort(self._all_class_sizes)[::-1]
        self._all_classes = self._all_classes[idx]
        # todo: keep only correct version after testing
        if self.removed_size_array_typo_error is True:
            self._all_class_sizes = self._all_class_sizes[idx]
        else:
            self._all_class_size = self._all_class_sizes[idx]

        # each mpi rank takes its share
        # distribute approx by size
        self._classes = self._all_classes[self._mpi_rank::self._mpi_size]
        self._class_sizes = self._all_class_sizes[self._mpi_rank::
                                                  self._mpi_size]

        self._log.all_debug("Take classes %s: %s(%s)"
                            % (self._classes.size,
                               self._classes[:25],
                               self._class_sizes[:25]))

        self._W_train_shape = (self._classes.size, self._dimensions)
        pass

    def _teardown_class_distribution(self):
        pass

    def _setup_data_distribution(self, X, y):
        self._X = X
        self._y = y

        self._yi = np.empty((y.shape[0],), self.idtype)
        self._yi_local = (np.ones((y.shape[0],), self.idtype) *
                          self._all_classes.size)
        for i, c in enumerate(self._all_classes):
            self._yi[self._y == c] = i
        for i, c in enumerate(self._classes):
            self._yi_local[self._y == c] = i

        tmp = self._X.copy()
        tmp.data **= 2
        self._K = np.asarray(tmp.sum(axis=1)).flatten()

        if self.remove_zero_K_samples is True:
            idx = self._K != 0
            removed = self._K.shape[0]-idx.sum()
            self._X = self._X[idx]
            self._y = self._y[idx]
            self._yi = self._yi[idx]
            self._yi_local = self._yi_local[idx]
            self._K = self._K[idx]
            self._log.info("Remove %i samples where K is 0." % removed)

        self._n_samples = X.shape[0]
        r = np.arange(self._X.shape[0], dtype=self.idtype)
        self._idx = r.copy()

        self._alpha_shape = (self._classes.size, self._X.shape[0])
        pass

    def _teardown_data_distribution(self):
        del self._X
        del self._y
        del self._yi
        del self._yi_local
        del self._K
        del self._idx
        pass

    def _setup_train_model(self):
        self._W = self._zeros_with_cache_aligned_rows(
            "weights", self._W_train_shape, dtype=self.dtype)

        self._alpha = self._zeros_with_cache_aligned_rows(
            "alpha", self._alpha_shape, dtype=self.dtype)
        pass

    def _teardown_train_model(self):
        # remove eventually added dimensions
        s = self._W_train_shape
        if len(s) == 2:
            self._W = self._W[:s[0], :s[1]]
        elif len(s) == 3:
            self._W = self._W[:s[0], :s[1], :s[2]]
        else:
            raise NotImplementedError()

        del self._W_train_shape

        del self._alpha
        del self._alpha_shape
        pass

    def _setup(self, X, y):
        try:
            super(DefaultSetupMixin, self)._setup(X, y)
        except AttributeError as e:
            pass

        self._setup_class_distribution(X, y)
        self._setup_data_distribution(X, y)
        self._setup_train_model()
        pass

    def _teardown(self):
        try:
            super(DefaultSetupMixin, self)._teardown()
        except AttributeError as e:
            pass

        self._teardown_train_model()
        self._teardown_data_distribution()
        self._teardown_class_distribution()

        # the final model should have shape:
        #  (self._classes.size, self._dimensions)
        assert self._W.shape == (self._classes.size, self._dimensions)
        pass
