import contextlib
import cProfile
from line_profiler import LineProfiler
import numpy as np
import os
import pstats
import shutil
import sklearn.datasets
import subprocess
import sys
import time
import tempfile
import warnings


__all__ = [
    "folds",
    "heuristic_conversion",

    "setup_cython",
    "time_it",

    "get_data",

    "stdout_redirected",
    "merged_stderr_stdout",
    "capture_clib_stdoutput",
]


###############################################################################


def folds(n, s):
    fold_size = int(float(s)/n)
    missing = s - fold_size * n
    a, b = None, 0
    for i in xrange(n):
        a = b
        b = a+fold_size
        if i < missing:
            b += 1
        yield a, b
    pass


def heuristic_conversion(s):
    s = s.lower()
    try:
        return int(s)
    except:
        pass
    try:
        return float(s)
    except:
        pass
    if s == "true":
        return True
    if s == "false":
        return False
    if s == "none":
        return None
    return s


###############################################################################


def setup_cython(dtype, idtype, mpi, mode="testing", build_dir=None):
    """
    Sets up cython and recompiles it in case the data types changed.

    Depending on the mode, different compile directories are use.
    This is useful when running long tests while still developing on the same
    machine.
    """

    """
    Todo: cython when packaging.

    Keep file structure like that and same files are connected via symlinks:

    Sources not __init__ -> no include:
    xcsvm/cython_src/solvers/base.pyx

    Mirrors for different type combinations:
    xcsvm/cython/dtypefloat32idtypeuint32/include.pyx
    xcsvm/cython/dtypefloat32idtypeuint32/solvers/base.pyx

    When packaging compile with distutils. Ship .c files.
    Loading cython:

    * try to load.
      * it works -> we have an egg installed.
      * fails -> development -> install via pyximport.
    """

    # todo: this should be removed when used in an egg.

    build_dir_base = ".pyxbld_testing"
    if mode == "production":
        build_dir_base = ".pyxbld"
    if build_dir is not None:
        # overwrite
        build_dir_base = build_dir
    build_dir = os.path.join(os.path.expanduser("~"), build_dir_base)

    if mpi.rank() == 0:
        # compile first on node 0, we assume shared file system
        def f():
            import pyximport
            pyximport.install(build_dir=build_dir)

            import xcsvm.solvers.cython.dtype_f32_idtype_ui32.base
            import xcsvm.solvers.cython.dtype_f32_idtype_ui32.ww
            import xcsvm.solvers.cython.dtype_f32_idtype_ui32.llwmr

            import xcsvm.solvers.cython.dtype_f32_idtype_ui64.base
            import xcsvm.solvers.cython.dtype_f32_idtype_ui64.ww
            import xcsvm.solvers.cython.dtype_f32_idtype_ui64.llwmr

            import xcsvm.solvers.cython.dtype_f64_idtype_ui32.base
            import xcsvm.solvers.cython.dtype_f64_idtype_ui32.ww
            import xcsvm.solvers.cython.dtype_f64_idtype_ui32.llwmr

            import xcsvm.solvers.cython.dtype_f64_idtype_ui64.base
            import xcsvm.solvers.cython.dtype_f64_idtype_ui64.ww
            import xcsvm.solvers.cython.dtype_f64_idtype_ui64.llwmr
        capture_clib_stdoutput(f, print_only_on_failure=True)

        mpi.barrier()
    else:
        mpi.barrier()
        import pyximport
        pyximport.install(build_dir=build_dir)
    pass


def time_it(f, name="Function", profile=False, line_profile=False):
    if profile is True or line_profile is True:
        if line_profile is True:
            # experimental:
            import xcsvm.solvers.llw
            import xcsvm.solvers.ww
            fs = (
                xcsvm.solvers.llw_c.llw_mr_sparse_solver_updates__variant_0,
                xcsvm.solvers.llw_c.llw_mr_sparse_solver_updates__variant_1,

                xcsvm.solvers.ww_c.ww_sparse_solver_updates_local__variant_0,
                xcsvm.solvers.ww_c.ww_sparse_solver_updates_local__variant_1,
            )
            profiler = LineProfiler(*fs)
        else:
            profiler = cProfile.Profile()

        start = time.time()
        ret = profiler.runctx("f()", globals(), locals())
        end = time.time()
    else:
        start = time.time()
        ret = f()
        end = time.time()

    msg = ("%s time: %.3f s" % (name, end-start))

    if profile is True or line_profile is True:
        print "*" * 79
        print "Profile:"
        if line_profile is True:
            profiler.print_stats()
        else:
            s = pstats.Stats(profiler)
            s = s.strip_dirs().sort_stats("time").reverse_order()
            s.print_stats()
        print "*" * 79
    return ret, msg


###############################################################################


def get_data(f_name, dtype):
    if os.path.isfile(f_name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, y = sklearn.datasets.load_svmlight_file(f_name, dtype=dtype)
            if y.dtype != dtype:
                y = y.astype(dtype)
            return X, y
    if os.path.isdir(f_name):
        raise NotImplementedError()
    raise Exception("Dataset not found: %s" % f_name)


###############################################################################
# inspired by: http://stackoverflow.com/questions/
#               4675728/redirect-stdout-to-a-file-in-python


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()

    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
        stdout = sys.__stdout__

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows
    #       when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def merged_stderr_stdout():  # $ exec 2>&1
    return stdout_redirected(to=sys.__stdout__, stdout=sys.__stderr__)


def capture_clib_stdoutput(f, print_only_on_failure=False):
    with tempfile.TemporaryFile() as tmp_f:
        try:
            with stdout_redirected(to=tmp_f), merged_stderr_stdout():
                ret = f()
        except:
            tmp_f.flush()
            tmp_f.seek(0)
            print tmp_f.read()
            raise
        if print_only_on_failure is not True:
            tmp_f.flush()
            tmp_f.seek(0)
            print tmp_f.read()
    return ret
