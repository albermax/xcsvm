import numpy as np
import os
import sys

from nose.plugins.attrib import attr as testattr

from ... import solvers
from .. import base as ubase
from .. import log as ulog
from .. import mpi as umpi

__DEFAULT_DTYPE__ = "float64"
__DEFAULT_BUILD_DIR__ = ".pyxbld_nosetests"
__DEFAULT_TOLERANCE__ = 0.001

__DATASET_DIR__ = "tests/data"
__DATASETS__ = {
    "dryrun": (
        "lshtc1_dryrun_task1_tfidf_l2_norm_applied.train.libsvm",
        "lshtc1_dryrun_task1_tfidf_l2_norm_applied.test.libsvm"
    ),
    "glass": (
        "glass_9dim_6cl.train.libsvm",
        "glass_9dim_6cl.test.libsvm"
    ),
    "iris": (
        "iris_4dim_3cl.normalized.train.libsvm",
        "iris_4dim_3cl.normalized.test.libsvm"
    ),
    "news20": (
        "news20_62061dim_20cl.normalized.train.libsvm",
        "news20_62061dim_20cl.normalized.test.libsvm"
    ),
    "splice": (
        "splice_60dim_2cl.standardized.train.libsvm",
        "splice_60dim_2cl.standardized.test.libsvm"
    ),
}
__SMALL_DATASETS__ = (
    "glass",
    "iris",
    "news20",
    #"splice"
)
__LARGE_DATASETS__ = (
    "dryrun",
)


__get_dataset_cache__ = {}
def __get_dataset(name, dtype="float32"):
    key = (name, dtype)
    if key in __get_dataset_cache__:
        return __get_dataset_cache__[key]

    module_path = os.path.join(os.path.dirname(__file__), "..", "..")
    base_path = os.path.abspath(module_path)
    path = os.path.join(base_path, __DATASET_DIR__)

    f_train = os.path.join(path, __DATASETS__[name][0])
    f_test = os.path.join(path, __DATASETS__[name][1])

    X_train, y_train = ubase.get_data(f_train, dtype)
    X_test, y_test = ubase.get_data(f_test, dtype)

    ret = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
    __get_dataset_cache__[key] = ret
    return ret


def _do_test_datasets(solver_id, options, datasets,
                      reference_results=None,
                      mpi_comm=None,
                      tolerance=None):
    dtype = options.get("dtype", __DEFAULT_DTYPE__)

    if tolerance is None:
        tolerance = __DEFAULT_TOLERANCE__
    if "build_dir" not in options:
        options["build_dir"] = __DEFAULT_BUILD_DIR__

    log = ulog.MPILogger(1)
    mpi = umpi.MPIHelper(mpi_comm)

    ret = {}
    for ds_name in datasets:
        mpi.barrier()

        log.info("="*80)
        log.info("Test dataset: %s" % ds_name)
        log.info("."*80)

        ds = __get_dataset(ds_name, dtype=dtype)

        class_ = solvers.SOLVERS[solver_id]
        solver = class_(**options)

        solver.fit(ds["X_train"], ds["y_train"])
        y_hat = solver.predict(ds["X_test"])

        # workaround
        if y_hat is None:
            acc = 0
        else:
            acc = (ds["y_test"] == y_hat).mean()
        acc = mpi.sum(acc)
        ret[ds_name] = (y_hat, acc)

        log.info("Accuracy: %f" % acc)

        if reference_results is not None:
            epsilon = solver.epsilon
            C = solver.C

            ref = None
            try:
                # todo: find better way to work around when C
                # either float32 or float64
                ref = reference_results[ds_name][
                    float("%f" % epsilon)][float("%f" % C)]
            except:
                pass

            if ref is None:
                # when reference result is given
                # we assume one wants to use it!
                raise Exception("No reference result for test: "
                                "dataset %s; epsilon %g; C %g."
                                % (ds_name, epsilon, C))

            ref_acc = ref["accuracy"]
            s = "%f != %f" % (ref_acc, acc)
            assert np.abs(ref_acc-acc) < tolerance, s
            # todo: check also other properties
        else:
            log.warn("No reference result.")
        pass

    return ret


def _do_test_small_datasets(solver_id, options,
                            datasets=None,
                            reference_results=None,
                            mpi_comm=None,
                            tolerance=None):
    if datasets is None:
        datasets = __SMALL_DATASETS__
    else:
        datasets = set(datasets).intersection(__SMALL_DATASETS__)

    return _do_test_datasets(solver_id, options,
                             datasets,
                             reference_results,
                             mpi_comm,
                             tolerance)


def _do_test_large_datasets(solver_id, options,
                            datasets=None,
                            reference_results=None,
                            mpi_comm=None,
                            tolerance=None):
    if datasets is None:
        datasets = __LARGE_DATASETS__
    else:
        datasets = set(datasets).intersection(__LARGE_DATASETS__)

    return _do_test_datasets(solver_id, options,
                             datasets,
                             reference_results,
                             mpi_comm,
                             tolerance)
