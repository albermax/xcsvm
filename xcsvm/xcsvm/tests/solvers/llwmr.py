from ...utils.tests import base as base
from ...utils.tests import mpi as mpit


# key order: dataset; epsilon; C
__REFERENCE_RESULTS__ = {
    "dryrun": {
         0.1: {
             0.1: {
                 "accuracy": 1-0.9300,
             },
             1: {
                 "accuracy": 1-0.9300,
             },
             10: {
                 "accuracy": 1-0.9300,
             },
         },
     },
    "glass": {
         0.001: {
             0.1: {
                 "accuracy": 1-0.6667,
             },
             1: {
                 "accuracy": 1-0.6190,
             },
             10: {
                 "accuracy": 1-0.3333,
             },
         },
     },
    "iris": {
         0.001: {
             0.1: {
                 "accuracy": 1-0.1333,
             },
             1: {
                 "accuracy": 1-0.2667,
             },
             10: {
                 "accuracy": 1-0.2667,
             },
         },
     },
    "news20": {
         0.001: {
             0.1: {
                 "accuracy": 1-0.2923,
             },
             1: {
                 "accuracy": 1-0.2297,
             },
             10: {
                 "accuracy": 1-0.1615,
             },
         },
     },
}


def do_llwmr_tsd(options_update={}, mpi_comm=None,
                 datasets=None, reference=True):
    solver_id = "llw_mr_sparse"
    # default options
    options = {
        "epsilon": 0.001,
        "C": 10**-1,
        "mpi_comm": mpi_comm,
    }
    options.update(options_update)
    reference_results = __REFERENCE_RESULTS__
    if reference is False:
        reference_results = None
    base._do_test_small_datasets(solver_id, options,
                                 datasets=datasets,
                                 reference_results=reference_results,
                                 mpi_comm=mpi_comm)
    pass


def do_llwmr_tld(options_update={}, mpi_comm=None,
                 datasets=None, reference=True):
    solver_id = "llw_mr_sparse"
    # default options
    options = {
        "epsilon": 0.1,
        "C": 10**-1,
        "mpi_comm": mpi_comm,
    }
    options.update(options_update)
    tolerance = 0.01
    reference_results = __REFERENCE_RESULTS__
    if reference is False:
        reference_results = None
    base._do_test_large_datasets(solver_id, options,
                                 datasets=datasets,
                                 reference_results=reference_results,
                                 mpi_comm=mpi_comm,
                                 tolerance=tolerance)
    pass


###############################################################################
# .............................................................................
# BASE SOLVER TESTS
# .............................................................................
###############################################################################


###############################################################################
# Running default config


def test_default_sd():
    do_llwmr_tsd()


@base.testattr("slow")
def test_default_ld():
    do_llwmr_tld()


###############################################################################
# Parameter C and max_iter


@base.testattr("slow")
def test_C_1_sd():
    do_llwmr_tsd({"C": 10**0})


@base.testattr("slow")
def test_C_1_ld():
    do_llwmr_tld({"C": 10**0})


@base.testattr("slow")
def test_C_10_sd():
    do_llwmr_tsd({"C": 10**1, "max_iter": 10000})


@base.testattr("slow")
def test_C_10_ld():
    do_llwmr_tld({"C": 10**1, "max_iter": 10000})


###############################################################################
# Parameter epsilon


def test_small_epsilon_sd():
    do_llwmr_tsd({"epsilon": 0.0001}, reference=False)


@base.testattr("slow")
def test_small_epsilon_ld():
    do_llwmr_tld({"epsilon": 0.01}, reference=False)


###############################################################################
# Parameter shuffle


def test_no_shuffle_sd():
    do_llwmr_tsd({"shuffle": False})


@base.testattr("slow")
def test_no_shuffle_ld():
    do_llwmr_tld({"shuffle": False})


###############################################################################
# Parameter seed


def test_seed_12345_sd():
    do_llwmr_tsd({"seed": 12345})


@base.testattr("slow")
def test_seed_12345_ld():
    do_llwmr_tld({"seed": 12345})


###############################################################################
# Parameter dtype


@base.testattr("slow")
def test_dtype_float32_sd():
    do_llwmr_tsd({"dtype": "float32"})


@base.testattr("slow")
def test_dtype_float32_ld():
    do_llwmr_tld({"dtype": "float32"})


@base.testattr("slow")
def test_dtype_float64_sd():
    do_llwmr_tsd({"dtype": "float64"})


@base.testattr("slow")
def test_dtype_float64_ld():
    do_llwmr_tld({"dtype": "float64"})


###############################################################################
# Parameter idtype


@base.testattr("slow")
def test_idtype_uint32_sd():
    do_llwmr_tsd({"idtype": "uint32"})


@base.testattr("slow")
def test_idtype_uint32_ld():
    do_llwmr_tld({"idtype": "uint32"})


@base.testattr("slow")
def test_idtype_uint64_sd():
    do_llwmr_tsd({"idtype": "uint64"})


@base.testattr("slow")
def test_idtype_uint64_ld():
    do_llwmr_tld({"idtype": "uint64"})


###############################################################################
# Parameter nr_threads


def test_nr_threads_2_sd():
    do_llwmr_tsd({"nr_threads": 2})


@base.testattr("slow")
def test_nr_threads_2_ld():
    do_llwmr_tld({"nr_threads": 2})


def test_nr_threads_5_sd():
    do_llwmr_tsd({"nr_threads": 5})


@base.testattr("slow")
def test_nr_threads_5_ld():
    do_llwmr_tld({"nr_threads": 5})


###############################################################################
# .............................................................................
# LLW SOLVER TESTS
# .............................................................................
###############################################################################


###############################################################################
# Parameter folds


def test_folds_2_sd():
    do_llwmr_tsd({"folds": 2})


@base.testattr("slow")
def test_folds_2_ld():
    do_llwmr_tld({"folds": 2})


def test_folds_5_sd():
    do_llwmr_tsd({"folds": 5})


@base.testattr("slow")
def test_folds_5_ld():
    do_llwmr_tld({"folds": 5})


###############################################################################
# Parameter variant


def test_variant_1_sd():
    do_llwmr_tsd({"variant": 1})


@base.testattr("slow")
def test_variant_1_ld():
    do_llwmr_tld({"variant": 1})


###############################################################################
# Parameter shrinking


def test_shrinking_1_sd():
    do_llwmr_tsd({"shrinking": 1})


@base.testattr("slow")
def test_shrinking_1_ld():
    do_llwmr_tld({"shrinking": 1})


###############################################################################
# Spreading computation with openmpi


@mpit.wrap(2)
def test_nr_proc_2_sd(comm):
    do_llwmr_tsd({}, comm)


@base.testattr("slow")
@mpit.wrap(2)
def test_nr_proc_2_ld(comm):
    do_llwmr_tld({}, comm)


@mpit.wrap(3)
def test_nr_proc_3_sd(comm):
    do_llwmr_tsd({}, comm)


@base.testattr("slow")
@mpit.wrap(3)
def test_nr_proc_3_ld(comm):
    do_llwmr_tld({}, comm)
