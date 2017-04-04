from ...utils.tests import base as base
from ...utils.tests import mpi as mpit


# key order: dataset; epsilon; C
__REFERENCE_RESULTS__ = {
    "dryrun": {
         0.1: {
             0.1: {
                 "accuracy": 1-0.5759,
             },
             1: {
                 "accuracy": 1-0.5457,
             },
             10: {
                 "accuracy": 1-0.5441,
             },
         },
     },
    "glass": {
         0.001: {
             0.1: {
                 "accuracy": 1-0.3810,
             },
             1: {
                 "accuracy": 1-0.1905,
             },
             10: {
                 "accuracy": 1-0.1905,
             },
         },
     },
    "iris": {
         0.001: {
             0.1: {
                 "accuracy": 1-0.0667,
             },
             1: {
                 "accuracy": 1-0.1333,
             },
             10: {
                 "accuracy": 1-0.1333,
             },
         },
     },
    "news20": {
         0.001: {
             0.1: {
                 "accuracy": 1-0.1532,
             },
             1: {
                 "accuracy": 1-0.1480,
             },
             10: {
                 "accuracy": 1-0.1598,
             },
         },
     },
}


def do_ww_tsd(options_update={}, mpi_comm=None,
              datasets=None, reference=True):
    solver_id = "ww_sparse"
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


def do_ww_tld(options_update={}, mpi_comm=None,
              datasets=None, reference=True):
    solver_id = "ww_sparse"
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
    do_ww_tsd()


@base.testattr("slow")
def test_default_ld():
    do_ww_tld()


###############################################################################
# Parameter C and max_iter


@base.testattr("slow")
def test_C_1_sd():
    do_ww_tsd({"C": 10**0})


@base.testattr("slow")
def test_C_1_ld():
    do_ww_tld({"C": 10**0})


@base.testattr("slow")
def test_C_10_sd():
    do_ww_tsd({"C": 10**1, "max_iter": 10000})


@base.testattr("slow")
def test_C_10_ld():
    do_ww_tld({"C": 10**1})


###############################################################################
# Parameter epsilon


def test_small_epsilon_sd():
    do_ww_tsd({"epsilon": 0.0001}, reference=False)


@base.testattr("slow")
def test_small_epsilon_ld():
    do_ww_tld({"epsilon": 0.01}, reference=False)


###############################################################################
# Parameter shuffle


def test_no_shuffle_sd():
    do_ww_tsd({"shuffle": False})


@base.testattr("slow")
def test_no_shuffle_ld():
    do_ww_tld({"shuffle": False})


###############################################################################
# Parameter seed


def test_seed_12345_sd():
    do_ww_tsd({"seed": 12345})


@base.testattr("slow")
def test_seed_12345_ld():
    do_ww_tld({"seed": 12345})


###############################################################################
# Parameter dtype


@base.testattr("slow")
def test_dtype_float32_sd():
    do_ww_tsd({"dtype": "float32"})


@base.testattr("slow")
def test_dtype_float32_ld():
    do_ww_tld({"dtype": "float32"})


@base.testattr("slow")
def test_dtype_float64_sd():
    do_ww_tsd({"dtype": "float64"})


@base.testattr("slow")
def test_dtype_float64_ld():
    do_ww_tld({"dtype": "float64"})


###############################################################################
# Parameter idtype


@base.testattr("slow")
def test_idtype_uint32_sd():
    do_ww_tsd({"idtype": "uint32"})


@base.testattr("slow")
def test_idtype_uint32_ld():
    do_ww_tld({"idtype": "uint32"})


@base.testattr("slow")
def test_idtype_uint64_sd():
    do_ww_tsd({"idtype": "uint64"})


@base.testattr("slow")
def test_idtype_uint64_ld():
    do_ww_tld({"idtype": "uint64"})


###############################################################################
# Parameter nr_threads


def test_nr_threads_2_sd():
    do_ww_tsd({"nr_threads": 2})


@base.testattr("slow")
def test_nr_threads_2_ld():
    do_ww_tld({"nr_threads": 2})


def test_nr_threads_5_sd():
    do_ww_tsd({"nr_threads": 5})


@base.testattr("slow")
def test_nr_threads_5_ld():
    do_ww_tld({"nr_threads": 5})


###############################################################################
# .............................................................................
# WW SOLVER TESTS
# .............................................................................
###############################################################################


###############################################################################
# Parameter folds


def test_folds_2_sd():
    do_ww_tsd({"folds": 2})


@base.testattr("slow")
def test_folds_2_ld():
    do_ww_tld({"folds": 2})


def test_folds_5_sd():
    do_ww_tsd({"folds": 5})


@base.testattr("slow")
def test_folds_5_ld():
    do_ww_tld({"folds": 5})


###############################################################################
# Parameter variant


def test_variant_0_sd():
    do_ww_tsd({"variant": 0})


@base.testattr("slow")
def test_variant_0_ld():
    do_ww_tld({"variant": 0})


###############################################################################
# Parameter group count


def test_group_count_2_sd():
    do_ww_tsd({
        "group_count": 2,
    })


@base.testattr("slow")
def test_group_count_2_ld():
    do_ww_tld({
        "group_count": 2,
    })


def test_group_count_3_sd():
    do_ww_tsd({
        "group_count": 3,
    })


@base.testattr("slow")
def test_group_count_3_ld():
    do_ww_tld({
        "group_count": 3,
    })


def test_group_count_4_sd():
    do_ww_tsd({
        "group_count": 4,
    })


@base.testattr("slow")
def test_group_count_4_ld():
    do_ww_tld({
        "group_count": 4,
    })


###############################################################################
# Parameter grouping citeria


def test_grouping_criteria_classes_sd():
    do_ww_tsd({
        "group_count": 4,
        "grouping_criteria": "classes",
    })


@base.testattr("slow")
def test_grouping_citeria_classes_ld():
    do_ww_tld({
        "group_count": 4,
        "grouping_criteria": "classes",
    })


def test_grouping_criteria_samples_sd():
    do_ww_tsd({
        "group_count": 4,
        "grouping_criteria": "samples",
    })


@base.testattr("slow")
def test_grouping_citeria_samples_ld():
    do_ww_tld({
        "group_count": 4,
        "grouping_criteria": "samples",
    })


def test_grouping_criteria_samples_x_classes_sd():
    do_ww_tsd({
        "group_count": 4,
        "grouping_criteria": "samples_x_classes",
    })


@base.testattr("slow")
def test_grouping_citeria_samples_x_classes_ld():
    do_ww_tld({
        "group_count": 4,
        "grouping_criteria": "samples_x_classes",
    })


###############################################################################
# Parameter grouping shuffle sizes


def test_grouping_shuffle_sizes_classes_sd():
    do_ww_tsd({
        "group_count": 4,
        "grouping_criteria": "classes",
        "grouping_shuffle_sizes": True,
    })


@base.testattr("slow")
def test_grouping_shuffle_sizes_classes_ld():
    do_ww_tld({
        "group_count": 4,
        "grouping_criteria": "classes",
        "grouping_shuffle_sizes": True,
    })


def test_grouping_shuffle_sizes_samples_sd():
    do_ww_tsd({
        "group_count": 4,
        "grouping_criteria": "samples",
        "grouping_shuffle_sizes": True,
    })


@base.testattr("slow")
def test_grouping_shuffle_sizes_samples_ld():
    do_ww_tld({
        "group_count": 4,
        "grouping_criteria": "samples",
        "grouping_shuffle_sizes": True,
    })


def test_grouping_shuffle_sizes_samples_x_classes_sd():
    do_ww_tsd({
        "group_count": 4,
        "grouping_criteria": "samples_x_classes",
        "grouping_shuffle_sizes": True,
    })


@base.testattr("slow")
def test_grouping_shuffle_sizes_samples_x_classes_ld():
    do_ww_tld({
        "group_count": 4,
        "grouping_criteria": "samples_x_classes",
        "grouping_shuffle_sizes": True,
    })


###############################################################################
# Spreading computation with openmp


def test_nr_threads_2_sd():
    do_ww_tsd({
        "group_count": 4,
        "nr_threads": 2,
    })


@base.testattr("slow")
def test_nr_threads_2_ld():
    do_ww_tld({
        "group_count": 4,
        "nr_threads": 2,
    })


def test_nr_threads_5_sd():
    do_ww_tsd({
        "group_count": 10,
        "nr_threads": 5,
    })


@base.testattr("slow")
def test_nr_threads_5_ld():
    do_ww_tld({
        "group_count": 10,
        "nr_threads": 5,
    })


###############################################################################
# Parameter shrinking


def test_shrinking_1_sd():
    do_ww_tsd({"shrinking": 1})


@base.testattr("slow")
def test_shrinking_1_ld():
    do_ww_tld({"shrinking": 1})


###############################################################################
# .............................................................................
# WW SOLVER MPI TESTS
# .............................................................................
###############################################################################


###############################################################################
# Spreading computation with openmpi


@mpit.wrap(2)
def test_nr_proc_2_sd(comm):
    do_ww_tsd(
        {
            "group_count": 4,
        },
        comm,
        # need at least 4 classes
        datasets=["glass", "news20"]
    )


@base.testattr("slow")
@mpit.wrap(2)
def test_nr_proc_2_ld(comm):
    do_ww_tld({
        "group_count": 4,
    }, comm)


@mpit.wrap(3)
def test_nr_proc_3_sd(comm):
    do_ww_tsd(
        {
            "group_count": 6,
        },
        comm,
        # need at least 6 classes
        datasets=["glass", "news20"]
    )


@base.testattr("slow")
@mpit.wrap(3)
def test_nr_proc_3_ld(comm):
    do_ww_tld({
        "group_count": 6,
    }, comm)


###############################################################################
# Parameter shrinking and MPI


@mpit.wrap(3)
def test_nr_proc_3_sd(comm):
    do_ww_tsd(
        {
            "group_count": 6,
            "shrinking": 1,
        },
        comm,
        # need at least 6 classes
        datasets=["glass", "news20"]
    )


@base.testattr("slow")
@mpit.wrap(3)
def test_nr_proc_3_ld(comm):
    do_ww_tld({
        "group_count": 6,
        "shrinking": 1,
    }, comm)


###############################################################################
# Parameter reduced_mem_allocation


@mpit.wrap(2)
def test_reduced_mem_allocation_sd(comm):
    do_ww_tsd(
        {
            "reduce_mem_allocation": True,
            "group_count": 4,
        },
        comm,
        # need at least 4 classes
        datasets=["glass", "news20"]
    )


@base.testattr("slow")
@mpit.wrap(2)
def test_reduced_mem_allocation_mpi_ld(comm):
    do_ww_tld({
        "reduce_mem_allocation": True,
        "group_count": 16,
    }, comm)


###############################################################################
# Parameter inner_repeat


@mpit.wrap(2)
def test_inner_repeat_sd(comm):
    do_ww_tsd(
        {
            "group_count": 4,
            "inner_repeat": 2,
        },
        comm,
        # need at least 4 classes
        datasets=["glass", "news20"]
    )


@base.testattr("slow")
@mpit.wrap(2)
def test_inner_repeat_ld(comm):
    do_ww_tld({
        "group_count": 4,
        "inner_repeat": 2,
    }, comm)


###############################################################################
# Parameter shuffle_rounds


@mpit.wrap(2)
def test_shuffle_rounds_False_sd(comm):
    do_ww_tsd(
        {
            "group_count": 4,
            "shuffle_rounds": False,
        },
        comm,
        # need at least 4 classes
        datasets=["glass", "news20"]
    )


@base.testattr("slow")
@mpit.wrap(2)
def test_shuffle_rounds_False_ld(comm):
    do_ww_tld({
        "group_count": 4,
        "shuffle_rounds": False,
    }, comm)


###############################################################################
# Parameter mpi_com_switch


@mpit.wrap(2)
def test_mpi_com_switch_False_sd(comm):
    do_ww_tsd(
        {
            "group_count": 4,
            "mpi_com_switch": False,
        },
        comm,
        # need at least 4 classes
        datasets=["glass", "news20"]
    )


@base.testattr("slow")
@mpit.wrap(2)
def test_mpi_com_switch_False_ld(comm):
    do_ww_tld({
        "group_count": 4,
        " mpi_com_switch": False,
    }, comm)


###############################################################################
# Parameter mpi_send_sparse


@mpit.wrap(2)
def test_mpi_send_sparse_False_sd(comm):
    do_ww_tsd(
        {
            "group_count": 4,
            "mpi_send_sparse": False,
        },
        comm,
        # need at least 4 classes
        datasets=["glass", "news20"]
    )


@base.testattr("slow")
@mpit.wrap(2)
def test_mpi_send_sparse_False_ld(comm):
    do_ww_tld({
        "group_count": 4,
        "mpi_send_sparse": False,
    }, comm)


###############################################################################
# Parameter mpi_fast_sparsify


@mpit.wrap(2)
def test_mpi_fast_sparsify_False_sd(comm):
    do_ww_tsd(
        {
            "group_count": 4,
            "mpi_fast_sparsify": False,
        },
        comm,
        # need at least 4 classes
        datasets=["glass", "news20"]
    )


@base.testattr("slow")
@mpit.wrap(2)
def test_mpi_fast_sparsify_False_ld(comm):
    do_ww_tld({
        "group_count": 4,
        "mpi_fast_sparsify": False,
    }, comm)


###############################################################################
# Parameter mpi_send_changes_back


@mpit.wrap(2)
def test_mpi_send_changes_back_False_sd(comm):
    do_ww_tsd(
        {
            "group_count": 4,
            "mpi_send_changes_back": False,
        },
        comm,
        # need at least 4 classes
        datasets=["glass", "news20"]
    )


@base.testattr("slow")
@mpit.wrap(2)
def test_mpi_send_changes_back_False_ld(comm):
    do_ww_tld({
        "group_count": 4,
        "mpi_send_changes_back": False,
    }, comm)


###############################################################################
# Parameter mpi_cache_sparsity


@mpit.wrap(2)
def test_mpi_cache_sparsity_with_check_sd(comm):
    do_ww_tsd(
        {
            "group_count": 4,
            "mpi_cache_sparsity": True,
            "mpi_cache_sparsity_check": True,
        },
        comm,
        # need at least 4 classes
        datasets=["glass", "news20"]
    )


@base.testattr("slow")
@mpit.wrap(2)
def test_mpi_cache_sparsity_with_check_ld(comm):
    do_ww_tld({
        "group_count": 4,
        "mpi_cache_sparsity": True,
        "mpi_cache_sparsity_check": True,
    }, comm)


@mpit.wrap(2)
def test_mpi_cache_sparsity_False_sd(comm):
    do_ww_tsd(
        {
            "group_count": 4,
            "mpi_cache_sparsity": False,
        },
        comm,
        # need at least 4 classes
        datasets=["glass", "news20"]
    )


@base.testattr("slow")
@mpit.wrap(2)
def test_mpi_cache_sparsity_False_ld(comm):
    do_ww_tld({
        "group_count": 4,
        "mpi_cache_sparsity": False,
    }, comm)


###############################################################################
# Parameter mpi_local_folds


@mpit.wrap(2)
def test_mpi_local_folds_False_sd(comm):
    do_ww_tsd(
        {
            "group_count": 4,
            "mpi_local_folds": False,
            "folds": 3,
        },
        comm,
        # need at least 4 classes
        datasets=["glass", "news20"]
    )


@base.testattr("slow")
@mpit.wrap(2)
def test_mpi_local_folds_False_ld(comm):
    do_ww_tld({
        "group_count": 4,
        "mpi_local_folds": False,
        "folds": 3,
    }, comm)


@mpit.wrap(2)
def test_mpi_local_folds_sd(comm):
    do_ww_tsd(
        {
            "group_count": 4,
            "mpi_local_folds": True,
            "folds": 3,
        },
        comm,
        # need at least 4 classes
        datasets=["glass", "news20"]
    )


@base.testattr("slow")
@mpit.wrap(2)
def test_mpi_local_folds_ld(comm):
    do_ww_tld({
        "group_count": 4,
        "mpi_local_folds": True,
        "folds": 3,
    }, comm)
