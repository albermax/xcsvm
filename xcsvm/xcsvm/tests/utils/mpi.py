from ...utils import mpi as umpi

from ...utils.tests import mpi as mpit


###############################################################################


@mpit.wrap(3)
def test_mpi_3_nodes(comm):
    assert comm.Get_size() == 3

    mpi = umpi.MPIHelper(comm)

    assert mpi.sum(3) == 3*3
    pass


@mpit.wrap(4)
def test_mpi_4_nodes(comm):
    assert comm.Get_size() == 4

    mpi = umpi.MPIHelper(comm)

    assert mpi.sum(3) == 3*4
    pass
