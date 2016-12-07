from mpi4py import MPI


__all__ = [
    "MPIHelper"
]


###############################################################################


class MPIHelper(object):
    """
    Helps to bundle the MPI communication directed to
    a communication channel called comm.
    """

    def __init__(self, comm=None):
        if comm is None:
            comm = MPI.COMM_WORLD

        self._comm = comm

    ###########################################################################
    # Status operations.

    def rank(self):
        return self._comm.Get_rank()

    def size(self):
        return self._comm.Get_size()

    ###########################################################################
    # Broadcast operations.

    def barrier(self):
        return self._comm.Barrier()

    def bcast(self, *args, **kwargs):
        return self._comm.bcast(*args, **kwargs)

    ###########################################################################
    # Send and recv operations.

    def send(self, *args, **kwargs):
        return self._comm.send(*args, **kwargs)

    def recv(self, *args, **kwargs):
        return self._comm.recv(*args, **kwargs)

    def sendrecv(self, *args, **kwargs):
        return self._comm.sendrecv(*args, **kwargs)

    def Send(self, *args, **kwargs):
        return self._comm.Send(*args, **kwargs)

    def Recv(self, *args, **kwargs):
        return self._comm.Recv(*args, **kwargs)

    def Sendrecv(self, *args, **kwargs):
        return self._comm.Sendrecv(*args, **kwargs)

    ###########################################################################
    # Reduce operations.

    def _reduce(self, x_local, op):
        return self._comm.allreduce(x_local, op=op)

    def _reduce_at_root(self, x_local, op):
        return self._comm.reduce(x_local, op=op)

    def sum(self, x_local):
        return self._reduce(x_local, MPI.SUM)

    def sum_at_root(self, x_local):
        return self._reduce_at_root(x_local, MPI.SUM)

    def max(self, x_local):
        return self._reduce(x_local, MPI.MAX)

    def max_at_root(self, x_local):
        return self._reduce_at_root(x_local, MPI.MAX)

    def land(self, x_local):
        return self._reduce(x_local, MPI.LAND)

    def land_at_root(self, x_local):
        return self._reduce_at_root(x_local, MPI.LAND)
