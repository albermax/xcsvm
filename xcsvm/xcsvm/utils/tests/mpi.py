
import unittest


__all__ = ["wrap"]


def wrap(n):
    def dec(func):
        def test_f(*args, **kwargs):
            msg = ("MPI nose testing is currently due "
                   "to a license conflict not included.")
            #assert False, msg
            raise unittest.SkipTest(msg)
        test_f.__name__ = func.__name__
        return test_f
    return dec
