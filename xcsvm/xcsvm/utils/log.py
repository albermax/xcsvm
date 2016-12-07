import logging
import os
import sys


__all__ = [
    "MPILogger"
]


###############################################################################


LOGGER_NAME = "xcsvm_rank_%s"
LOGGING_FORMAT = "%(asctime)s - %(filename)10s -" +\
                 " %(levelname)7s: %(message)s"
# LOGGING_FORMAT = "%(asctime)s - %(filename)10s - " +\
#                  "%(funcName)10s - %(levelname)7s: %(message)s"


###############################################################################


class MPILogger(object):
    """
    Logger for an MPI environment.

    This logger eats all log messages that are not issued
    by mpi rank 0, except the special log functions prefixed with all_
    are used.

    Additionally, the logger reports the filenname of the caller.
    """

    def __init__(self, level=None, rank=None, custom=None):
        if level is None:
            level = logging.WARNING
        else:
            if level in [0, 1, 2]:
                level = logging.WARNING - level * 10

        rank_str = rank
        if rank_str is None:
            rank_str = "X"
        format = "[Rank %s] %s" % (rank_str, LOGGING_FORMAT)

        logger_name = LOGGER_NAME
        if custom is not None:
            logger_name = custom
        logger = self._get_logger(LOGGER_NAME % rank)

        # todo: detect nose log handler and overwrite format.
        if len(logger.handlers) == 0:
            logger.setLevel(level)

            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)

            formatter = logging.Formatter(format)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        self._logger = logger
        if rank is None:
            rank = 0
        self._rank = rank
        pass

    class CustomLogger(logging.Logger):
        def findCaller(self):
            """
            Find the stack frame of the caller so that we can note the source
            file name, line number and function name.
            """
            f = sys._getframe(3)
            # On some versions of IronPython, currentframe() returns None if
            # IronPython isn't run with -X:Frames.
            if f is not None:
                f = f.f_back

            _srcfiles = [os.path.normcase(__file__),
                         logging._srcfile]

            rv = "(unknown file)", 0, "(unknown function)"
            while hasattr(f, "f_code"):
                co = f.f_code
                filename = os.path.normcase(co.co_filename)
                if filename in _srcfiles:
                    f = f.f_back
                    continue
                rv = (co.co_filename, f.f_lineno, co.co_name)
                break
            return rv

    def _get_logger(self, name):
        old_class = logging.getLoggerClass()
        logging.setLoggerClass(MPILogger.CustomLogger)
        logger = logging.getLogger(name)
        logging.setLoggerClass(old_class)
        return logger

    ###########################################################################
    # Standard logging functions. Log only at mpi rank 0.

    def log(self, *args, **kwargs):
        if self._rank != 0:
            return None
        return self._logger.log(*args, **kwargs)

    def debug(self, *args, **kwargs):
        if self._rank != 0:
            return None
        return self._logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        if self._rank != 0:
            return None
        return self._logger.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        if self._rank != 0:
            return None
        return self._logger.warn(*args, **kwargs)

    def error(self, *args, **kwargs):
        if self._rank != 0:
            return None
        return self._logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        if self._rank != 0:
            return None
        return self._logger.critical(*args, **kwargs)

    ###########################################################################
    # Standard logging functions. Log on all mpi ranks.

    def all_log(self, *args, **kwargs):
        return self._logger.log(*args, **kwargs)

    def all_debug(self, *args, **kwargs):
        return self._logger.debug(*args, **kwargs)

    def all_info(self, *args, **kwargs):
        return self._logger.info(*args, **kwargs)

    def all_warn(self, *args, **kwargs):
        return self._logger.warn(*args, **kwargs)

    def all_error(self, *args, **kwargs):
        return self._logger.error(*args, **kwargs)

    def all_critical(self, *args, **kwargs):
        return self._logger.critical(*args, **kwargs)
