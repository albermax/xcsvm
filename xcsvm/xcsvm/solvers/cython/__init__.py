import importlib
import numpy as np


def get_module(name, dtype=None, idtype=None):
    dtypes = {None: "f64",
              np.float32: "f32",
              np.float64: "f64"}[dtype]
    idtypes = {None: "ui64",
               np.uint32: "ui32",
               np.uint64: "ui64"}[idtype]

    module_name = ".dtype_%s_idtype_%s.%s" % (dtypes, idtypes, name)

    ret = importlib.import_module(module_name, __name__)
    print dtype, idtype, module_name, ret
    return ret