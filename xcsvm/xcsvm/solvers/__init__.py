from .base import *
from .llwmr import *
from .ww import *

SOLVERS = {
    "llw_mr_sparse": LLW_MR_Sparse_Solver,
    "ww_sparse": WW_Sparse_Solver,
}
