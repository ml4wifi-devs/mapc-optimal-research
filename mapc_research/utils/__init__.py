import pulp as plp
from mapc_research.utils.copt_pulp import *

SOLVERS = {
    "pulp": plp.PULP_CBC_CMD,
    "copt": COPT_DLL
}

