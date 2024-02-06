import pulp as plp
from mapc_research.utils.copt_pulp import *

SOLVERS = {
    "pulp": plp.PULP_CBC_CMD,
    "copt": COPT_DLL,
    "scip": plp.SCIP_CMD,
    "glpk": plp.GLPK_CMD,
    "choco": plp.CHOCO_CMD
}

