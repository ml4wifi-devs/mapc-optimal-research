import pickle
import pulp as plp
from lz4.frame import compress, decompress

SOLVERS = {
    "pulp": plp.PULP_CBC_CMD,
    "scip": plp.SCIP_CMD,
    "glpk": plp.GLPK_CMD,
    "choco": plp.CHOCO_CMD,
    "gurobi": plp.GUROBI_CMD,
    "cplex": plp.CPLEX_CMD,
    "highs": plp.HiGHS_CMD
}


def save(obj, filename):
    serialized_obj = pickle.dumps(obj)
    compressed_obj = compress(serialized_obj)
    with open(filename, 'wb') as f:
        f.write(compressed_obj)

def load(filename):
    with open(filename, 'rb') as f:
        compressed_obj = f.read()
    decompressed_obj = decompress(compressed_obj)
    obj = pickle.loads(decompressed_obj)
    return obj
