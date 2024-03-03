"""
Debugging problem of Nones in dual variables of the Main problem. It turns out,
that many of the solvers do not support dual variables by default. I do not know
how to enable them in any of the solvers, but I know that COPT_DLL() does support.
"""

import pulp as plp
from argparse import ArgumentParser
from mapc_research.utils.copt_pulp import COPT_DLL

SOLVERS = {
    "pulp": plp.PULP_CBC_CMD,
    "copt": COPT_DLL,
    "scip": plp.SCIP_CMD,
    "glpk": plp.GLPK_CMD,
    "choco": plp.CHOCO_CMD,
    "gurobi": plp.GUROBI_CMD,
    "cplex": plp.CPLEX_CMD,
    "highs": plp.HiGHS_CMD,
    "coin": plp.COIN_CMD,
}


def example(solver):
    # Create LP problem
    prob = plp.LpProblem("lp_ex1", plp.LpMaximize)

    # Create variables
    x = plp.LpVariable("x", 0.1, 0.6)
    y = plp.LpVariable("y", 0.2, 1.5)
    z = plp.LpVariable("z", 0.3, 2.8)

    # Create constraints
    prob += 1.5*x + 1.2*y + 1.8*z <= 2.6, "c1"
    prob += 0.8*x + 0.6*y + 0.9*z >= 1.2, "c2"

    # Set objective
    prob += 1.2*x + 1.8*y + 2.1*z, "obj"

    # Solve problem
    gurobi_options=[("PreDual", 2), ("QCPDual", 1)]
    prob.solve(solver(keepFiles=True))

    # Print solution
    print("\nOptimization status:", plp.LpStatus[prob.status])
    print("Objective value:", plp.value(prob.objective))

    print("Variable solution:")
    for v in prob.variables():
        print(v.name, "=", v.varValue)
    
    print("\nConstraints:")
    for name, c in prob.constraints.items():
        print(name, ":", c, "pi =", c.pi, "slack =", c.slack)

        # Print all attributes of the constraint
        # print("\n  Attributes of constraint:")
        # for attr in dir(c):
        #     print(attr, ":", getattr(c, attr))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-s", "--solver", type=str, default="pulp", choices=SOLVERS.keys())
    args = parser.parse_args()
    solver = SOLVERS[args.solver]

    # Run example
    example(solver)